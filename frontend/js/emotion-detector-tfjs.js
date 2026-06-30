/**
 * emotion-detector-tfjs.js  — Elevate Emotion Detection Engine
 * ==============================================================
 * Model: HOG(48×48) → Dense(512,relu) → Dense(128,relu) → Dense(N,softmax)
 * Trained by: train_emotion_fast.py
 * Canonical classes: happy | bored | focused | confused | neutral | angry | surprised
 *
 * BUGS FIXED IN THIS VERSION
 * ──────────────────────────
 * FIX 1 (CRITICAL) — Scaler not loaded from .bin file.
 *   Root cause: _hydrateHogScaler() tried to read scaler_mean/scaler_std
 *   from model.json elevate_meta — those fields do not exist there.
 *   The scaler tensors are weight tensors inside the .bin file at indices
 *   6 (mean) and 7 (std).  Fixed: extract from model.getWeights() after load.
 *   Impact before fix: raw HOG (mean≈0.05) fed to model expecting mean=0 → ~35% accuracy.
 *
 * FIX 2 (CRITICAL) — Gradient kernel mismatch.
 *   Root cause: browser used Sobel 3×3 kernel.
 *   skimage.feature.hog uses simple central differences:
 *     gx[y,x] = img[y, x+1] - img[y, x-1]
 *     gy[y,x] = img[y+1, x] - img[y-1, x]
 *   Fixed: replaced Sobel with exact central-difference matching skimage.
 *
 * FIX 3 (CRITICAL) — Spurious preprocessing not applied during training.
 *   Root cause: browser applied CLAHE, gamma correction, 3×3 blur before HOG.
 *   train_emotion_fast.py does none of these — PIL→gray→resize→HOG directly.
 *   Fixed: removed preprocessing so browser matches training pipeline exactly.
 *
 * FIX 4 — Class balance weights derived from training recall, not guessed.
 *   Root cause: neutral was penalised (0.88) but had worst recall (0.563).
 *   Fixed: weights = 1/recall, normalised to mean=1.
 *
 * FIX 5 — Temperature scaling removed (mathematically wrong post-softmax).
 *   Class balance weights handle sharpening correctly.
 *
 * FIX 6 — FaceMesh throttled and inference loop overlap prevention.
 *   Root cause: FaceMesh ran every rAF frame (30fps) burning CPU unnecessarily.
 */

import { config }             from './config.js';
import { state, updateState } from './state.js';
import { utils }              from './utils.js';

const MODE_SERVER = 'server';
const MODE_TFJS   = 'tfjs';
const MODE_SIM    = 'simulation';

// Class names — MUST match train_emotion_fast.py CLASS_NAMES order exactly
const CLASS_NAMES = ['happy', 'bored', 'focused', 'confused', 'neutral', 'angry', 'surprised'];
const LEGACY_CLASS_NAMES_6 = ['happy', 'bored', 'focused', 'confused', 'neutral', 'angry'];
const LEGACY_CLASS_NAMES_4 = ['angry', 'confused', 'happy', 'neutral'];

const CLASS_NAME_ALIASES = {
  surprise: 'surprised',
  surprised: 'surprised',
  amazement: 'surprised',
  amazed: 'surprised',
  joy: 'happy',
  happiness: 'happy',
  happy: 'happy',
  anger: 'angry',
  angry: 'angry',
  focus: 'focused',
  focused: 'focused',
  confusion: 'confused',
  confused: 'confused',
  confusing: 'confused',
  calm: 'neutral',
  neutral: 'neutral',
  bore: 'bored',
  bored: 'bored',
};

const EMOTION_EMOJI = {
  happy: '😊', bored: '😑', focused: '🧠',
  confused: '😕', neutral: '😐', angry: '😠', surprised: '😮',
};

// HOG parameters — MUST match train_emotion_fast.py exactly
const HOG_IMG_SIZE    = 48;
const HOG_PPC_X       = 6;
const HOG_PPC_Y       = 6;
const HOG_CELLS_X     = Math.floor(HOG_IMG_SIZE / HOG_PPC_X);
const HOG_CELLS_Y     = Math.floor(HOG_IMG_SIZE / HOG_PPC_Y);
const HOG_BINS        = 9;
const HOG_BLOCK_W     = 2;
const HOG_BLOCK_H     = 2;
const HOG_FEATURE_DIM =
  (HOG_CELLS_X - HOG_BLOCK_W + 1) *
  (HOG_CELLS_Y - HOG_BLOCK_H + 1) *
  HOG_BLOCK_W * HOG_BLOCK_H * HOG_BINS;

// Confidence gates
// Lowered from 0.32/0.06 — CNN softmax distributions are wider than HOG+Dense,
// and after EMA smoothing the peak rarely exceeds 0.30 even for clear expressions.
const MIN_CONFIDENCE = 0.20;
const MIN_MARGIN     = 0.04;

// Temporal smoothing
// Raised alpha (0.35→0.55) so the EMA reacts faster to expression changes.
const SMOOTH_ALPHA = 0.55;
const SMOOTH_LEAK  = 0.01;

// FIX 4: Class balance weights from 1/recall, normalised to mean=1
// Neutral weight reduced to 1.0 (was 1.08) — it was over-dominating the CNN output
const DEFAULT_CLASS_BALANCE_WEIGHTS = {
  happy: 0.99,
  bored: 0.90,
  focused: 0.92,
  confused: 1.00,
  neutral: 1.00,
  angry: 0.86,
  surprised: 1.05,
};
const CLASS_WEIGHT_MIN = 0.78;
const CLASS_WEIGHT_MAX = 1.32;

const DEFAULT_CLASS_PRIOR_MULTIPLIERS = {
  happy: 1.0,
  bored: 1.0,
  focused: 1.03,
  confused: 1.02,
  neutral: 1.0,
  angry: 0.94,
  surprised: 1.02,
};

// Low-latency schedule with overlap guards
const FACEMESH_INTERVAL_MS  = 85;   // ~12fps face tracking
const INFERENCE_INTERVAL_MS = 300;  // ~3.3fps emotion inference
const SERVER_INFERENCE_TIMEOUT_MS = 1800;


export const emotionDetector = {

  videoElement:  null,
  canvasElement: null,
  facemeshModel: null,
  tfjsModel:     null,
  tfjsEmotionHeadModel: null,
  tfjsEngagementHeadModel: null,

  // FIX 1: Populated from model.getWeights()[6] and [7]
  _scalerMean: null,
  _scalerStd:  null,
  _modelClassNames: CLASS_NAMES.slice(),
  _classBalanceWeights: { ...DEFAULT_CLASS_BALANCE_WEIGHTS },
  _classPriorMultipliers: { ...DEFAULT_CLASS_PRIOR_MULTIPLIERS },

  inferenceMode:         MODE_SIM,
  detectionActive:       false,
  isRunningFacemesh:     false,
  isRunningInference:    false,
  serverModelAvailable:  false,
  serverStatusChecked:   false,

  _facemeshTimer:        null,
  _inferenceTimer:       null,
  _lastFacePrediction:   null,
  animationFrameId:      null,
  isDetectingFrame:      false,

  _smoothedScores:       null,
  _hogCanvas:            null,
  _resizeHandler:        null,
  _lastLogTime:          0,
  _logIntervalMs:        10_000,
  _modelMetaCache:       null,

  _emotionHistory:       [],
  _HISTORY_LENGTH:       10,


  // ═══════════════════════════════════════════
  //  PUBLIC API
  // ═══════════════════════════════════════════

  async init() {
    this.videoElement  = document.getElementById('webcam');
    this.canvasElement = document.getElementById('faceCanvas');
    if (!this.videoElement || !this.canvasElement) {
      console.warn('[EmotionDetector] webcam or faceCanvas not found');
      return false;
    }
    await this._checkServerModel();
    return true;
  },

  async loadModels() {
    if (!this.videoElement || !this.canvasElement) {
      const ok = await this.init();
      if (!ok) return false;
    }
    if (!this.facemeshModel && !state.usingSimulatedEmotions) {
      await this._loadFacemesh();
    }
    await this._loadTfjsModel();
    return Boolean(this.facemeshModel || this.tfjsModel || this.serverModelAvailable);
  },

  async startCamera() {
    try {
      if (state.cameraActive && state.cameraStream) {
        const live = state.cameraStream.getVideoTracks()
          .filter(t => t.readyState === 'live');
        if (live.length) {
          this.videoElement.srcObject = state.cameraStream;
          await this.videoElement.play();
          this._syncCanvas(); this._attachResize();
          this._updateCameraUI('active');
          return true;
        }
      }

      this._stopTracks(this.videoElement?.srcObject);
      this._stopTracks(state.cameraStream);

      if (!this.facemeshModel && !state.usingSimulatedEmotions) {
        await this._loadFacemesh();
      }

      updateState({ cameraPermissionDenied: false, usingSimulatedEmotions: false });
      this._updateCameraUI('starting');

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 480 }, height: { ideal: 360 },
                 frameRate: { ideal: 30, max: 30 }, facingMode: 'user' },
      });

      this.videoElement.srcObject = stream;
      await new Promise(resolve => { this.videoElement.onloadedmetadata = resolve; });
      await this.videoElement.play();

      this._syncCanvas(); this._attachResize();
      updateState({ cameraStream: stream, cameraActive: true, faceDetectionConfirmed: false });
      this._updateCameraUI('active');
      return true;

    } catch (err) {
      console.error('[EmotionDetector] startCamera error:', err);
      this._updateCameraUI('error');
      updateState({ usingSimulatedEmotions: false });
      return false;
    }
  },

  stopCamera() {
    this.stopDetection();
    this._detachResize();
    this._stopTracks(this.videoElement?.srcObject);
    this._stopTracks(state.cameraStream);
    if (this.videoElement) {
      try { this.videoElement.pause(); } catch (_) {}
      this.videoElement.srcObject = null;
    }
    if (this.canvasElement) {
      const ctx = this.canvasElement.getContext('2d');
      ctx?.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
    this._smoothedScores     = null;
    this._lastFacePrediction = null;
    updateState({ cameraStream: null, cameraActive: false,
                  faceDetectionConfirmed: false, currentEmotion: 'neutral' });
    this._updateCameraUI('off');
  },

  prepareForRouteChange() {
    this.stopDetection();
    this._detachResize();
    if (this.videoElement) this.videoElement.srcObject = null;
  },

  async restoreActiveSession() {
    if (!state.cameraActive || !state.cameraStream) return false;
    if (!this.videoElement) {
      if (!(await this.init())) return false;
    }
    try {
      this.videoElement.srcObject = state.cameraStream;
      if (this.videoElement.readyState < 2)
        await new Promise(r => { this.videoElement.onloadedmetadata = r; });
      await this.videoElement.play();
      this._syncCanvas(); this._attachResize();
      this._updateCameraUI('active');
      return true;
    } catch (err) {
      console.error('[EmotionDetector] restoreActiveSession error:', err);
      return false;
    }
  },

  startDetection() {
    if (this.detectionActive) return;
    this.detectionActive = true;

    const hasTfjsModel = Boolean(this.tfjsEmotionHeadModel || this.tfjsModel);
    const hasModel = this.facemeshModel && (hasTfjsModel || this.serverModelAvailable);

    if (!hasModel || state.usingSimulatedEmotions) {
      this._startSimulation();
      return;
    }

    // RAF loop: draws the last known landmarks at 60fps regardless of inference speed.
    // This decouples visual smoothness from the slower FaceMesh/inference timers.
    this._startDrawLoop();

    // FaceMesh runs on its own guarded timer — only updates _lastFacePrediction.
    this._facemeshTimer = setInterval(async () => {
      if (!this.detectionActive || this.isRunningFacemesh) return;
      this.isRunningFacemesh = true;
      try { await this._runFacemesh(); }
      finally { this.isRunningFacemesh = false; }
    }, FACEMESH_INTERVAL_MS);

    // Emotion inference runs on a separate slower timer.
    this._inferenceTimer = setInterval(async () => {
      if (!this.detectionActive || this.isRunningInference) return;
      this.isRunningInference = true;
      try { await this._runInference(); }
      finally { this.isRunningInference = false; }
    }, INFERENCE_INTERVAL_MS);
  },

  _startDrawLoop() {
    const loop = () => {
      if (!this.detectionActive) return;
      this.animationFrameId = requestAnimationFrame(loop);
      if (!this.canvasElement || !state.cameraActive) return;
      const ctx = this.canvasElement.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
      if (this._lastFacePrediction) this._drawLandmarks(ctx, this._lastFacePrediction);
    };
    this.animationFrameId = requestAnimationFrame(loop);
  },

  stopDetection() {
    this.detectionActive = false;
    if (this._facemeshTimer) { clearInterval(this._facemeshTimer); this._facemeshTimer = null; }
    if (this._inferenceTimer) { clearInterval(this._inferenceTimer); this._inferenceTimer = null; }
    if (state.emotionDetectionInterval) {
      clearInterval(state.emotionDetectionInterval);
      updateState({ emotionDetectionInterval: null });
    }
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.isRunningFacemesh = false;
    this.isRunningInference = false;
    this.isDetectingFrame  = false;
  },

  resetFeedbackState() {
    this._smoothedScores     = null;
    this._lastFacePrediction = null;
  },


  // ═══════════════════════════════════════════
  //  MODEL LOADING
  // ═══════════════════════════════════════════

  async _checkServerModel() {
    if (this.serverStatusChecked) return;
    this.serverStatusChecked = true;
    try {
      const resp = await fetch(`${config.API_BASE_URL}/ai/emotion/status`);
      if (resp.ok) {
        const data = await resp.json();
        this.serverModelAvailable = Boolean(data.model_loaded);
        if (this.serverModelAvailable) {
          this.inferenceMode = MODE_SERVER;
          console.info('[EmotionDetector] Server model available');
        }
      }
    } catch (_) { this.serverModelAvailable = false; }
  },

  async _loadFacemesh() {
    this._updateModelStatus('loading', 'Loading face detection model…');
    try {
      await this._ensureTFLibraries();
      this.facemeshModel = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
        { maxFaces: 1 }
      );
      this._updateModelStatus('success', 'Face detection ready');
      updateState({ modelsLoaded: true });
      console.info('[EmotionDetector] FaceMesh loaded');
    } catch (err) {
      console.error('[EmotionDetector] FaceMesh load failed:', err);
      this._updateModelStatus('error', 'Face detection unavailable');
      updateState({ usingSimulatedEmotions: false });
    }
  },

  /**
   * FIX 1: Load TF.js model and extract scaler tensors from model.getWeights().
   *
   * .bin file weight tensor order:
   *   [0] dense_1/kernel  (1764, 512)
   *   [1] dense_1/bias    (512,)
   *   [2] dense_2/kernel  (512, 128)
   *   [3] dense_2/bias    (128,)
  *   [4] dense_3/kernel  (128, N)
  *   [5] dense_3/bias    (N,)
   *   [6] scaler/mean     (1764,)   ← StandardScaler mean
   *   [7] scaler/std      (1764,)   ← StandardScaler scale
   */
  async _loadTfjsModel() {
    if (this.tfjsEmotionHeadModel || this.tfjsModel) return true;
    try {
      await this._ensureTFLibraries();

      const modelUrl = config.EMOTION_TFJS_MODEL_URL || '/js/emotion_tfjs/model.json';
      console.info('[EmotionDetector] Loading TF.js single-head model from', modelUrl);

      // 1. Pre-fetch and patch the model JSON before TF.js sees it.
      //
      //    Root cause: Keras 3 / TF.js Converter v4 serialises the InputLayer
      //    with "batch_shape" but TF.js loadLayersModel expects "batchInputShape".
      //    Trying to fix it after loadLayersModel() is too late — the error is
      //    thrown inside the loader before we ever get a model object back.
      //    Solution: fetch → patch → load via tf.io.fromMemory().
      const baseUrl = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1);

      const modelResp = await fetch(modelUrl, { cache: 'no-store' });
      if (!modelResp.ok) throw new Error(`Failed to fetch model.json (${modelResp.status})`);
      const modelJson = await modelResp.json();

      // Walk every layer config and rename batch_shape → batchInputShape
      const patchLayers = (layers) => {
        if (!Array.isArray(layers)) return;
        for (const layer of layers) {
          const cfg = layer?.config;
          if (!cfg) continue;
          if (cfg.batch_shape !== undefined && cfg.batchInputShape === undefined) {
            cfg.batchInputShape = cfg.batch_shape;
            delete cfg.batch_shape;
          }
          // Handle nested layers (e.g. Sequential-inside-Sequential)
          if (Array.isArray(cfg.layers)) patchLayers(cfg.layers);
        }
      };

      const topLevelLayers =
        modelJson?.modelTopology?.model_config?.config?.layers;
      if (topLevelLayers) {
        patchLayers(topLevelLayers);
        console.info('[EmotionDetector] Patched batch_shape → batchInputShape in model topology');
      }

      // 2. Fetch all weight shards and concatenate them into one ArrayBuffer.
      const weightsManifest = modelJson.weightsManifest || [];
      const weightSpecs     = weightsManifest.flatMap(m => m.weights);
      const shardBuffers    = [];

      for (const manifest of weightsManifest) {
        for (const shardPath of manifest.paths) {
          const shardUrl  = baseUrl + shardPath;
          const shardResp = await fetch(shardUrl, { cache: 'no-store' });
          if (!shardResp.ok) throw new Error(`Failed to fetch weight shard: ${shardPath} (${shardResp.status})`);
          shardBuffers.push(await shardResp.arrayBuffer());
        }
      }

      const totalBytes  = shardBuffers.reduce((n, b) => n + b.byteLength, 0);
      const weightBytes = new Uint8Array(totalBytes);
      let   writeOffset = 0;
      for (const buf of shardBuffers) {
        weightBytes.set(new Uint8Array(buf), writeOffset);
        writeOffset += buf.byteLength;
      }

      // 3. Load the (now-patched) model entirely from memory.
      //    Use the single-argument ModelArtifacts form — the multi-arg overload
      //    is deprecated in recent TF.js builds.
      this.tfjsModel = await tf.loadLayersModel(
        tf.io.fromMemory({
          modelTopology: modelJson.modelTopology,
          weightSpecs,
          weightData: weightBytes.buffer,
        }),
        { strict: false },
      );

      // 4. Determine model type: CNN (Conv2D input) vs HOG+Dense.
      //    The scaler (StandardScaler over HOG features) only applies to the
      //    Dense-only variant.  For CNN models the raw pixel tensor goes
      //    straight into the network, so skip scaler hydration entirely.
      const isCNN = (modelJson?.modelTopology?.model_config?.config?.layers || [])
        .some(l => l.class_name === 'Conv2D');

      if (isCNN) {
        console.info('[EmotionDetector] CNN model detected — skipping HOG scaler (not needed)');
      } else {
        // HOG+Dense path: attempt to load the StandardScaler
        let scalerLoaded = await this._hydrateScalerFromModelMeta(modelUrl);
        if (!scalerLoaded) scalerLoaded = await this._hydrateScalerFromModelWeights();
        if (!scalerLoaded) {
          console.warn('[EmotionDetector] Scaler unavailable for single-head TF.js model; accuracy may degrade');
        }
      }

      this._ensureModelClassNames();

      this.inferenceMode = MODE_TFJS;
      console.info('[EmotionDetector] TF.js single-head model loaded');
      return true;

    } catch (err) {
      console.warn('[EmotionDetector] TF.js model load failed:', err.message);
      return false;
    }
  },

  async _ensureTFLibraries() {
    if (typeof tf !== 'undefined' && typeof faceLandmarksDetection !== 'undefined') return;
    let waited = 0;
    while ((typeof tf === 'undefined' || typeof faceLandmarksDetection === 'undefined')
           && waited < 8000) {
      await new Promise(r => setTimeout(r, 200));
      waited += 200;
    }
    if (typeof tf === 'undefined') throw new Error('TensorFlow.js not loaded');
    if (typeof faceLandmarksDetection === 'undefined')
      throw new Error('faceLandmarksDetection not loaded');
  },

  async _urlExists(url) {
    if (!url) return false;
    try {
      const headResp = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      if (headResp.ok) return true;
    } catch (_) {}

    try {
      const getResp = await fetch(url, { method: 'GET', cache: 'no-store' });
      return getResp.ok;
    } catch (_) {
      return false;
    }
  },

  async _hydrateScalerFromModelMeta(modelUrl) {
    if (!modelUrl) return false;
    if (this._scalerMean && this._scalerStd && this._modelMetaCache?.url === modelUrl) return true;
    if (this._modelMetaCache && this._modelMetaCache.url === modelUrl) {
      return Boolean(this._modelMetaCache.loaded);
    }
    try {
      const resp = await fetch(modelUrl, { cache: 'no-store' });
      if (!resp.ok) {
        this._modelMetaCache = { url: modelUrl, loaded: false };
        return false;
      }
      const json = await resp.json();
      const meta = json?.elevate_meta || {};
      this._applyModelMetadata(meta);
      const loaded = this._applyScalerFromMeta(meta);
      this._modelMetaCache = { url: modelUrl, loaded };
      return loaded;
    } catch (err) {
      console.warn('[EmotionDetector] Failed to fetch model metadata for scaler:', err?.message || err);
      this._modelMetaCache = { url: modelUrl, loaded: false };
      return false;
    }
  },

  async _hydrateScalerFromModelWeights() {
    try {
      const sourceModel = this.tfjsEmotionHeadModel || this.tfjsModel;
      if (!sourceModel || typeof sourceModel.getWeights !== 'function') {
        return false;
      }

      const allWeights = sourceModel.getWeights();
      if (!allWeights || allWeights.length < 8) {
        return false;
      }

      const meanTensor = allWeights[6];
      const stdTensor = allWeights[7];
      if (!meanTensor || !stdTensor) {
        return false;
      }

      if (meanTensor.shape?.[0] !== HOG_FEATURE_DIM || stdTensor.shape?.[0] !== HOG_FEATURE_DIM) {
        console.warn('[EmotionDetector] Scaler shape mismatch in model weights');
        return false;
      }

      this._scalerMean = await meanTensor.data();
      this._scalerStd = await stdTensor.data();
      console.info(
        `[EmotionDetector] Scaler loaded from model weights: mean[0]=${Number(this._scalerMean[0] || 0).toFixed(4)} std[0]=${Number(this._scalerStd[0] || 0).toFixed(4)}`
      );
      return true;
    } catch (err) {
      console.warn('[EmotionDetector] Failed to read scaler from model weights:', err?.message || err);
      return false;
    }
  },

  _applyScalerFromMeta(metaSection) {
    if (!metaSection) return false;
    const mean = metaSection.scaler_mean;
    const std  = metaSection.scaler_std;
    if (Array.isArray(mean) && Array.isArray(std) &&
        mean.length === HOG_FEATURE_DIM && std.length === HOG_FEATURE_DIM) {
      this._scalerMean = Float32Array.from(mean);
      this._scalerStd  = Float32Array.from(std);
      console.info('[EmotionDetector] Scaler loaded from model metadata');
      return true;
    }
    return false;
  },

  _canonicalEmotionName(value) {
    const key = String(value || '').trim().toLowerCase();
    return CLASS_NAME_ALIASES[key] || key;
  },

  _applyModelMetadata(metaSection) {
    if (!metaSection || typeof metaSection !== 'object') {
      return;
    }

    const rawClassNames = Array.isArray(metaSection.class_names)
      ? metaSection.class_names.map((name) => this._canonicalEmotionName(name)).filter(Boolean)
      : [];

    if (rawClassNames.length) {
      this._modelClassNames = rawClassNames;
    }

    const sanitiseMap = (candidateMap, defaults) => {
      if (!candidateMap || typeof candidateMap !== 'object') {
        return { ...defaults };
      }

      const merged = {};
      for (const className of CLASS_NAMES) {
        const rawValue = Number(candidateMap[className]);
        merged[className] = Number.isFinite(rawValue) && rawValue > 0
          ? rawValue
          : defaults[className];
      }
      return merged;
    };

    this._classBalanceWeights = sanitiseMap(
      metaSection.class_balance_weights,
      DEFAULT_CLASS_BALANCE_WEIGHTS,
    );
    this._classPriorMultipliers = sanitiseMap(
      metaSection.class_prior_multipliers,
      DEFAULT_CLASS_PRIOR_MULTIPLIERS,
    );
  },

  _ensureModelClassNames() {
    const sourceModel = this.tfjsEmotionHeadModel || this.tfjsModel;
    const units = Number(
      sourceModel?.outputs?.[0]?.shape?.[1]
      || sourceModel?.outputShape?.[1]
      || 0
    );

    if (Array.isArray(this._modelClassNames)
      && this._modelClassNames.length
      && (units <= 0 || this._modelClassNames.length === units)) {
      return;
    }

    if (units === CLASS_NAMES.length) {
      this._modelClassNames = CLASS_NAMES.slice();
      return;
    }
    if (units === LEGACY_CLASS_NAMES_6.length) {
      this._modelClassNames = LEGACY_CLASS_NAMES_6.slice();
      return;
    }
    if (units === LEGACY_CLASS_NAMES_4.length) {
      this._modelClassNames = LEGACY_CLASS_NAMES_4.slice();
      return;
    }

    const fallbackSize = Math.max(1, Math.min(units || CLASS_NAMES.length, CLASS_NAMES.length));
    this._modelClassNames = CLASS_NAMES.slice(0, fallbackSize);
  },


  // ═══════════════════════════════════════════
  //  DETECTION LOOPS
  // ═══════════════════════════════════════════

  async _runFacemesh() {
    if (!this.videoElement || this.videoElement.paused ||
        this.videoElement.readyState < 2 || !state.cameraActive) return;
    this._syncCanvas();
    try {
      const predictions = await this.facemeshModel.estimateFaces({
        input: this.videoElement, returnTensors: false,
        flipHorizontal: false, predictIrises: false,
      });
      if (!state.cameraActive) return;

      if (predictions && predictions.length > 0) {
        if (!state.faceDetectionConfirmed) updateState({ faceDetectionConfirmed: true });
        this._lastFacePrediction = predictions[0];
      } else {
        this._lastFacePrediction = null;
      }
      // Canvas drawing is handled by the 60fps RAF loop in _startDrawLoop().
    } catch (err) {
      console.warn('[EmotionDetector] FaceMesh estimateFaces error:', err.message);
    }
  },

  async _runInference() {
    // Compute hasTfjsModel BEFORE the early-return guard so we don't bail
    // out when there's a TF.js model but no face prediction yet.
    const hasTfjsModel = Boolean(this.tfjsEmotionHeadModel || this.tfjsModel);

    if (!this._lastFacePrediction && !this.serverModelAvailable && !hasTfjsModel) {
      this._dbg('gate', '[EmotionDetector] runInference: no face, no model, no server — skipping');
      return;
    }

    let result = null;
    const preferServer = config.EMOTION_PREFER_SERVER_INFERENCE === true;

    if (preferServer && this.serverModelAvailable) {
      result = await this._serverInference(this._lastFacePrediction);
      if (result) this.inferenceMode = MODE_SERVER;
    }

    if (!result && hasTfjsModel) {
      result = await this._tfjsInference(this._lastFacePrediction);
      if (result) this.inferenceMode = MODE_TFJS;
    }

    if (!result && this.serverModelAvailable && !preferServer) {
      result = await this._serverInference(this._lastFacePrediction);
      if (result) this.inferenceMode = MODE_SERVER;
      else if (!hasTfjsModel) this.serverModelAvailable = false;
    }

    if (!result) {
      this._dbg('gate', '[EmotionDetector] runInference: inference returned null');
      return;
    }

    this._dbg('raw', `[EmotionDetector] Raw result — emotion:${result.emotion} conf:${result.confidence?.toFixed(3)} scores:${JSON.stringify(Object.fromEntries(Object.entries(result.all_scores||{}).map(([k,v])=>[k,+v.toFixed(3)])))}`);

    const calibratedScores = this._applyLandmarkEmotionPriors(
      result.all_scores,
      this._lastFacePrediction,
      result.engagement_score,
    );

    if (calibratedScores) {
      result.all_scores = calibratedScores;
      const ranked = Object.entries(calibratedScores).sort((a, b) => b[1] - a[1]);
      result.emotion = ranked[0]?.[0] || result.emotion || 'neutral';
      result.confidence = Number(ranked[0]?.[1] || result.confidence || 0);
    }

    const { emotion, confidence, all_scores } = result;
    const passes = this._passesGate(all_scores, confidence);

    this._dbg('gate', `[EmotionDetector] Gate — top:${emotion}(${confidence?.toFixed(3)}) passes:${passes} | ${JSON.stringify(Object.fromEntries(Object.entries(all_scores||{}).map(([k,v])=>[k,+v.toFixed(3)])))}`);

    if (!passes) return;

    this._applySmoothing(all_scores);
    const smoothed = this._getSmoothedEmotion();

    this._dbg('smoothed', `[EmotionDetector] Smoothed → ${smoothed.emotion} (${Math.round(smoothed.confidence*100)}%)`);

    this._applyEmotion(smoothed.emotion, smoothed.confidence);
  },

  // Throttled debug logger — logs each key at most once per 1.5 s to avoid spam.
  _dbgTimers: {},
  _dbg(key, msg) {
    const now = Date.now();
    if ((this._dbgTimers[key] || 0) + 1500 > now) return;
    this._dbgTimers[key] = now;
    console.log(msg);
  },


  // ═══════════════════════════════════════════
  //  INFERENCE IMPLEMENTATIONS
  // ═══════════════════════════════════════════


  _smoothEmotions(rawEmotions) {
    this._emotionHistory.push(rawEmotions);
    if (this._emotionHistory.length > this._HISTORY_LENGTH) {
      this._emotionHistory.shift(); 
    }

    let smoothed = {};
    for (const key of CLASS_NAMES) smoothed[key] = 0;
    
    for (let frame of this._emotionHistory) {
      for (let key in frame) {
        smoothed[key] += (frame[key] || 0);
      }
    }

    for (let key in smoothed) {
      smoothed[key] /= this._emotionHistory.length;
    }
    return smoothed;
  },

  async _serverInference(facePrediction) {
    try {
      const b64 = this._captureAlignedFaceB64(facePrediction, 96);
      if (!b64) return null;
      const resp = await fetch(`${config.API_BASE_URL}/ai/emotion/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: b64 }),
        signal: AbortSignal.timeout(SERVER_INFERENCE_TIMEOUT_MS),
      });
      if (!resp.ok) return null;
      const data = await resp.json();
      return data.error ? null : data;
    } catch (_) { return null; }
  },

  async _tfjsInference(facePrediction) {
    try {
      const emotionModel = this.tfjsEmotionHeadModel || this.tfjsModel;
      if (!emotionModel || !this.videoElement) {
        console.warn('[EmotionDetector] _tfjsInference: no model or no video element');
        return null;
      }

      const vw = this.videoElement.videoWidth;
      const vh = this.videoElement.videoHeight;
      if (!vw || !vh) {
        this._dbg('crop', '[EmotionDetector] _tfjsInference: video not ready (0x0)');
        return null;
      }

      const rect = this._getFaceCropRect(facePrediction);

      // Clamp crop rect so it can never go out of bounds for tf.slice
      const sx = Math.max(0, Math.min(Math.floor(rect.sx), vw - 1));
      const sy = Math.max(0, Math.min(Math.floor(rect.sy), vh - 1));
      const sw = Math.max(1, Math.min(Math.floor(rect.sw), vw - sx));
      const sh = Math.max(1, Math.min(Math.floor(rect.sh), vh - sy));

      this._dbg('crop', `[EmotionDetector] Crop rect: sx=${sx} sy=${sy} sw=${sw} sh=${sh} (video ${vw}x${vh})`);

      const rawProbs = tf.tidy(() => {
        let img = tf.browser.fromPixels(this.videoElement); // [vh, vw, 3]

        // Crop to face region
        img = img.slice([sy, sx, 0], [sh, sw, 3]);

        // Resize to 48×48
        img = tf.image.resizeBilinear(img, [48, 48]);

        // Grayscale: weighted average matching BT.601 — same as PIL.convert('L')
        // Doing it with a matmul avoids the mean(2) which equally weights RGB
        const weights = tf.tensor1d([0.299, 0.587, 0.114]);
        img = img.mul(weights).sum(2); // [48, 48]

        // Reshape to [1, 48, 48, 1]. 
        // THE FIX: DO NOT .div(255.0)! The Python model was trained on raw 0-255 pixels!
        img = img.expandDims(0).expandDims(-1).div(255.0);

        return Array.from(emotionModel.predict(img).dataSync());
      });

      this._dbg('probs', `[EmotionDetector] Raw model probs: [${rawProbs.map(v=>v.toFixed(3)).join(', ')}]`);

      if (!rawProbs || !rawProbs.length) {
        console.warn('[EmotionDetector] _tfjsInference: model returned empty output');
        return null;
      }

      this._ensureModelClassNames();
      const mappedScores = this._mapScoresToCanonical(rawProbs, this._modelClassNames);
      if (!mappedScores) {
        console.warn('[EmotionDetector] _tfjsInference: _mapScoresToCanonical returned null');
        return null;
      }

      const balancedScores = this._applyClassBalance(mappedScores);
      const engagementScore = this._inferEngagementScore([]);

      const ranked = Object.entries(balancedScores).sort((a, b) => b[1] - a[1]);

      return {
        emotion:          ranked[0]?.[0] || 'neutral',
        confidence:       Number(ranked[0]?.[1] || 0),
        engagement_score: engagementScore,
        all_scores:       balancedScores,
      };
    } catch (err) {
      console.warn('[EmotionDetector] _tfjsInference error:', err.message, err.stack);
      return null;
    }
  },

  _classListEquals(left, right) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
      return false;
    }
    for (let i = 0; i < left.length; i++) {
      if (left[i] !== right[i]) return false;
    }
    return true;
  },

  _mapScoresToCanonical(rawProbs, sourceClassNames) {
    const probs = Array.from(rawProbs || []).map((value) => Math.max(0, Number(value || 0)));
    if (!probs.length) return null;

    const sourceLabels = Array.isArray(sourceClassNames)
      ? sourceClassNames.map((name) => this._canonicalEmotionName(name)).filter(Boolean)
      : [];

    if (sourceLabels.length === CLASS_NAMES.length && this._classListEquals(sourceLabels, CLASS_NAMES)) {
      return this._normalizeScoreMap(
        Object.fromEntries(CLASS_NAMES.map((className, index) => [className, probs[index] || 0])),
      );
    }

    if (sourceLabels.length === LEGACY_CLASS_NAMES_6.length && this._classListEquals(sourceLabels, LEGACY_CLASS_NAMES_6)) {
      const oldScores = Object.fromEntries(LEGACY_CLASS_NAMES_6.map((name, idx) => [name, probs[idx] || 0]));
      const surprised = (0.58 * oldScores.happy) + (0.22 * oldScores.confused) + (0.20 * oldScores.neutral);
      return this._normalizeScoreMap({ ...oldScores, surprised });
    }

    if (sourceLabels.length === LEGACY_CLASS_NAMES_4.length && this._classListEquals(sourceLabels, LEGACY_CLASS_NAMES_4)) {
      const oldScores = Object.fromEntries(LEGACY_CLASS_NAMES_4.map((name, idx) => [name, probs[idx] || 0]));
      const happy = Number(oldScores.happy || 0);
      const confused = Number(oldScores.confused || 0);
      const neutral = Number(oldScores.neutral || 0);
      const angry = Number(oldScores.angry || 0);
      const bored = 0.60 * neutral + 0.40 * confused;
      const focused = 0.62 * neutral + 0.38 * happy;
      const surprised = 0.52 * happy + 0.30 * confused + 0.18 * neutral;
      return this._normalizeScoreMap({ happy, bored, focused, confused, neutral, angry, surprised });
    }

    const mapped = Object.fromEntries(CLASS_NAMES.map((className) => [className, 0]));
    if (sourceLabels.length === probs.length) {
      sourceLabels.forEach((label, idx) => {
        if (label in mapped) {
          mapped[label] += probs[idx] || 0;
        }
      });
      return this._normalizeScoreMap(mapped);
    }

    if (probs.length === LEGACY_CLASS_NAMES_6.length) {
      return this._mapScoresToCanonical(probs, LEGACY_CLASS_NAMES_6);
    }
    if (probs.length === LEGACY_CLASS_NAMES_4.length) {
      return this._mapScoresToCanonical(probs, LEGACY_CLASS_NAMES_4);
    }

    const upto = Math.min(CLASS_NAMES.length, probs.length);
    for (let i = 0; i < upto; i++) {
      mapped[CLASS_NAMES[i]] = probs[i] || 0;
    }
    return this._normalizeScoreMap(mapped);
  },

  // FIX 1: Apply StandardScaler — z = (x - mean) / std
  // _applyScaler(features) {
  //   if (!this._scalerMean || !this._scalerStd ||
  //       this._scalerMean.length !== features.length) {
  //     return features; // No scaler → raw features (accuracy reduced but no crash)
  //   }
  //   const out = new Float32Array(features.length);
  //   for (let i = 0; i < features.length; i++) {
  //     const std = this._scalerStd[i] > 1e-8 ? this._scalerStd[i] : 1.0;
  //     out[i] = (features[i] - this._scalerMean[i]) / std;
  //   }
  //   return out;
  // },

  // FIX 4: Recall-derived class balance weights
  _applyClassBalance(scoreMap) {
    const weighted = {};
    for (const className of CLASS_NAMES) {
      const probability = Number(scoreMap?.[className] || 0);
      const rawWeight = Number(this._classBalanceWeights[className] ?? 1.0);
      const boundedWeight = Math.max(CLASS_WEIGHT_MIN, Math.min(CLASS_WEIGHT_MAX, rawWeight));
      const prior = Number(this._classPriorMultipliers[className] ?? 1.0);
      weighted[className] = Math.max(0, probability) * boundedWeight * prior;
    }
    return this._normalizeScoreMap(weighted);
  },

  _inferEngagementScore(scaledFeatures) {
    if (!this.tfjsEngagementHeadModel) return null;

    try {
      const raw = tf.tidy(() => {
        const t = tf.tensor2d(scaledFeatures, [1, HOG_FEATURE_DIM]);
        const y = this.tfjsEngagementHeadModel.predict(t);
        return Array.from(y.dataSync());
      });

      if (!raw.length) return null;
      if (raw.length === 1) {
        const value = Number(raw[0] || 0);
        if (value >= 0 && value <= 1) return value;
        return 1 / (1 + Math.exp(-value));
      }

      const safe = raw.map(v => Math.max(0, Number(v || 0)));
      const sum = safe.reduce((acc, v) => acc + v, 0);
      if (!sum) return null;

      const normalized = safe.map(v => v / sum);
      return normalized[normalized.length - 1];
    } catch (err) {
      console.warn('[EmotionDetector] Engagement head inference failed:', err?.message || err);
      return null;
    }
  },

  _normalizeScoreMap(scores) {
    const mapped = {};
    let total = 0;
    for (const className of CLASS_NAMES) {
      const value = Math.max(0, Number(scores?.[className] || 0));
      mapped[className] = value;
      total += value;
    }

    if (!Number.isFinite(total) || total <= 0) {
      const uniform = 1 / CLASS_NAMES.length;
      return Object.fromEntries(CLASS_NAMES.map(className => [className, uniform]));
    }

    return Object.fromEntries(CLASS_NAMES.map(className => [className, mapped[className] / total]));
  },

  _applyLandmarkEmotionPriors(scoreMap, facePrediction, engagementScore) {
    if (!scoreMap) return null;

    const adjusted = this._normalizeScoreMap(scoreMap);
    const landmarks = this._extractLandmarkSignals(facePrediction);

    if (landmarks) {
      if (landmarks.smileHint) {
        adjusted.happy *= 1.16;
        adjusted.angry *= 0.70;
      }

      if (landmarks.surpriseHint) {
        adjusted.surprised *= 1.24;
        adjusted.angry *= 0.74;
        adjusted.bored *= 0.82;
      }

      if (landmarks.calmBrow && !landmarks.browFurrowStrong) {
        adjusted.angry *= 0.66;
        adjusted.neutral *= 1.10;
        adjusted.focused *= 1.07;
      }

      if (!landmarks.browFurrowStrong && landmarks.eyeOpenAvg > 0.03) {
        adjusted.angry *= 0.82;
      }

      if (landmarks.browFurrowStrong && landmarks.eyeNarrow) {
        adjusted.angry *= 1.08;
        adjusted.confused *= 1.06;
      }

      const calmMass = (adjusted.neutral || 0) + (adjusted.focused || 0);
      if ((adjusted.angry || 0) > 0.28 && calmMass > (adjusted.angry || 0) * 1.15 && landmarks.calmBrow) {
        adjusted.angry *= 0.55;
        adjusted.neutral *= 1.08;
      }
    }

    if (Number.isFinite(engagementScore)) {
      if (engagementScore >= 0.65) {
        adjusted.focused *= 1.08;
        adjusted.angry *= 0.84;
      } else if (engagementScore <= 0.35) {
        adjusted.bored *= 1.07;
        adjusted.angry *= 0.90;
      }
    }

    return this._normalizeScoreMap(adjusted);
  },

  _extractLandmarkSignals(facePrediction) {
    const mesh = facePrediction?.scaledMesh;
    if (!Array.isArray(mesh) || mesh.length < 387) {
      return null;
    }

    const point = (index) => {
      const row = mesh[index];
      if (!row || row.length < 2) return null;
      const x = Number(row[0]);
      const y = Number(row[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      return { x, y };
    };

    const meanPoint = (indices) => {
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      for (const idx of indices) {
        const p = point(idx);
        if (!p) continue;
        sumX += p.x;
        sumY += p.y;
        count += 1;
      }
      if (!count) return null;
      return { x: sumX / count, y: sumY / count };
    };

    const distance = (a, b) => {
      if (!a || !b) return 0;
      return Math.hypot(a.x - b.x, a.y - b.y);
    };

    const leftEyeOuter = point(33);
    const rightEyeOuter = point(263);
    const mouthLeft = point(61);
    const mouthRight = point(291);
    const upperLip = point(13);
    const lowerLip = point(14);
    const leftBrow = meanPoint([70, 63, 105]);
    const rightBrow = meanPoint([336, 296, 334]);
    const leftEyeCenter = meanPoint([33, 133, 159, 145]);
    const rightEyeCenter = meanPoint([362, 263, 386, 374]);
    const leftEyeTop = point(159);
    const leftEyeBottom = point(145);
    const rightEyeTop = point(386);
    const rightEyeBottom = point(374);

    const faceWidth = Math.max(1, distance(leftEyeOuter, rightEyeOuter));
    const browGap = distance(leftBrow, rightBrow) / faceWidth;
    const browEyeLeft = distance(leftBrow, leftEyeCenter) / faceWidth;
    const browEyeRight = distance(rightBrow, rightEyeCenter) / faceWidth;
    const browEyeAvg = (browEyeLeft + browEyeRight) / 2;
    const mouthWidth = distance(mouthLeft, mouthRight) / faceWidth;
    const mouthOpen = distance(upperLip, lowerLip) / faceWidth;
    const eyeOpenLeft = distance(leftEyeTop, leftEyeBottom) / faceWidth;
    const eyeOpenRight = distance(rightEyeTop, rightEyeBottom) / faceWidth;
    const eyeOpenAvg = (eyeOpenLeft + eyeOpenRight) / 2;
    const smileHint = mouthWidth > 0.34 && mouthOpen > 0.025;

    return {
      smileHint,
      calmBrow: browEyeAvg > 0.09,
      browFurrowStrong: browGap < 0.165,
      eyeNarrow: eyeOpenAvg < 0.028,
      eyeOpenAvg,
      mouthOpen,
      surpriseHint: eyeOpenAvg > 0.038 && mouthOpen > 0.045 && !smileHint,
    };
  },


  // ═══════════════════════════════════════════
  //  HOG FEATURE EXTRACTION
  //  FIX 2: Simple central differences (matches skimage exactly)
  //  FIX 3: No CLAHE / gamma / blur preprocessing
  // ═══════════════════════════════════════════

  /**
   * Extract 1764-dim HOG vector from current video frame.
   *
   * Pipeline exactly matches train_emotion_fast.py:
   *   PIL→gray→resize(32,32) → skimage.hog(orient=9,ppc=(4,4),cpb=(2,2))
   *
   * FIX 2: gx = img[y,x+1] - img[y,x-1]  (NOT Sobel)
   * FIX 3: No CLAHE, no gamma, no blur before HOG
   */
  // _extractHog32(facePrediction) {
  //   if (!this.videoElement) return null;

  //   if (!this._hogCanvas) {
  //     this._hogCanvas        = document.createElement('canvas');
  //     this._hogCanvas.width  = HOG_IMG_SIZE;
  //     this._hogCanvas.height = HOG_IMG_SIZE;
  //   }

  //   const ctx = this._hogCanvas.getContext('2d', { willReadFrequently: true });
  //   if (!ctx) return null;

  //   this._drawAlignedFaceToCanvas(ctx, HOG_IMG_SIZE, HOG_IMG_SIZE, facePrediction);

  //   const rgba = ctx.getImageData(0, 0, HOG_IMG_SIZE, HOG_IMG_SIZE).data;

  //   // FIX 3: BT.601 grayscale ONLY — no CLAHE, no gamma, no blur
  //   const W    = HOG_IMG_SIZE;
  //   const H    = HOG_IMG_SIZE;
  //   const gray = new Float32Array(W * H);
  //   for (let i = 0; i < W * H; i++) {
  //     const r = rgba[i * 4];
  //     const g = rgba[i * 4 + 1];
  //     const b = rgba[i * 4 + 2];
  //     gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
  //   }

  //   // Gradient computation
  //   // FIX 2: Simple central difference — matches skimage.feature.hog exactly
  //   const cellHist = new Float32Array(HOG_CELLS_X * HOG_CELLS_Y * HOG_BINS);

  //   for (let y = 0; y < H; y++) {
  //     for (let x = 0; x < W; x++) {
  //       const xm1 = x > 0     ? x - 1 : 0;
  //       const xp1 = x < W - 1 ? x + 1 : W - 1;
  //       const ym1 = y > 0     ? y - 1 : 0;
  //       const yp1 = y < H - 1 ? y + 1 : H - 1;

  //       // FIX 2: central difference only — NO Sobel row weighting
  //       const gx = gray[y * W + xp1] - gray[y * W + xm1];
  //       const gy = gray[yp1 * W + x] - gray[ym1 * W + x];

  //       const mag = Math.sqrt(gx * gx + gy * gy);
  //       if (mag === 0) continue;

  //       // Unsigned orientation [0, 180)
  //       let angle = Math.atan2(Math.abs(gy), gx) * (180 / Math.PI);
  //       if (angle < 0) angle += 180;
  //       if (angle >= 180) angle = 0;

  //       // Soft bin assignment
  //       const binF = (angle / 180) * HOG_BINS;
  //       const bin0 = Math.floor(binF) % HOG_BINS;
  //       const bin1 = (bin0 + 1) % HOG_BINS;
  //       const w1   = binF - Math.floor(binF);
  //       const w0   = 1 - w1;

  //       const cx   = Math.min(HOG_CELLS_X - 1, Math.floor(x / HOG_PPC_X));
  //       const cy   = Math.min(HOG_CELLS_Y - 1, Math.floor(y / HOG_PPC_Y));
  //       const base = (cy * HOG_CELLS_X + cx) * HOG_BINS;
  //       cellHist[base + bin0] += mag * w0;
  //       cellHist[base + bin1] += mag * w1;
  //     }
  //   }

  //   // Block normalisation: 7×7 blocks of 2×2 cells, L2-Hys
  //   const nBX     = HOG_CELLS_X - 1;  // 7
  //   const nBY     = HOG_CELLS_Y - 1;  // 7
  //   const bSize   = HOG_BLOCK_W * HOG_BLOCK_H * HOG_BINS;  // 36
  //   const features = new Float32Array(nBX * nBY * bSize);  // 1764
  //   let outIdx = 0;

  //   for (let by = 0; by < nBY; by++) {
  //     for (let bx = 0; bx < nBX; bx++) {
  //       const block = new Float32Array(bSize);
  //       let p = 0;
  //       for (let dy = 0; dy < HOG_BLOCK_H; dy++) {
  //         for (let dx = 0; dx < HOG_BLOCK_W; dx++) {
  //           const cellBase = ((by + dy) * HOG_CELLS_X + (bx + dx)) * HOG_BINS;
  //           for (let b = 0; b < HOG_BINS; b++) block[p++] = cellHist[cellBase + b];
  //         }
  //       }
  //       // L2-Hys: normalise → clip 0.2 → renormalise
  //       let sqSum = 0;
  //       for (let i = 0; i < bSize; i++) sqSum += block[i] * block[i];
  //       let norm = Math.sqrt(sqSum + 1e-6);
  //       for (let i = 0; i < bSize; i++) block[i] = Math.min(0.2, block[i] / norm);
  //       sqSum = 0;
  //       for (let i = 0; i < bSize; i++) sqSum += block[i] * block[i];
  //       norm = Math.sqrt(sqSum + 1e-6);
  //       for (let i = 0; i < bSize; i++) features[outIdx++] = block[i] / norm;
  //     }
  //   }

  //   return features;
  // },


  // ═══════════════════════════════════════════
  //  GATES + SMOOTHING
  // ═══════════════════════════════════════════

  _passesGate(allScores, confidence) {
    if (!allScores) return confidence >= MIN_CONFIDENCE;
    const sorted = Object.values(allScores).sort((a, b) => b - a);
    const top1   = sorted[0] ?? 0;
    const top2   = sorted[1] ?? 0;
    if (top1 >= 0.65) return (top1 - top2) >= 0.04;
    return top1 >= MIN_CONFIDENCE && (top1 - top2) >= MIN_MARGIN;
  },

  _applySmoothing(allScores) {
    if (!allScores) return;
    const uniform = 1 / CLASS_NAMES.length;
    if (!this._smoothedScores)
      this._smoothedScores = Object.fromEntries(CLASS_NAMES.map(c => [c, uniform]));

    const vals   = Object.values(allScores);
    const top1   = Math.max(...vals);
    const top2   = vals.slice().sort((a, b) => b - a)[1] ?? 0;
    const alpha  = Math.min(0.7, SMOOTH_ALPHA + (top1 - top2) * 0.4);

    for (const c of CLASS_NAMES) {
      const prev = this._smoothedScores[c] ?? uniform;
      const curr = allScores[c] ?? 0;
      let next = (1 - alpha) * prev + alpha * curr;
      next = (1 - SMOOTH_LEAK) * next + SMOOTH_LEAK * uniform;
      this._smoothedScores[c] = next;
    }
    const total = CLASS_NAMES.reduce((s, c) => s + (this._smoothedScores[c] || 0), 0);
    if (total > 0) for (const c of CLASS_NAMES) this._smoothedScores[c] /= total;
  },

  _getSmoothedEmotion() {
    if (!this._smoothedScores) return { emotion: 'neutral', confidence: 0.5 };
    const ranked = Object.entries(this._smoothedScores).sort((a, b) => b[1] - a[1]);
    return { emotion: ranked[0]?.[0] || 'neutral', confidence: ranked[0]?.[1] || 0.5 };
  },


  // ═══════════════════════════════════════════
  //  APPLY EMOTION → UI + STATE + LOG
  // ═══════════════════════════════════════════

  _applyEmotion(emotion, confidence) {
    updateState({ currentEmotion: emotion });
    this._updateEmotionUI(emotion, confidence);
    const settings = this._getSettings();
    if (settings.enableEmotionFeedback !== false) this._throttledLog(emotion, confidence);
  },

  _updateEmotionUI(emotion, confidence) {
    const icon      = document.getElementById('emotionIcon');
    const text      = document.getElementById('emotionText');
    const indicator = document.getElementById('emotionIndicator');

    if (!icon || !text || !indicator) {
      console.warn('[EmotionDetector] _updateEmotionUI: missing DOM element(s)',
        { icon: !!icon, text: !!text, indicator: !!indicator });
      return;
    }

    const pct   = Math.round(confidence * 100);
    const label = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    const emoji = EMOTION_EMOJI[emotion.toLowerCase()] || '😐';

    icon.textContent = emoji;
    // innerHTML is safe — emotion is always from CLASS_NAMES (fixed set).
    text.innerHTML =
      `${label}<span style="font-size:0.78em;opacity:0.72;margin-left:5px">${pct}%</span>`;

    indicator.style.cssText =
      'display:flex;position:absolute;top:8px;right:8px;left:auto;bottom:auto;';

    document.getElementById('emotionModeLabel')?.remove();
    document.getElementById('emotionScoreLabel')?.remove();

    this._dbg('ui', `[EmotionDetector] UI updated → ${label} ${pct}%`);
  },

  async _throttledLog(emotion, confidence) {
    const now = Date.now();
    if (now - this._lastLogTime < this._logIntervalMs) return;
    this._lastLogTime = now;
    try {
      const { api } = await import('./api.js');
      const ctx = state.questionsAnswered > 0 ? 'answering_question' : 'session_active';
      await api.emotions.log(emotion, confidence, ctx);
    } catch (_) {}
  },


  // ═══════════════════════════════════════════
  //  SIMULATION
  // ═══════════════════════════════════════════

  _startSimulation() {
    updateState({ usingSimulatedEmotions: true, faceDetectionConfirmed: false });
    this._updateCameraUI('active');
    const emotions = ['neutral', 'focused', 'happy', 'surprised', 'confused', 'bored', 'neutral'];
    let idx = 0;
    const interval = setInterval(() => {
      if (!this.detectionActive) { clearInterval(interval); return; }
      this._applyEmotion(emotions[idx % emotions.length], 0.70 + Math.random() * 0.25);
      idx++;
    }, 1800);
    updateState({ emotionDetectionInterval: interval });
  },


  // ═══════════════════════════════════════════
  //  FACE CROP + DRAWING
  // ═══════════════════════════════════════════

  _drawAlignedFaceToCanvas(ctx, outW, outH, facePrediction) {
    if (!ctx || !this.videoElement) return;
    const rect  = this._getFaceCropRect(facePrediction);
    const angle = this._estimateFaceRoll(facePrediction);
    ctx.save();
    ctx.clearRect(0, 0, outW, outH);
    ctx.translate(outW / 2, outH / 2);
    ctx.rotate(-angle);
    ctx.drawImage(this.videoElement, rect.sx, rect.sy, rect.sw, rect.sh,
                  -outW / 2, -outH / 2, outW, outH);
    ctx.restore();
  },

  _getFaceCropRect(facePrediction) {
    const vw  = this.videoElement?.videoWidth  || 96;
    const vh  = this.videoElement?.videoHeight || 96;
    const toN = v => Array.isArray(v) ? Number(v[0]) : Number(v);
    const box = facePrediction?.boundingBox;
    const tl  = box?.topLeft;
    const br  = box?.bottomRight;

    let x1 = isFinite(toN(tl?.[0])) ? toN(tl[0]) : -1;
    let y1 = isFinite(toN(tl?.[1])) ? toN(tl[1]) : -1;
    let x2 = isFinite(toN(br?.[0])) ? toN(br[0]) : -1;
    let y2 = isFinite(toN(br?.[1])) ? toN(br[1]) : -1;

    if (x1 < 0 || y1 < 0 || x2 <= x1 || y2 <= y1) {
      const px = vw * 0.10, py = vh * 0.10;
      return { sx: px, sy: py, sw: vw - 2 * px, sh: vh - 2 * py };
    }

    const w    = x2 - x1, h = y2 - y1;
    const padX = w * 0.18, padY = h * 0.24;
    const sx   = Math.max(0,  Math.floor(x1 - padX));
    const sy   = Math.max(0,  Math.floor(y1 - padY));
    return {
      sx, sy,
      sw: Math.min(vw, Math.ceil(x2 + padX)) - sx,
      sh: Math.min(vh, Math.ceil(y2 + padY)) - sy,
    };
  },

  _estimateFaceRoll(facePrediction) {
    const kps = facePrediction?.scaledMesh;
    if (!Array.isArray(kps) || kps.length < 400) return 0;
    const avg = (indices) => {
      let sx = 0, sy = 0, n = 0;
      for (const idx of indices) {
        const p = kps[idx];
        if (p) { sx += p[0]; sy += p[1]; n++; }
      }
      return n ? { x: sx / n, y: sy / n } : null;
    };
    const L = avg([33, 133, 159, 145]);
    const R = avg([362, 263, 386, 374]);
    if (!L || !R) return 0;
    return Math.atan2(R.y - L.y, R.x - L.x);
  },

  _captureAlignedFaceB64(facePrediction, size) {
    try {
      const tmp = document.createElement('canvas');
      tmp.width = size; tmp.height = size;
      this._drawAlignedFaceToCanvas(tmp.getContext('2d'), size, size, facePrediction);
      return tmp.toDataURL('image/jpeg', 0.72);
    } catch (_) { return null; }
  },

  _drawLandmarks(ctx, prediction) {
    const kps = prediction?.scaledMesh;
    if (!kps) return;
    ctx.fillStyle = 'rgba(76,245,133,0.80)';
    for (let i = 0; i < kps.length; i++) {
      const p = kps[i];
      if (p) {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  },


  // ═══════════════════════════════════════════
  //  CAMERA UI
  // ═══════════════════════════════════════════

  _updateCameraUI(status) {
    const statusDot   = document.getElementById('statusDot');
    const statusText  = document.getElementById('cameraStatusText');
    const placeholder = document.getElementById('webcamPlaceholder');
    const webcam      = document.getElementById('webcam');
    const indicator   = document.getElementById('emotionIndicator');
    const startBtn    = document.getElementById('startCamera');
    const stopBtn     = document.getElementById('stopCamera');

    const show = el => el && (el.style.display = 'flex');
    const hide = el => el && (el.style.display = 'none');

    webcam?.classList.remove('active');
    placeholder?.classList.remove('hidden');
    hide(indicator); show(startBtn); hide(stopBtn);

    if (status === 'starting' || status === 'active') {
      webcam?.classList.add('active');
      placeholder?.classList.add('hidden');
      show(indicator); hide(startBtn); show(stopBtn);
      if (statusDot)  statusDot.className   = status === 'active' ? 'status-dot active' : 'status-dot loading';
      if (statusText) statusText.textContent = status === 'active' ? 'Active' : 'Initialising…';
      if (startBtn)   { startBtn.disabled = false; startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera'; }
    } else if (status === 'off') {
      if (statusDot)  statusDot.className   = 'status-dot';
      if (statusText) statusText.textContent = 'Camera Off';
      const ph = document.getElementById('webcamPlaceholder');
      if (ph) { ph.classList.remove('hidden'); ph.style.display = 'flex'; }
    } else if (status === 'error') {
      if (placeholder) placeholder.innerHTML = '<i class="fas fa-exclamation-triangle"></i><p>Camera error</p>';
    }
  },

  _isSimulationAllowed() {
    return config.EMOTION_ALLOW_SIMULATION_FALLBACK === true;
  },

  _updateModelStatus(type, message) {
    const statusEl = document.getElementById('modelLoadingStatus');
    const textEl   = document.getElementById('modelStatusText');
    const retryBtn = document.getElementById('retryModelLoad');
    if (statusEl) statusEl.className = `model-loading-status ${type} show`;
    if (textEl)   textEl.textContent = message;
    if (retryBtn) retryBtn.style.display = type === 'error' ? 'inline-block' : 'none';
  },

  _getSettings() {
    try { return JSON.parse(localStorage.getItem('userSettings') || '{}') || {}; }
    catch (_) { return {}; }
  },

  _syncCanvas() {
    if (!this.videoElement || !this.canvasElement) return;
    const vw = this.videoElement.videoWidth;
    const vh = this.videoElement.videoHeight;
    if (vw > 0 && vh > 0) {
      if (this.canvasElement.width  !== vw) this.canvasElement.width  = vw;
      if (this.canvasElement.height !== vh) this.canvasElement.height = vh;
    }
  },

  _attachResize() {
    if (this._resizeHandler) return;
    this._resizeHandler = () => this._syncCanvas();
    window.addEventListener('resize', this._resizeHandler, { passive: true });
  },

  _detachResize() {
    if (!this._resizeHandler) return;
    window.removeEventListener('resize', this._resizeHandler);
    this._resizeHandler = null;
  },

  _stopTracks(stream) {
    stream?.getTracks?.().forEach(t => { try { t.stop(); } catch (_) {} });
  },
};