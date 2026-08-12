/**
 * Elevate — final browser emotion detector
 * ----------------------------------------
 * Camera → Face detector → 96x96 RGB face crop
 *        → TF.js emotion model → temporal smoothing → UI
 *
 /*
 * IMPORTANT:
 * The exported Keras model contains its own preprocessing:
 * [0,1] → [-1,1]
 *
 * Therefore JavaScript must first convert camera pixels
 * from [0,255] to [0,1].
 */

import { config } from './config.js';
import { state, updateState } from './state.js';

const CLASS_NAMES = [
  'happy',
  'bored',
  'focused',
  'confused',
  'neutral',
  'angry',
  'surprised'
];

const EMOJI = {
  happy: '😊',
  bored: '😑',
  focused: '🧠',
  confused: '😕',
  neutral: '😐',
  angry: '😠',
  surprised: '😮'
};

const MODEL_URL =
  config.EMOTION_TFJS_MODEL_URL ||
  '/js/emotion_tfjs/model.json';

const MODEL_LOAD_TIMEOUT_MS = 30000;

// Facemesh no longer runs on a fixed interval — see _startFacemeshLoop().
// It now runs continuously, tied to actual camera frame delivery,
// so landmark updates are only bounded by hardware, not an arbitrary clock.
const INFERENCE_INTERVAL_MS = 300;

const SMOOTHING_ALPHA = 0.65;

const CROP_PAD_X = 0.22;
const CROP_PAD_Y = 0.28;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export const emotionDetector = {

  videoElement: null,
  canvasElement: null,

  facemeshModel: null,
  tfjsModel: null,

  detectionActive: false,

  isRunningFacemesh: false,
  isRunningInference: false,

  animationFrameId: null,

  // Increments whenever a new camera lifecycle starts/stops.
  // Prevents stale async operations from modifying a newer session.
  _cameraGeneration: 0,

  _facemeshLoopHandle: null,
  _facemeshUsesVideoFrameCallback: false,
  _inferenceTimer: null,

  _lastFacePrediction: null,
  _smoothedScores: null,

  _lastLogTime: 0,
  _logIntervalMs: 10000,

  _modelLoadPromise: null,

  _faceDetectionCanvas: null,
  _faceDetectionContext: null,

  // ============================================================
  // INITIALIZATION
  // ============================================================

  async init() {

    this.videoElement =
      document.getElementById('webcam');

    this.canvasElement =
      document.getElementById('faceCanvas');

    if (!this.videoElement) {
      console.error(
        '[EmotionDetector] #webcam not found'
      );

      return false;
    }

    if (!this.canvasElement) {
      console.warn(
        '[EmotionDetector] #faceCanvas not found; continuing without overlay'
      );
    }

    updateState({
      modelsLoaded: false,
      usingSimulatedEmotions: false,
      faceDetectionConfirmed: false
    });

    return true;
  },

  // ============================================================
  // MODEL LOADING
  // ============================================================

  async loadModels() {

    if (this._modelLoadPromise) {
      return this._modelLoadPromise;
    }

    this._modelLoadPromise =
      this._loadModelsImpl().catch(error => {

        this._modelLoadPromise = null;

        this._updateModelStatus(
          'error',
          `Emotion models failed: ${error.message}`
        );

        throw error;
      });

    return this._modelLoadPromise;
  },

  async _loadModelsImpl() {

    if (!this.videoElement) {

      const ok = await this.init();

      if (!ok) {
        return false;
      }
    }

    if (typeof tf === 'undefined') {

      throw new Error(
        'TensorFlow.js is not loaded'
      );
    }

    if (typeof faceLandmarksDetection === 'undefined') {

      throw new Error(
        'face-landmarks-detection is not loaded'
      );
    }

    await tf.ready();

    // Prefer WebGL.
    if (tf.getBackend() !== 'webgl') {

      try {

        await tf.setBackend('webgl');
        await tf.ready();

      } catch (_) {

        try {

          await tf.setBackend('cpu');
          await tf.ready();

        } catch (cpuError) {

          throw new Error(
            `No TensorFlow.js backend available: ${cpuError.message}`
          );
        }
      }
    }

    // ------------------------------------------------------------
    // Face detector
    // ------------------------------------------------------------

    this._updateModelStatus(
      'loading',
      'Loading face detector…'
    );

    this.facemeshModel =
      await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: 'tfjs',
          maxFaces: 1,
          refineLandmarks: false
        }
      );

    // ------------------------------------------------------------
    // Emotion classifier
    // ------------------------------------------------------------

    this._updateModelStatus(
      'loading',
      'Loading emotion CNN…'
    );

    const loadPromise =
      tf.loadLayersModel(MODEL_URL);

    const timeoutPromise =
      new Promise((_, reject) => {

        setTimeout(() => {

          reject(
            new Error(
              `model.json load exceeded ${MODEL_LOAD_TIMEOUT_MS / 1000}s`
            )
          );

        }, MODEL_LOAD_TIMEOUT_MS);
      });

    this.tfjsModel =
      await Promise.race([
        loadPromise,
        timeoutPromise
      ]);

    // ------------------------------------------------------------
    // Validate model
    // ------------------------------------------------------------

    const inputShape =
      this.tfjsModel.inputs?.[0]?.shape;

    const outputShape =
      this.tfjsModel.outputs?.[0]?.shape;

    if (
      !Array.isArray(inputShape) ||
      inputShape.length !== 4 ||
      inputShape[1] !== 96 ||
      inputShape[2] !== 96 ||
      inputShape[3] !== 3
    ) {

      throw new Error(
        `Unexpected emotion model input shape: ${JSON.stringify(inputShape)}`
      );
    }

    if (
      !Array.isArray(outputShape) ||
      outputShape.length !== 2 ||
      outputShape[1] !== CLASS_NAMES.length
    ) {

      throw new Error(
        `Unexpected emotion model output shape: ${JSON.stringify(outputShape)}`
      );
    }

    updateState({
      modelsLoaded: true,
      usingSimulatedEmotions: false
    });

    this._updateModelStatus(
      'loading',
      'Models loaded — looking for your face…'
    );

    console.info(
      '[EmotionDetector] Ready',
      `backend=${tf.getBackend()}`,
      `input=${JSON.stringify(inputShape)}`,
      `output=${JSON.stringify(outputShape)}`
    );

    return true;
  },

  // ============================================================
  // CAMERA
  // ============================================================

  async startCamera() {
    const generation = ++this._cameraGeneration;

    this._emitLifecycleState('starting');

    try {
      if (!this.videoElement) {

        const ok = await this.init();

        if (!ok || generation !== this._cameraGeneration) {
          return false;
        }
      }

      // Reuse existing stream if still alive.
      if (
        state.cameraActive &&
        state.cameraStream
      ) {
        const live = state.cameraStream
          .getVideoTracks()
          .some(track => track.readyState === 'live');

        if (live) {
          if (generation !== this._cameraGeneration) {
            return false;
          }

          this.videoElement.srcObject = state.cameraStream;

          await this.videoElement.play();

          if (generation !== this._cameraGeneration) {
            return false;
          }

          this._syncCanvas();
          this._updateCameraUI('searching');
          this._emitLifecycleState('searching');

          return true;
        }
      }

      this._stopTracks(this.videoElement.srcObject);

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: {
              ideal: 480,
              max: 640
            },

            height: {
              ideal: 360,
              max: 480
            },

            frameRate: {
              ideal: 30,
              max: 30
            },

            facingMode: 'user'
          },

          audio: false
        });

      // Stop was pressed while getUserMedia() was pending.
      if (generation !== this._cameraGeneration) {
        this._stopTracks(stream);
        return false;
      }

      this.videoElement.srcObject = stream;
      this.videoElement.muted = true;
      this.videoElement.playsInline = true;

      await new Promise((resolve, reject) => {

        let finished = false;

        const finish = () => {

          if (finished) {
            return;
          }

          finished = true;

          this.videoElement.removeEventListener(
            'loadedmetadata',
            finish
          );

          resolve();
        };

        this.videoElement.addEventListener(
          'loadedmetadata',
          finish,
          { once: true }
        );

        setTimeout(() => {
          if (!finished && this.videoElement.videoWidth > 0) {
            finish();

          } else if (!finished) {

            reject(
              new Error(
                'Camera video metadata did not become available'
              )
            );
          }

        }, 4000);
      });

      if (generation !== this._cameraGeneration) {
        this._stopTracks(stream);
        return false;
      }

      await this.videoElement.play();

      if (generation !== this._cameraGeneration) {
        this._stopTracks(stream);
        return false;
      }

      updateState({
        cameraStream: stream,

        cameraActive: true,

        cameraPermissionDenied: false,

        faceDetectionConfirmed: false,

        usingSimulatedEmotions: false

      });

      this._syncCanvas();

      this._updateCameraUI('starting');

      // Models may already be loaded. loadModels() is idempotent.
      await this.loadModels();
          
      // Stop may have happened while models were loading.
      if (generation !== this._cameraGeneration) {
        this._stopTracks(stream);
        return false;
      }
      
      this._updateCameraUI('searching');
      this._emitLifecycleState('searching');

      return true;

    } catch (error) {
      // A stale camera request must never overwrite the current UI/state.
      if (generation !== this._cameraGeneration) {
        return false;
      }

      console.error(
        '[EmotionDetector] startCamera failed:',
        error
      );

      updateState({

        cameraActive: false,
        cameraStream: null,
        cameraPermissionDenied: true,
        faceDetectionConfirmed: false,
        usingSimulatedEmotions: false

      });

      this._clearOverlay();
      this._updateCameraUI('error');
      this._emitLifecycleState('error');

      return false;
    }
  },

  async restoreActiveSession() {

    if (!this.videoElement) {
      await this.init();
    }

    if (!state.cameraStream) {
      return false;
    }

    const live =
      state.cameraStream
        .getVideoTracks()
        .some(track =>
          track.readyState === 'live'
        );

    if (!live) {
      return false;
    }

    this.videoElement.srcObject =
      state.cameraStream;

    await this.videoElement.play();

    this._syncCanvas();

    await this.loadModels();

    return true;
  },

  // ============================================================
  // DETECTION
  // ============================================================

  startDetection() {

    if (this.detectionActive) {
      return;
    }

    if (
      !this.facemeshModel ||
      !this.tfjsModel
    ) {

      console.warn(
        '[EmotionDetector] startDetection called before models are ready'
      );

      return;
    }

    if (!state.cameraActive) {

      console.warn(
        '[EmotionDetector] startDetection called while camera is inactive'
      );

      return;
    }

    this.detectionActive = true;

    this._updateModelStatus(
      'loading',
      'Looking for your face…'
    );

    this._startFacemeshLoop();

    this._inferenceTimer =
      setInterval(
        () => this._runInference(),
        INFERENCE_INTERVAL_MS
      );

    this._startDrawLoop();
  },

  _emitLifecycleState(status) {
  if (typeof window === 'undefined') {
    return;
  }

  window.dispatchEvent(
    new CustomEvent('elevate:emotion-state', {
      detail: { status }
    })
  );
},

  _clearOverlay() {
    if (!this.canvasElement) return;

    const ctx = this.canvasElement.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(
      0,
      0,
      this.canvasElement.width,
      this.canvasElement.height
    );
  },

  stopDetection() {

    this.detectionActive = false;

    this._stopFacemeshLoop();

    if (this._inferenceTimer) {

      clearInterval(
        this._inferenceTimer
      );

      this._inferenceTimer = null;
    }

    this.isRunningFacemesh = false;
    this.isRunningInference = false;

    if (this.animationFrameId) {

      cancelAnimationFrame(
        this.animationFrameId
      );

      this.animationFrameId = null;
    }

    if (state.emotionDetectionInterval) {

      clearInterval(
        state.emotionDetectionInterval
      );

      updateState({
        emotionDetectionInterval: null
      });
    }

    this._lastFacePrediction = null;
    this._smoothedScores = null;
    
    this._clearOverlay();
  },

  // ============================================================
  // STOP CAMERA
  // ============================================================

  stopCamera() {
    // Invalidate every pending start/getUserMedia/model-load continuation.
    ++this._cameraGeneration;

    this._emitLifecycleState('stopping');

    this.stopDetection();

    this._stopTracks(
      this.videoElement?.srcObject
    );

    this._stopTracks(
      state.cameraStream
    );

    if (this.videoElement) {

      this.videoElement.pause();

      this.videoElement.srcObject = null;
    }

    this._lastFacePrediction = null;
    this._smoothedScores = null;

    this._clearOverlay();

    updateState({
      cameraActive: false,

      cameraStream: null,

      faceDetectionConfirmed: false,

      currentEmotion: 'neutral',

      usingSimulatedEmotions: false,
      modelsLoaded: Boolean(
        this.tfjsModel &&
        this.facemeshModel
      )
    });

    this._updateCameraUI('off');
    this._emitLifecycleState('off');
},

  prepareForRouteChange() {

    this.stopDetection();

    this._lastFacePrediction = null;
    this._smoothedScores = null;
  },

  resetFeedbackState() {

    this._lastFacePrediction = null;
    this._smoothedScores = null;
  },

  // ============================================================
  // FACE DETECTION
  // ============================================================

    async _runFaceDetection() {

    if (
      !this.detectionActive ||
      this.isRunningFacemesh ||
      !this.facemeshModel ||
      !this.videoElement ||
      !state.cameraActive ||
      this.videoElement.readyState < 2
    ) {
      return;
    }

    const generation = this._cameraGeneration;
    this.isRunningFacemesh = true;

    try {

      const video = this.videoElement;

      const width = video.videoWidth;
      const height = video.videoHeight;

      if (!width || !height) {
        return;
      }

      // FaceMesh is more reliable when given an actual image/canvas
      // rather than the live HTMLVideoElement.
      //
      // Keep the canvas at the video's native resolution so the
      // returned bounding-box coordinates remain compatible with
      // _getFaceCropRect() and the emotion model crop.
      if (
        !this._faceDetectionCanvas ||
        this._faceDetectionCanvas.width !== width ||
        this._faceDetectionCanvas.height !== height
      ) {

        this._faceDetectionCanvas =
          document.createElement('canvas');

        this._faceDetectionCanvas.width = width;
        this._faceDetectionCanvas.height = height;

        this._faceDetectionContext =
          this._faceDetectionCanvas.getContext('2d', {
            willReadFrequently: false
          });
      }

      const ctx = this._faceDetectionContext;

      if (!ctx) {
        return;
      }

      ctx.drawImage(
        video,
        0,
        0,
        width,
        height
      );

      const faces =
        await this.facemeshModel.estimateFaces(
          this._faceDetectionCanvas,
          {
            flipHorizontal: false,
            staticImageMode: false
          }
        );

      if (
        !this.detectionActive ||
        !state.cameraActive ||
        generation !== this._cameraGeneration
      ) {
        return;
      }

      if (
        faces &&
        faces.length > 0
      ) {

        const face = faces[0];

        this._lastFacePrediction =
          this._normalizeFace(face);

        if (
          !state.faceDetectionConfirmed
        ) {

          updateState({
            faceDetectionConfirmed: true
          });

          this._updateModelStatus(
            'success',
            'Ready — face detected'
          );
          
          this._updateCameraUI('active');
          this._emitLifecycleState('active');
        }

      } else {

        this._lastFacePrediction = null;

        if (
          state.faceDetectionConfirmed
        ) {

          updateState({
            faceDetectionConfirmed: false
          });
        }
      }

    } catch (error) {

      console.warn(
        '[EmotionDetector] face detection error:',
        error
      );

    } finally {

      this.isRunningFacemesh = false;
    }
  },

  _normalizeFace(face) {

    const box = face?.box || {};

    const x = Number(box.xMin);
    const y = Number(box.yMin);
    const w = Number(box.width);
    const h = Number(box.height);

    return {

      ...face,

      boundingBox: {

        topLeft: [
          x,
          y
        ],

        bottomRight: [
          x + w,
          y + h
        ]
      }
    };
  },

  // ============================================================
  // FACE DETECTION LOOP
  // ============================================================
  //
  // Runs continuously rather than on a fixed setInterval. Each cycle
  // waits for _runFaceDetection() to finish, then immediately asks
  // for the next real camera frame via requestVideoFrameCallback
  // (falling back to requestAnimationFrame in browsers without it).
  //
  // This is what actually removes the head-movement lag: the old
  // fixed 85ms timer capped facemark updates at ~12fps no matter how
  // fast the camera/backend could go, and the camera itself only
  // delivers a new frame every ~33ms — so the timer and the real
  // frame supply were out of sync. Tying the loop directly to frame
  // delivery means every new frame gets processed as soon as it's
  // available, bounded only by hardware.
  //
  // The emotion inference timer (INFERENCE_INTERVAL_MS) is separate
  // and unaffected — it keeps running on its own fixed interval.

  _startFacemeshLoop() {

    if (!this.videoElement) {
      return;
    }

    this._facemeshUsesVideoFrameCallback =
      typeof this.videoElement.requestVideoFrameCallback === 'function';

    const step = async () => {

      if (!this.detectionActive) {
        this._facemeshLoopHandle = null;
        return;
      }

      await this._runFaceDetection();

      if (!this.detectionActive) {
        this._facemeshLoopHandle = null;
        return;
      }

      this._facemeshLoopHandle =
        this._facemeshUsesVideoFrameCallback
          ? this.videoElement.requestVideoFrameCallback(step)
          : requestAnimationFrame(step);
    };

    this._facemeshLoopHandle =
      this._facemeshUsesVideoFrameCallback
        ? this.videoElement.requestVideoFrameCallback(step)
        : requestAnimationFrame(step);
  },

  _stopFacemeshLoop() {

    if (!this._facemeshLoopHandle) {
      return;
    }

    if (
      this._facemeshUsesVideoFrameCallback &&
      typeof this.videoElement?.cancelVideoFrameCallback === 'function'
    ) {

      this.videoElement.cancelVideoFrameCallback(
        this._facemeshLoopHandle
      );

    } else {

      cancelAnimationFrame(
        this._facemeshLoopHandle
      );
    }

    this._facemeshLoopHandle = null;
  },

  // ============================================================
  // EMOTION INFERENCE
  // ============================================================

  async _runInference() {

    if (
      !this.detectionActive ||
      this.isRunningInference ||
      !this.tfjsModel ||
      !this._lastFacePrediction
    ) {
      return;
    }

    const generation = this._cameraGeneration;

    this.isRunningInference = true;

    try {
      const face = this._lastFacePrediction;

      const scores =
        await this._predictFace(face);

      // The inference may have completed after Stop
      // or after a newer camera session started.
      if (
        !this.detectionActive ||
        !state.cameraActive ||
        generation !== this._cameraGeneration
      ) {
        return;
      }

      if (
        !scores ||
        scores.length !== CLASS_NAMES.length
      ) {
        return;
      }

      const smoothed =
        this._smooth(scores);

      const ranked =
        CLASS_NAMES
          .map((name, index) => ({
            name,
            score: smoothed[index]
          }))
          .sort(
            (a, b) =>
              b.score - a.score
          );

      const top = ranked[0];

      this._applyEmotion(
        top.name,
        top.score
      );

      console.debug(
        '[EmotionDetector] prediction:',
        top.name,
        Math.round(
          top.score * 100
        ) + '%'
      );

    } catch (error) {
      if (
        generation !== this._cameraGeneration
      ) {
        return;
      }

      console.warn(
        '[EmotionDetector] inference error:',
        error
      );

    } finally {

      this.isRunningInference = false;
    }
},

  async _predictFace(face) {

    const rect =
      this._getFaceCropRect(face);

    const video =
      this.videoElement;

    const output =
      tf.tidy(() => {

        let image =
          tf.browser.fromPixels(video);

        image =
          tf.slice(
            image,

            [
              rect.sy,
              rect.sx,
              0
            ],

            [
              rect.sh,
              rect.sw,
              3
            ]
          );

        image =
          tf.image.resizeBilinear(
            image,
            [96, 96],
            true
          );

        /*
         * Production model expects float32 RGB
         * values in [0,1].
         *
         * Keras Rescaling layer performs:
         * [0,1] -> [-1,1]
         */
        image =
          image
            .toFloat()
            .div(255.0)
            .expandDims(0);

        return this.tfjsModel.predict(image);
      });

    try {
      const values =
        await output.data();

      return Array.from(values);

    } finally {
      output.dispose();
    }
  },

  // ============================================================
  // TEMPORAL SMOOTHING
  // ============================================================

  _smooth(scores) {

    if (!this._smoothedScores) {

      this._smoothedScores =
        scores.slice();

      return this._smoothedScores;
    }

    for (
      let i = 0;
      i < scores.length;
      i++
    ) {

      this._smoothedScores[i] =
        SMOOTHING_ALPHA * scores[i] +
        (1 - SMOOTHING_ALPHA) *
        this._smoothedScores[i];
    }

    const sum =
      this._smoothedScores.reduce(
        (a, b) => a + b,
        0
      );

    if (sum > 0) {

      for (
        let i = 0;
        i < this._smoothedScores.length;
        i++
      ) {

        this._smoothedScores[i] /=
          sum;
      }
    }

    return this._smoothedScores;
  },

  // ============================================================
  // APPLY EMOTION
  // ============================================================

  _applyEmotion(
    emotion,
    confidence
  ) {

    updateState({
      currentEmotion: emotion
    });

    this._updateEmotionUI(
      emotion,
      confidence
    );

    const settings =
      this._getSettings();

    if (
      settings.enableEmotionFeedback !== false
    ) {

      this._throttledLog(
        emotion,
        confidence
      );
    }
  },

  _getSettings() {

    try {

      return state.settings || {};

    } catch (_) {

      return {};
    }
  },

  async _throttledLog(
    emotion,
    confidence
  ) {

    const now = Date.now();

    if (
      now - this._lastLogTime <
      this._logIntervalMs
    ) {
      return;
    }

    this._lastLogTime = now;

    try {

      const { api } =
        await import('./api.js');

      const context =
        state.questionsAnswered > 0
          ? 'answering_question'
          : 'session_active';

      await api.emotions.log(
        emotion,
        confidence,
        context
      );

    } catch (_) {

      // Logging failure must never
      // stop emotion detection.
    }
  },

  // ============================================================
  // FACE CROP
  // ============================================================

  _getFaceCropRect(face) {

    const vw =
      this.videoElement?.videoWidth ||
      96;

    const vh =
      this.videoElement?.videoHeight ||
      96;

    const box =
      face?.boundingBox;

    const tl =
      box?.topLeft;

    const br =
      box?.bottomRight;

    let x1 =
      Number(tl?.[0]);

    let y1 =
      Number(tl?.[1]);

    let x2 =
      Number(br?.[0]);

    let y2 =
      Number(br?.[1]);

    if (
      ![
        x1,
        y1,
        x2,
        y2
      ].every(Number.isFinite)
    ) {

      return {

        sx: Math.floor(vw * 0.1),

        sy: Math.floor(vh * 0.1),

        sw: Math.floor(vw * 0.8),

        sh: Math.floor(vh * 0.8)
      };
    }

    const width =
      Math.max(
        1,
        x2 - x1
      );

    const height =
      Math.max(
        1,
        y2 - y1
      );

    const padX =
      width * CROP_PAD_X;

    const padY =
      height * CROP_PAD_Y;

    const sx =
      clamp(
        Math.floor(x1 - padX),
        0,
        vw - 1
      );

    const sy =
      clamp(
        Math.floor(y1 - padY),
        0,
        vh - 1
      );

    const ex =
      clamp(
        Math.ceil(x2 + padX),
        sx + 1,
        vw
      );

    const ey =
      clamp(
        Math.ceil(y2 + padY),
        sy + 1,
        vh
      );

    return {

      sx,

      sy,

      sw:
        Math.max(
          1,
          ex - sx
        ),

      sh:
        Math.max(
          1,
          ey - sy
        )
    };
  },

  // ============================================================
  // CANVAS
  // ============================================================

  _syncCanvas() {

    if (
      !this.canvasElement ||
      !this.videoElement
    ) {
      return;
    }

    const rect =
      this.videoElement
        .getBoundingClientRect();

    const width =
      this.videoElement.videoWidth ||
      480;

    const height =
      this.videoElement.videoHeight ||
      360;

    this.canvasElement.width =
      width;

    this.canvasElement.height =
      height;

    this.canvasElement.style.width =
      `${rect.width}px`;

    this.canvasElement.style.height =
      `${rect.height}px`;
  },

  _startDrawLoop() {

    if (this.animationFrameId) {

      cancelAnimationFrame(
        this.animationFrameId
      );
    }

    const draw = () => {

      if (!this.detectionActive) {

        this.animationFrameId = null;

        return;
      }

      this._drawOverlay();

      this.animationFrameId =
        requestAnimationFrame(draw);
    };

    draw();
  },

  _drawOverlay() {

    const canvas =
      this.canvasElement;

    const video =
      this.videoElement;

    if (!canvas || !video) {
      return;
    }

    const ctx =
      canvas.getContext('2d');

    if (!ctx) {
      return;
    }

    ctx.clearRect(
      0,
      0,
      canvas.width,
      canvas.height
    );

    const face =
      this._lastFacePrediction;

    if (!face) {
      return;
    }

    const keypoints =
      Array.isArray(face.keypoints)
        ? face.keypoints
        : [];

    if (keypoints.length > 0) {
      ctx.fillStyle =
        'rgba(76,245,133,0.75)';

      for (const point of keypoints) {
        const x =
          Number(point.x);

        const y =
          Number(point.y);

        if (
          !Number.isFinite(x) ||
          !Number.isFinite(y)
        ) {
          continue;
        }

        ctx.beginPath();

        ctx.arc(
          x,
          y,
          1.15,
          0,
          Math.PI * 2
        );

        ctx.fill();
      }

      return;
    }

    // Safe fallback if landmarks are unavailable.
    if (!face.boundingBox) {
      return;
    }

    const [x1, y1] =
      face.boundingBox.topLeft;

    const [x2, y2] =
      face.boundingBox.bottomRight;

    ctx.strokeStyle =
      'rgba(76,245,133,0.9)';

    ctx.lineWidth = 2;

    ctx.strokeRect(
      x1,
      y1,
      Math.max(1, x2 - x1),
      Math.max(1, y2 - y1)
    );
  },

  // ============================================================
  // MODEL STATUS UI
  // ============================================================

  _updateModelStatus(
    type,
    message
  ) {

    const statusEl =
      document.getElementById(
        'modelLoadingStatus'
      );

    const textEl =
      document.getElementById(
        'modelStatusText'
      );

    if (statusEl) {

      statusEl.className =
        `model-loading-status ${type} show`;
    }

    if (textEl) {

      textEl.textContent =
        message;
    }

    const debug =
      document.getElementById(
        'debugModels'
      );

    if (debug) {

      debug.textContent =
        message;
    }
  },

  // ============================================================
  // EMOTION UI
  // ============================================================

  _updateEmotionUI(
    emotion,
    confidence
  ) {

    const icon =
      document.getElementById(
        'emotionIcon'
      );

    const text =
      document.getElementById(
        'emotionText'
      );

    const indicator =
      document.getElementById(
        'emotionIndicator'
      );

    if (icon) {

      icon.textContent =
        EMOJI[emotion] || '😐';
    }

    if (text) {

      const label =
        emotion.charAt(0).toUpperCase() +
        emotion.slice(1);

      text.innerHTML =
        `${label}<span style="font-size:0.78em;opacity:0.72;margin-left:5px">${Math.round(confidence * 100)}%</span>`;
    }

    if (indicator) {

      indicator.style.display =
        'flex';
    }

    const debugEmotion =
      document.getElementById(
        'debugEmotion'
      );

    if (debugEmotion) {

      debugEmotion.textContent =
        `${emotion} (${Math.round(confidence * 100)}%)`;
    }
  },

  // ============================================================
  // CAMERA UI
  // ============================================================

  _updateCameraUI(status) {

    const dot =
      document.getElementById(
        'statusDot'
      );

    const text =
      document.getElementById(
        'cameraStatusText'
      );

    const placeholder =
      document.getElementById(
        'webcamPlaceholder'
      );

    const webcam =
      document.getElementById(
        'webcam'
      );

    const indicator =
      document.getElementById(
        'emotionIndicator'
      );

    if (
      status === 'starting' ||
      status === 'searching' ||
      status === 'active'
    ) {
      webcam?.classList.add('active');
    
      placeholder?.classList.add('hidden');
    
      if (indicator) {
        indicator.style.display =
          status === 'active'
            ? 'flex'
            : 'none';
      }
    
      if (dot) {

        dot.className =
          status === 'active'
            ? 'status-dot active'
            : 'status-dot loading';
      }
    
      if (text) {
        if (status === 'starting') {
          text.textContent = 'Initialising…';
        } else if (status === 'searching') {
          text.textContent = 'Detecting face…';
        } else {
          text.textContent = 'Active';
        }
      }
    
    } else if (status === 'off') {
      webcam?.classList.remove('active');
    
      placeholder?.classList.remove('hidden');
    
      if (indicator) {
        indicator.style.display = 'none';
      }
    
      if (dot) {
        dot.className = 'status-dot';
      }
    
      if (text) {
        text.textContent = 'Camera Off';
      }
    
    } else if (status === 'error') {
      webcam?.classList.remove('active');
    
      placeholder?.classList.remove('hidden');
    
      if (indicator) {
        indicator.style.display = 'none';
      }
    
      if (dot) {
        dot.className = 'status-dot error';
      }
    
      if (text) {
        text.textContent = 'Camera Error';
      }
    }
  },

  // ============================================================
  // MEDIA TRACK CLEANUP
  // ============================================================

  _stopTracks(stream) {

    if (!stream) {
      return;
    }

    try {

      for (
        const track
        of stream.getTracks()
      ) {

        track.stop();
      }

    } catch (_) {}
  }
};