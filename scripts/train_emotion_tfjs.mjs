import fs from 'fs';
import path from 'path';
import * as tf from '@tensorflow/tfjs';
import { Jimp } from 'jimp';

const ROOT = process.cwd();
const DATASET_DIR = path.join(ROOT, 'dataset');
const OUT_DIR = path.join(ROOT, 'frontend', 'js', 'emotion_tfjs');
const METRICS_DIR = path.join(ROOT, 'backend', 'ai_models');

const CLASS_NAMES = ['angry', 'confused', 'happy', 'neutral'];
const IMG_SIZE = 48;
const BATCH_SIZE = 24;
const EPOCHS = 6;
const LR = 1e-3;
const SEED = 42;
const MAX_TRAIN_PER_CLASS = 320;
const MAX_VAL_PER_CLASS = 80;
const MAX_TEST_PER_CLASS = 80;
const LOAD_CONCURRENCY = 4;

function seededRand(seed) {
  let x = Math.sin(seed) * 10000;
  return () => {
    x = Math.sin(x) * 10000;
    return x - Math.floor(x);
  };
}

function listImageFiles(dirPath) {
  if (!fs.existsSync(dirPath)) return [];
  return fs
    .readdirSync(dirPath)
    .filter((f) => /\.(jpg|jpeg|png)$/i.test(f))
    .map((f) => path.join(dirPath, f));
}

async function loadImageTensor(filePath) {
  const image = await Jimp.read(filePath);
  image.resize({ w: IMG_SIZE, h: IMG_SIZE });
  const { data, width, height } = image.bitmap;

  const rgb = new Float32Array(width * height * 3);
  let j = 0;
  for (let i = 0; i < data.length; i += 4) {
    rgb[j++] = data[i] / 255;
    rgb[j++] = data[i + 1] / 255;
    rgb[j++] = data[i + 2] / 255;
  }

  return tf.tensor3d(rgb, [height, width, 3]);
}

function augmentImage(img, rand) {
  return tf.tidy(() => {
    let out = img;

    if (rand() > 0.5) out = tf.reverse(out, [1]);
    const brightDelta = (rand() - 0.5) * 0.12;
    out = out.add(brightDelta);
    const contrastFactor = 0.9 + rand() * 0.2;
    out = out.sub(0.5).mul(contrastFactor).add(0.5);

    const noise = tf.randomNormal(out.shape, 0, 0.01);
    out = out.add(noise);

    return out.clipByValue(0, 1);
  });
}

function splitClassFiles(files, rand) {
  const shuffled = [...files];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const n = shuffled.length;
  const nTrain = Math.max(1, Math.floor(n * 0.7));
  const nVal = Math.max(1, Math.floor(n * 0.15));
  const train = shuffled.slice(0, nTrain);
  const val = shuffled.slice(nTrain, nTrain + nVal).slice(0, MAX_VAL_PER_CLASS);
  const test = shuffled.slice(nTrain + nVal).slice(0, MAX_TEST_PER_CLASS);

  return { train, val, test: test.length ? test : val.slice(0, 1) };
}

function buildDatasetEntries() {
  const rand = seededRand(SEED);
  const perClass = {};
  const split = { train: [], val: [], test: [] };

  for (const cls of CLASS_NAMES) {
    const clsDir = path.join(DATASET_DIR, cls);
    const files = listImageFiles(clsDir);
    if (!files.length) {
      throw new Error(`No images found for class '${cls}' in ${clsDir}`);
    }
    perClass[cls] = files.length;
    const parts = splitClassFiles(files, rand);

    parts.train.forEach((f) => split.train.push({ file: f, label: cls }));
    parts.val.forEach((f) => split.val.push({ file: f, label: cls }));
    parts.test.forEach((f) => split.test.push({ file: f, label: cls }));
  }

  return { perClass, split };
}

function oversampleBalanced(entries) {
  const byClass = new Map(CLASS_NAMES.map((c) => [c, []]));
  entries.forEach((e) => byClass.get(e.label).push(e));

  const targetCount = Math.min(
    MAX_TRAIN_PER_CLASS,
    Math.max(...CLASS_NAMES.map((c) => byClass.get(c).length))
  );
  const rand = seededRand(SEED + 9);

  const out = [];
  for (const cls of CLASS_NAMES) {
    const clsEntries = byClass.get(cls);
    const baseCount = Math.min(clsEntries.length, targetCount);

    for (let i = 0; i < baseCount; i += 1) {
      out.push({ ...clsEntries[i], augment: false });
    }

    let i = 0;
    while (baseCount + i < targetCount) {
      const sample = clsEntries[i % clsEntries.length];
      out.push({ ...sample, augment: true, augSeed: Math.floor(rand() * 1e9) });
      i += 1;
    }
  }
  return out;
}

async function toTensors(entries) {
  const xs = [];
  const ys = [];

  for (let i = 0; i < entries.length; i += LOAD_CONCURRENCY) {
    const chunk = entries.slice(i, i + LOAD_CONCURRENCY);
    const loaded = await Promise.all(
      chunk.map(async (e) => {
        const labelIdx = CLASS_NAMES.indexOf(e.label);
        const rand = seededRand((e.augSeed || SEED) + labelIdx);
        const img = await loadImageTensor(e.file);
        const tensor = e.augment ? augmentImage(img, rand) : img;
        if (tensor !== img) img.dispose();
        return { tensor, labelIdx };
      })
    );

    for (const item of loaded) {
      xs.push(item.tensor);
      ys.push(item.labelIdx);
    }
  }

  const x = tf.stack(xs);
  xs.forEach((t) => t.dispose());
  const y = tf.oneHot(tf.tensor1d(ys, 'int32'), CLASS_NAMES.length);
  return { x, y, yRaw: ys };
}

function buildModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({ inputShape: [IMG_SIZE, IMG_SIZE, 3], filters: 24, kernelSize: 3, padding: 'same', activation: 'relu' }),
      tf.layers.batchNormalization(),
      tf.layers.maxPooling2d({ poolSize: 2 }),

      tf.layers.conv2d({ filters: 48, kernelSize: 3, padding: 'same', activation: 'relu' }),
      tf.layers.batchNormalization(),
      tf.layers.maxPooling2d({ poolSize: 2 }),

      tf.layers.conv2d({ filters: 72, kernelSize: 3, padding: 'same', activation: 'relu' }),
      tf.layers.batchNormalization(),
      tf.layers.maxPooling2d({ poolSize: 2 }),

      tf.layers.separableConv2d({ filters: 96, kernelSize: 3, padding: 'same', activation: 'relu' }),
      tf.layers.globalAveragePooling2d({}),
      tf.layers.dropout({ rate: 0.35 }),
      tf.layers.dense({ units: 64, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.25 }),
      tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(LR),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

function confusionAndPerClass(yTrue, yPred) {
  const n = CLASS_NAMES.length;
  const cm = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < yTrue.length; i += 1) cm[yTrue[i]][yPred[i]] += 1;

  const perClass = {};
  for (let i = 0; i < n; i += 1) {
    const tp = cm[i][i];
    const fn = cm[i].reduce((a, b) => a + b, 0) - tp;
    let fp = 0;
    for (let r = 0; r < n; r += 1) fp += cm[r][i];
    fp -= tp;

    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    perClass[CLASS_NAMES[i]] = {
      precision: Number(precision.toFixed(4)),
      recall: Number(recall.toFixed(4)),
      f1: Number(f1.toFixed(4)),
      support: cm[i].reduce((a, b) => a + b, 0),
    };
  }

  return { cm, perClass };
}

async function saveTfjsLayersModel(model, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const weightsPath = path.join(outDir, 'group1-shard1of1.bin');
  const modelPath = path.join(outDir, 'model.json');

  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      if (!artifacts.weightData || !artifacts.weightSpecs || !artifacts.modelTopology) {
        throw new Error('Invalid model artifacts received for save.');
      }

      fs.writeFileSync(weightsPath, Buffer.from(artifacts.weightData));

      const modelJson = {
        format: 'layers-model',
        generatedBy: 'custom-tfjs-task5',
        convertedBy: null,
        modelTopology: artifacts.modelTopology,
        weightsManifest: [
          {
            paths: ['group1-shard1of1.bin'],
            weights: artifacts.weightSpecs,
          },
        ],
      };
      fs.writeFileSync(modelPath, JSON.stringify(modelJson));

      return {
        modelArtifactsInfo: tf.io.getModelArtifactsInfoForJSON(artifacts),
      };
    })
  );
}

async function main() {
  await tf.setBackend('cpu');
  await tf.ready();

  console.log('[EMOTION] Task 5 training pipeline started');

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.mkdirSync(METRICS_DIR, { recursive: true });

  const { perClass: rawCounts, split } = buildDatasetEntries();
  console.log('[EMOTION] Raw counts:', rawCounts);

  const trainBalanced = oversampleBalanced(split.train);
  console.log(`[EMOTION] Balanced train samples: ${trainBalanced.length}`);

  console.log('[EMOTION] Loading training tensors...');
  const train = await toTensors(trainBalanced);
  console.log('[EMOTION] Loading validation tensors...');
  const val = await toTensors(split.val.map((e) => ({ ...e, augment: false })));
  console.log('[EMOTION] Loading test tensors...');
  const test = await toTensors(split.test.map((e) => ({ ...e, augment: false })));

  console.log('[EMOTION] Building model...');
  const model = buildModel();

  console.log('[EMOTION] Starting fit...');
  const history = await model.fit(train.x, train.y, {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    validationData: [val.x, val.y],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;
        const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
        console.log(
          `[EMOTION] epoch=${epoch + 1} loss=${logs.loss.toFixed(4)} acc=${acc.toFixed(4)} ` +
          `val_loss=${logs.val_loss.toFixed(4)} val_acc=${valAcc.toFixed(4)}`
        );
      },
    },
  });

  const testPred = model.predict(test.x);
  const testPredRaw = await testPred.argMax(-1).data();
  const { cm, perClass: perClassMetrics } = confusionAndPerClass(test.yRaw, Array.from(testPredRaw));

  const evalOut = await model.evaluate(test.x, test.y, { batchSize: BATCH_SIZE });
  const testLoss = Number((await evalOut[0].data())[0].toFixed(4));
  const testAcc = Number((await evalOut[1].data())[0].toFixed(4));

  await saveTfjsLayersModel(model, OUT_DIR);

  const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z');
  const metrics = {
    timestamp,
    dataset_counts: rawCounts,
    split_counts: {
      train_raw: split.train.length,
      train_balanced: trainBalanced.length,
      val: split.val.length,
      test: split.test.length,
    },
    model: {
      input_shape: [IMG_SIZE, IMG_SIZE, 3],
      classes: CLASS_NAMES,
      architecture: 'lightweight_cnn_tfjs',
    },
    aggregate_metrics: {
      test_loss: testLoss,
      test_accuracy: testAcc,
    },
    per_class_metrics: perClassMetrics,
    confusion_matrix: cm,
    history: history.history,
  };

  const metricsPath = path.join(METRICS_DIR, `emotion_tfjs_metrics_${timestamp}.json`);
  const latestInfoPath = path.join(METRICS_DIR, 'emotion_model_info.json');
  fs.writeFileSync(metricsPath, JSON.stringify(metrics, null, 2));
  fs.writeFileSync(latestInfoPath, JSON.stringify(metrics, null, 2));

  console.log(`[EMOTION] Saved TFJS model to ${OUT_DIR}`);
  console.log(`[EMOTION] Saved metrics to ${metricsPath}`);

  train.x.dispose();
  train.y.dispose();
  val.x.dispose();
  val.y.dispose();
  test.x.dispose();
  test.y.dispose();
  testPred.dispose();

  console.log('[EMOTION] Task 5 complete');
}

main().catch((err) => {
  console.error('[EMOTION] Training failed:', err);
  process.exit(1);
});
