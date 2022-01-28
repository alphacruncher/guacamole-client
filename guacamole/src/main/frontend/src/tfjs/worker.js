importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js");
importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"
);

const worker = self;
var scaleFactor = 2.0;
var scaleModel = null;

var offscreen = new OffscreenCanvas(1, 1);
var context = offscreen.getContext("2d");
var tileSize = 64;
var numBlobProcessing = 0;

class DirtyRectsQueue {
  constructor() {
    this.queue = [];
    this.priority = [];
    this.push = this.push.bind(this);
    this.tryNext = this.tryNext.bind(this);
    this.numReceived = 0;
    this.numSent = 0;
  }

  push(i, j, dirtyTime, priority) {
    const index = this.queue.findIndex((rect) => rect.i === i && rect.j === j);
    var element = { i, j, dirtyTime };
    if (index >= 0) {
      const prevElement = this.queue.splice(index, 1);
      const prevPrio = this.priority.splice(index, 1);
      if (prevElement.dirtyTime > dirtyTime) {
        element = prevElement[0];
        priority = prevPrio[0];
      }
    }
    var insertIndex = 0;
    while (
      this.priority[insertIndex] < priority &&
      insertIndex < this.priority.length
    ) {
      insertIndex += 1;
    }
    this.queue.splice(insertIndex, 0, element);
    this.priority.splice(insertIndex, 0, priority);
    setTimeout(this.tryNext, 0);
  }

  tryNext() {
    if (this.queue.length === 0) {
      return;
    }
    if (!scaleModel || this.numReceived - this.numSent > 4) {
      setTimeout(this.tryNext, 0);
      return;
    }

    const queueItem = this.queue.pop();
    this.priority.pop();
    this.numReceived += 1;
    const startTime = performance.now()
    const rawPixels = context.getImageData(
      queueItem.i * tileSize,
      queueItem.j * tileSize,
      tileSize,
      tileSize
    ).data.buffer;
    console.log(performance.now() - startTime);
    const view = new Uint8Array(rawPixels);
    var meanR = 0;
    var meanRsq = 0;
    var meanG = 0;
    var meanGsq = 0;
    var meanB = 0;
    var meanBsq = 0;
    const stepSize = 4;
    const numChannels = 4;
    const denominator = view.length / stepSize / numChannels;
    for (let i = 0; i < view.length; i += numChannels * stepSize) {
      meanR += view[i] / denominator;
      meanRsq += (view[i] * view[i]) / denominator;
      meanG += view[i + 1] / denominator;
      meanGsq += (view[i + 1] * view[i + 1]) / denominator;
      meanB += view[i + 2] / denominator;
      meanBsq += (view[i + 2] * view[i + 2]) / denominator;
    }
    const priority =
      meanRsq + meanGsq + meanBsq - meanR ** 2 - meanG ** 2 - meanB ** 2;
    if (priority < 1) {
      this.numSent += 1;
      return 0;
    }

    //let pixels = new ImageData(128, 128);
    // pixels.data.set(new Uint8ClampedArray(queueItem.pixels));
    tf.tidy(() => {
      const pixels = tf
        .tensor3d(new Uint8ClampedArray(rawPixels), [
          tileSize,
          tileSize,
          4,
        ])
        .slice([0, 0, 0], [tileSize, tileSize, 3]);
      const exampleLocal = pixels.cast("float32").div(255.0).expandDims(0);

      if (scaleModel) {
        const predictionLocal = scaleModel.execute({ input_1: exampleLocal });
        // predictionLocal.dataSync();
        const predictionClippedLocal = predictionLocal
          .clipByValue(0, 1)
          .squeeze(0);
        tf.browser.toPixels(predictionClippedLocal, null).then((pixels) => {
          const buffer = pixels.buffer;
          postMessage(
            {
              i: queueItem.i,
              j: queueItem.j,
              pixels: buffer,
              dirtyTime: queueItem.dirtyTime,
            },
            [buffer]
          );
          predictionClippedLocal.dispose();
          predictionLocal.dispose();
          exampleLocal.dispose();
          this.numSent += 1;
        });
      }
      return 0;
    });
  }
}

const requestsQueue = new DirtyRectsQueue();

onmessage = function (e) {
  if (e.data.isInit) {
    tf.setBackend(e.data.modelType || "webgl");
    scaleFactor = e.data.scaleFactor || 2.0;
    tf.ready().then(() => {
      tf.loadGraphModel(`/tfjs/scale_${scaleFactor * 100}/model.json`).then(
        (model) => {
          scaleModel = model;
        }
      );
    });
    return;
  }
  if (e.data.isResize) {
    offscreen.width = e.data.width;
    offscreen.height = e.data.height;
  }

  if (e.data.isBitmap) {
    numBlobProcessing += 1;
    tf.ready().then(() => {
      const bitmap = e.data.bitmap;
      context.drawImage(bitmap, e.data.x, e.data.y);
      const iLower = Math.floor(e.data.x / tileSize);
      const iUpper = Math.floor((e.data.x + bitmap.width - 1) / tileSize);
      const jLower = Math.floor(e.data.y / tileSize);
      const jUpper = Math.floor((e.data.y + bitmap.height - 1) / tileSize);

      for (let i = iUpper; i >= iLower; i--) {
        for (let j = jLower; j <= jUpper; j++) {
          requestsQueue.push(i, j, e.data.dirtyTime, e.data.dirtyTime);
        }
      }
      numBlobProcessing -= 1;
    });
  }
};
