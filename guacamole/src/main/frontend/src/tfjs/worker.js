importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js");
importScripts(
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js"
);

const worker = self;
var scaleFactor = 2.0;
var scaleModel = null;

var offscreen = new OffscreenCanvas(1024, 1024);
var offscreenTwo = new OffscreenCanvas(1024, 1024);
var contextTwo = offscreenTwo.getContext("webgl");
var contextTwoBitmap = offscreenTwo.getContext("bitmaprenderer");

var context = offscreen.getContext("2d");
var tileSize = 4 * 64;
var numBlobProcessing = 0;

var updateBitmaps = {};
var updateTensors = {};
var updateRefs = {};

class DirtyRectsQueue {
  constructor() {
    this.queue = [];
    this.priority = [];
    this.push = this.push.bind(this);
    this.tryNext = this.tryNext.bind(this);
    this.numReceived = 0;
    this.numSent = 0;
  }

  push(i, j, startX, endX, startY, endY, dirtyTime, priority, id) {
    const index = this.queue.findIndex((rect) => rect.i === i && rect.j === j);
    var element = { i, j, startX, endX, startY, endY, dirtyTime, id };
    if (index >= 0) {
      const prevElement = this.queue.splice(index, 1);
      const prevPrio = this.priority.splice(index, 1);
      if (prevElement.dirtyTime > dirtyTime) {
        updateRefs[id] = updateRefs[id] - 1;
        if (updateRefs[id] === 0) {
          updateTensors[id].dispose();
          updateBitmaps[id] = null;
          updateRefs[id] = null;
        }
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
    if (!scaleModel || this.numReceived - this.numSent > 1) {
      setTimeout(this.tryNext, 0);
      return;
    }

    const queueItem = this.queue.pop();
    this.priority.pop();
    this.numReceived += 1;
    updateRefs[queueItem.id] = updateRefs[queueItem.id] - 1;
    if (!updateTensors[queueItem.id]) {
      updateTensors[queueItem.id] = tf.browser.fromPixels(
        updateBitmaps[queueItem.id].bitmap
      );
    }
    tf.tidy(() => {
      var pixels = null;
     
      pixels = updateTensors[queueItem.id].slice(
        [
          queueItem.j * tileSize +
            queueItem.startY / scaleFactor -
            updateBitmaps[queueItem.id].y,
          queueItem.i * tileSize +
            queueItem.startX / scaleFactor -
            updateBitmaps[queueItem.id].x,
          0,
        ],
        [
          queueItem.endY / scaleFactor - queueItem.startY / scaleFactor,
          queueItem.endX / scaleFactor - queueItem.startX / scaleFactor,
          3,
        ]
      );
      
      const p = tf.tensor(2).toInt();
      const priority = tf.sum(
        tf.sub(tf.mean(pixels.pow(p), [0, 1]), tf.mean(pixels, [0, 1]).pow(p))
      );
      const priorityValue = priority.dataSync()[0];

      if (priorityValue < 1) {
        this.numSent += 1;
        if (updateRefs[queueItem.id] === 0) {
          updateTensors[queueItem.id].dispose();
          updateBitmaps[queueItem.id] = null;
          updateRefs[queueItem.id] = null;
        }
        return 0;
      }

      const exampleLocal = pixels
        .cast("float32")
        .div(255.0)
        .expandDims(0);

      var predictionClippedLocal = null;
      if (scaleModel) {
        const predictionLocal = scaleModel.execute({ input_1: exampleLocal });
        predictionClippedLocal = predictionLocal
          .clipByValue(0, 1)
          .squeeze(0)
          .slice(
            [queueItem.startY, queueItem.startX],
            [
              queueItem.endY - queueItem.startY,
              queueItem.endX - queueItem.startX,
            ]
          );
        tf.browser.toPixels(predictionClippedLocal, null).then((pixels) => {
          const buffer = pixels.buffer;
          postMessage(
            {
              i: queueItem.i,
              j: queueItem.j,
              startX: queueItem.startX,
              endX: queueItem.endX,
              startY: queueItem.startY,
              endY: queueItem.endY,
              pixels: buffer,
              dirtyTime: queueItem.dirtyTime,
            },
            [buffer]
          );
          if (updateRefs[queueItem.id] === 0) {
            updateTensors[queueItem.id].dispose();
            updateBitmaps[queueItem.id] = null;
            updateRefs[queueItem.id] = null;
          }
          this.numSent += 1;
        });
      }
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
    const id = crypto.randomUUID();
    context.drawImage(e.data.bitmap, e.data.x, e.data.y);
    const iLower = Math.floor(e.data.x / tileSize);
    const iUpper = Math.ceil((e.data.x + e.data.bitmap.width) / tileSize) - 1;
    const jLower = Math.floor(e.data.y / tileSize);
    const jUpper = Math.ceil((e.data.y + e.data.bitmap.height) / tileSize) - 1;
    updateRefs[id] = (jUpper - jLower + 1) * (iUpper - iLower + 1);

    createImageBitmap(offscreen,iLower * tileSize, jLower * tileSize, (iUpper - iLower + 1) * tileSize, (jUpper - jLower + 1) * tileSize).then(bitmap => {
    //const bitmap = offscreen.transferToImageBitmap()
    e.data.x = iLower * tileSize;
    e.data.y = jLower * tileSize;
    
    tf.ready().then(() => {
      const startTime = performance.now();
      updateBitmaps[id] = {
        bitmap: bitmap,
        x: e.data.x,
        y: e.data.y,
      };


      for (let i = iUpper; i >= iLower; i--) {
        for (let j = jLower; j <= jUpper; j++) {
          // relative to within tile
          const startX = Math.max(e.data.x - i * tileSize, 0);
          const endX =
            Math.min(e.data.x + bitmap.width, (i + 1) * tileSize) -
            i * tileSize;
          const startY = Math.max(e.data.y - j * tileSize, 0);
          const endY =
            Math.min(e.data.y + bitmap.height, (j + 1) * tileSize) -
            j * tileSize;
          requestsQueue.push(
            i,
            j,
            startX * scaleFactor,
            endX * scaleFactor,
            startY * scaleFactor,
            endY * scaleFactor,
            e.data.dirtyTime,
            e.data.dirtyTime,
            id
          );
        }
      }
      numBlobProcessing -= 1;
    });
  })
  }
};
