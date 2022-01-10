
var scaleModel;
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js')
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js')
const worker = self;

class DirtyRectsQueue {
    constructor() {
        this.queue = [];
        this.push = this.push.bind(this);
        this.tryNext = this.tryNext.bind(this);
        this.scaleModel = null;
        this.numReceived = 0;
        this.numSent = 0;
        tf.setBackend('webgl').then(() => {
                tf.loadGraphModel("/tfjs/model.json").then(model => {
                    this.scaleModel = model;
                    console.log(this.scaleModel);
                });
            });    
    }

    push(i,j, pixels, dirtyTime) {
        const index = this.queue.findIndex(rect => rect.i === i && rect.j === j)
        if (index < 0) {
            this.queue.push({i,j, pixels, dirtyTime});
        } else {
            this.queue[index].pixels = pixels
            this.queue[index].dirtyTime = dirtyTime
        }
        setTimeout(this.tryNext, 0);
    }

    tryNext() {
        if (this.queue.length === 0) {
            return;
        } 
        if (!this.scaleModel || this.numReceived - this.numSent > 20) {
            setTimeout(this.tryNext, 0);
            return;
        }

        const queueItem = this.queue.pop();
        this.numReceived += 1;

        let pixels = new ImageData( 64, 64 );
        pixels.data.set( new Uint8ClampedArray( queueItem.pixels ) );
        tf.tidy(() => {
            const exampleLocal = tf.browser
                    .fromPixels(pixels, 3)
                    .cast("float32")
                    .div(255.0)
                    .expandDims(0)
            if (this.scaleModel) {
                const predictionLocal = this.scaleModel.predict({ input_1: exampleLocal });
                const predictionClippedLocal = predictionLocal.clipByValue(0, 1).squeeze(0)
                tf.browser.toPixels(
                    predictionClippedLocal,
                    null
                ).then(pixels => {
                    postMessage({ i: queueItem.i, j: queueItem.j, pixels: pixels.buffer, dirtyTime: queueItem.dirtyTime })
                    predictionClippedLocal.dispose();
                    predictionLocal.dispose();
                    exampleLocal.dispose();
                    this.numSent += 1;
                })
            }
            return 0;
        });
    }
}

const requestsQueue = new DirtyRectsQueue();

onmessage = function(e) {
    tf.ready().then(() => {
        requestsQueue.push(e.data.i, e.data.j, e.data.pixels, e.data.dirtyTime);
    });
  }