
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js')
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.js')

const worker = self;
var scaleFactor = 2.0;
var scaleModel = null;

class DirtyRectsQueue {
    constructor() {
        this.queue = [];
        this.priority = [];
        this.push = this.push.bind(this);
        this.tryNext = this.tryNext.bind(this);
        this.numReceived = 0;
        this.numSent = 0;   
    }

    push(i,j, pixels, dirtyTime, priority) {
        const index = this.queue.findIndex(rect => rect.i === i && rect.j === j)
        var element = {i,j, pixels, dirtyTime}
        if (index >= 0) {
            this.queue.splice(index, 1);
            this.priority.splice(index, 1);  
        }
        var insertIndex = 0
        while (this.priority[insertIndex] < priority && insertIndex < this.priority.length) {
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

        let pixels = new ImageData( 128, 128 );
        pixels.data.set( new Uint8ClampedArray( queueItem.pixels ) );
        tf.tidy(() => {
            const exampleLocal = tf.browser
                    .fromPixels(pixels, 3)
                    .cast("float32")
                    .div(255.0)
                    .expandDims(0)

            if (scaleModel) {
                const predictionLocal = scaleModel.predict({ input_1: exampleLocal });
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
    if (e.data.isInit) {
        tf.setBackend(e.data.modelType || 'webgl');
        scaleFactor = e.data.scaleFactor || 2.0;
        tf.loadGraphModel(`/tfjs/scale_${scaleFactor * 100}/model.json`).then(model => {
            scaleModel = model;
        });
        return;
    }
    tf.ready().then(() => {
        const view = new Uint8Array(e.data.pixels);
        var meanR = 0;
        var meanRsq = 0;
        var meanG = 0;
        var meanGsq = 0;
        var meanB = 0;
        var meanBsq = 0;
        const stepSize = 4;
        const numChannels = 4;
        const denominator = view.length / stepSize / numChannels;
        for (let i=0; i< view.length; i+=(numChannels*stepSize)) {
            meanR += view[i] / denominator;
            meanRsq += view[i] * view[i] / denominator;
            meanG += view[i+1] / denominator;
            meanGsq += view[i+1] * view[i+1] / denominator;
            meanB += view[i+2] / denominator;
            meanBsq += view[i+2] * view[i+2] / denominator;
        }
        const priority = meanRsq + meanGsq + meanBsq - meanR**2 - meanG**2 - meanB**2
        if (priority > 1) {
            requestsQueue.push(e.data.i, e.data.j, e.data.pixels, e.data.dirtyTime, priority);
        }
    });
  }