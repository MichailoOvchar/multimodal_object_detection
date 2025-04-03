import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./renderBox.js";
import labels from "./labels.json";

export class YoloWorker{
    numClass = labels.length;

    videoSource = null;
    modelSettings = {};
    canvasSource = null;

    detectionInWork = false;

    model = {
        net: null,
        inputShape: [1, 0, 0, 3],
    };

    constructor(callback = () => {}) {
        this.loadModel(callback);
    }

    async loadModel(callback){
        tf.ready().then(async () => {
            console.log('Start model loading')
            const yolo = await tf.loadGraphModel(
                `/yolo11n_web_model/model.json`,
                {
                    onProgress: (fractions) => {
                        console.log('Finished modal loading')
                    },
                }
            ); // load model

            // warming up model
            const dummyInput = tf.ones(yolo.inputs[0].shape);
            const warmupResults = yolo.execute(dummyInput);

            this.model = {
                net: yolo,
                inputShape: yolo.inputs[0].shape,
            }; // set model & input shape


            tf.dispose([warmupResults, dummyInput]);

            callback();

            // console.log(model);
            //
        });
    }

    /**
     * Preprocess image / frame before forwarded into the model
     * @param {HTMLVideoElement|HTMLImageElement} source
     * @param {Number} modelWidth
     * @param {Number} modelHeight
     * @returns input tensor, xRatio and yRatio
     */
    _preprocess (source, modelWidth, modelHeight) {
        let xRatio, yRatio; // ratios for boxes

        const input = tf.tidy(() => {
            const img = tf.browser.fromPixels(source);

            // padding image to square => [n, m] to [n, n], n > m
            const [h, w] = img.shape.slice(0, 2); // get source width and height
            const maxSize = Math.max(w, h); // get max size
            const imgPadded = img.pad([
                [0, maxSize - h], // padding y [bottom only]
                [0, maxSize - w], // padding x [right only]
                [0, 0],
            ]);

            xRatio = maxSize / w; // update xRatio
            yRatio = maxSize / h; // update yRatio

            return tf.image
                .resizeBilinear(imgPadded, [modelWidth, modelHeight]) // resize frame
                .div(255.0)
                .mul(1.5).clipByValue(0, 1)
                .expandDims(0); // add batch
        });

        return [input, xRatio, yRatio];
    }

    /**
     * Function run inference and do detection from source.
     * @param {HTMLImageElement|HTMLVideoElement} source
     * @param {tf.GraphModel} model loaded YOLO tensorflow.js model
     * @param {HTMLCanvasElement} canvasRef canvas reference
     * @param {VoidFunction} callback function to run after detection process
     */
    async detect (source, canvasRef, callback = () => {}) {
        let model = this.model;
        const [modelWidth, modelHeight] = model.inputShape.slice(1, 3); // get model width and height

        tf.engine().startScope(); // start scoping tf engine
        const [input, xRatio, yRatio] = this._preprocess(source, modelWidth, modelHeight); // preprocess image

        const res = model.net.execute(input); // inference model
        const transRes = res.transpose([0, 2, 1]); // transpose result [b, det, n] => [b, n, det]
        const boxes = tf.tidy(() => {
            const w = transRes.slice([0, 0, 2], [-1, -1, 1]); // get width
            const h = transRes.slice([0, 0, 3], [-1, -1, 1]); // get height
            const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
            const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
            return tf
                .concat(
                    [
                        y1,
                        x1,
                        tf.add(y1, h), //y2
                        tf.add(x1, w), //x2
                    ],
                    2
                )
                .squeeze();
        }); // process boxes [y1, x1, y2, x2]

        const [scores, classes] = tf.tidy(() => {
            // class scores
            const rawScores = transRes.slice([0, 0, 4], [-1, -1, this.numClass]).squeeze(0); // #6 only squeeze axis 0 to handle only 1 class models
            return [rawScores.max(1), rawScores.argMax(1)];
        }); // get max scores and classes index

        const nms = await tf.image.nonMaxSuppressionAsync(
            boxes,
            scores,
            500,
            0.45,
            0.2
        ); // NMS to filter boxes

        const boxes_data = boxes.gather(nms, 0).dataSync();
        const scores_data = scores.gather(nms, 0).dataSync();
        const classes_data = classes.gather(nms, 0).dataSync();

        renderBoxes(canvasRef, boxes_data, scores_data, classes_data, [
            xRatio,
            yRatio,
        ]); // render boxes
        tf.dispose([res, transRes, boxes, scores, classes, nms]); // clear memory

        callback();

        tf.engine().endScope(); // end of scoping
    };

    async blackAndWhiteDetector(vidSource, model, canvasRef, canvasHelp){
        this.videoSource = vidSource;
        this.modelSettings = model;
        this.canvasSource = canvasRef;

        this.detectionInWork = true;
        let myImage = document.createElement('img');
        myImage.width = vidSource.width;
        myImage.height = vidSource.height;
        let _startDetectFrame = async () => {
            myImage.onload = null;
            const ctx = canvasHelp.getContext("2d");
            ctx.drawImage(vidSource, 0, 0, canvasHelp.width, canvasHelp.height);

            const imageData = ctx.getImageData(0, 0, canvasHelp.width, canvasHelp.height);
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {
                const avg = 0.3 * data[i] + 0.59 * data[i + 1] + 0.11 * data[i + 2];

                // Додаємо 6 рівнів контрастності
                // let contrast;
                // if (avg < 43) contrast = 0;
                // else if (avg < 85) contrast = 51;
                // else if (avg < 128) contrast = 102;
                // else if (avg < 170) contrast = 153;
                // else if (avg < 213) contrast = 204;
                // else contrast = 255;

                data[i] = data[i + 1] = data[i + 2] = avg;
            }

            ctx.putImageData(imageData, 0, 0);
            myImage.onload = async () => {
                await this.detect(myImage, model, canvasRef, () => {
                    if(this.detectionInWork) {
                        requestAnimationFrame(_startDetectFrame); // get another frame
                    }
                });
            };

            myImage.src = canvasHelp.toDataURL('image/png');
        }
        _startDetectFrame();
    }

    /**
     * Function to detect video from every source.
     * @param {HTMLVideoElement} vidSource video source
     * @param {tf.GraphModel} model loaded YOLO tensorflow.js model
     * @param {HTMLCanvasElement} canvasRef canvas reference
     */
    detectVideo(vidSource, canvasRef){
        this.videoSource = vidSource;
        this.canvasSource = canvasRef;

        this.detectionInWork = true;

        /**
         * Function to detect every frame from video
         */
        let _startDetectFrame = async () => {
            if (vidSource.videoWidth === 0 && vidSource.srcObject === null) {
                const ctx = canvasRef.getContext("2d");
                ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
                return; // handle if source is closed
            }

            await this.detect(vidSource, canvasRef, () => {
                if(this.detectionInWork) {
                    requestAnimationFrame(_startDetectFrame); // get another frame
                }
            });

        }
        _startDetectFrame();
    };


}
