import * as tf from "@tensorflow/tfjs";
import labels from "./utils/labels.json";

export class YoloWorker{
    numClass = labels.length;

    videoSource = null;
    modelSettings = {};
    canvasSource = null;

    detectionInWork = false;

    #eventTarget = null;

    model = {
        net: null,
        inputShape: [1, 0, 0, 3],
    };

    constructor(callback = () => {}) {
        this.#eventTarget = new EventTarget();

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
            };


            tf.dispose([warmupResults, dummyInput]);

            callback();
        });
    }


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


    async detect (source, callback = () => {}) {
        let model = this.model;
        const [modelWidth, modelHeight] = model.inputShape.slice(1, 3);

        tf.engine().startScope();
        const [input, xRatio, yRatio] = this._preprocess(source, modelWidth, modelHeight);

        const res = model.net.execute(input);
        const transRes = res.transpose([0, 2, 1]);

        const boxes = tf.tidy(() => {
            const w = transRes.slice([0, 0, 2], [-1, -1, 1]);
            const h = transRes.slice([0, 0, 3], [-1, -1, 1]);
            const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2)); // x1
            const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2)); // y1
            return tf
                .concat(
                    [
                        y1,
                        x1,
                        tf.add(y1, h),
                        tf.add(x1, w),
                    ],
                    2
                )
                .squeeze();
        });

        const [scores, classes] = tf.tidy(() => {
            const rawScores = transRes.slice([0, 0, 4], [-1, -1, this.numClass]).squeeze(0);
            return [rawScores.max(1), rawScores.argMax(1)];
        });

        const nms = await tf.image.nonMaxSuppressionAsync(
            boxes,
            scores,
            20,
            0.45,
            0.3
        );

        const boxes_data = boxes.gather(nms, 0).dataSync();
        const scores_data = scores.gather(nms, 0).dataSync();
        const classes_data = classes.gather(nms, 0).dataSync();

        let result = [];
        for(let i = 0; i < scores_data.length; i++){

            let [y1, x1, y2, x2] = boxes_data.slice(i * 4, (i + 1) * 4);
            x1 *= xRatio;
            x2 *= xRatio;
            y1 *= yRatio;
            y2 *= yRatio;
            const width = x2 - x1;
            const height = y2 - y1;

            result.push({
                bbox: [x1, y1, width, height],
                score: scores_data[i],
                class: labels[classes_data[i]]
            });
        }


        this._trigger('detected', result);

        tf.dispose([res, transRes, boxes, scores, classes, nms]);

        callback();

        tf.engine().endScope();
    };

    // async blackAndWhiteDetector(vidSource, model, canvasRef, canvasHelp){
    //     this.videoSource = vidSource;
    //     this.modelSettings = model;
    //     this.canvasSource = canvasRef;

    //     this.detectionInWork = true;
    //     let myImage = document.createElement('img');
    //     myImage.width = vidSource.width;
    //     myImage.height = vidSource.height;
    //     let _startDetectFrame = async () => {
    //         myImage.onload = null;
    //         const ctx = canvasHelp.getContext("2d");
    //         ctx.drawImage(vidSource, 0, 0, canvasHelp.width, canvasHelp.height);

    //         const imageData = ctx.getImageData(0, 0, canvasHelp.width, canvasHelp.height);
    //         const data = imageData.data;

    //         for (let i = 0; i < data.length; i += 4) {
    //             const avg = 0.3 * data[i] + 0.59 * data[i + 1] + 0.11 * data[i + 2];

    //             // Додаємо 6 рівнів контрастності
    //             // let contrast;
    //             // if (avg < 43) contrast = 0;
    //             // else if (avg < 85) contrast = 51;
    //             // else if (avg < 128) contrast = 102;
    //             // else if (avg < 170) contrast = 153;
    //             // else if (avg < 213) contrast = 204;
    //             // else contrast = 255;

    //             data[i] = data[i + 1] = data[i + 2] = avg;
    //         }

    //         ctx.putImageData(imageData, 0, 0);
    //         myImage.onload = async () => {
    //             await this.detect(myImage, model, canvasRef, () => {
    //                 if(this.detectionInWork) {
    //                     requestAnimationFrame(_startDetectFrame); // get another frame
    //                 }
    //             });
    //         };

    //         myImage.src = canvasHelp.toDataURL('image/png');
    //     }
    //     _startDetectFrame();
    // }

    detectVideo(vidSource){
        this.videoSource = vidSource;

        this.detectionInWork = true;

        let _startDetectFrame = async () => {
            await this.detect(vidSource, () => {
                if(this.detectionInWork) {
                    // requestAnimationFrame(_startDetectFrame);
                    setTimeout(_startDetectFrame, 200);
                }
            });

        }
        _startDetectFrame();
    };


    on(event, callback){
        this.#eventTarget.addEventListener(event, callback);
    }
    off(event, callback){
        this.#eventTarget.removeEventListener(event, callback);
    }
    _trigger(event, detail){
        this.#eventTarget.dispatchEvent(new CustomEvent(event, {detail: detail}));
    }
}
