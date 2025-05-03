import * as cocoSsd from '@tensorflow-models/coco-ssd';

export class MobilenetWorker{

    videoSource = null;

    detectionInWork = false;

    model = null;

    #eventTarget = null;

    constructor(callback = () => {}) {
        this.#eventTarget = new EventTarget();

        this.loadModel(callback);
    }

    async loadModel(callback){

        cocoSsd.load().then((model) => {
            console.log('MobileNet finished loading');

            this.model = model;

            callback();
        })
    }

    async detect (source, callback = () => {}) {
        let result = await this.model.detect(source, 20, 0.15);

        this._trigger('detected', result);

        callback();
    };

    detectVideo(vidSource){
        this.videoSource = vidSource;

        this.detectionInWork = true;

        let _startDetectFrame = async () => {

            // const offscreenCanvas = document.createElement('canvas');
            // offscreenCanvas.width = vidSource.width;
            // offscreenCanvas.height = vidSource.height;
            // const ctx = offscreenCanvas.getContext('2d');

            // ctx.drawImage(vidSource, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

            await this.detect(vidSource, () => {
                if(this.detectionInWork) {
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
