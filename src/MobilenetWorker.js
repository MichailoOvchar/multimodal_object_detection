import * as mobilenet from "@tensorflow-models/mobilenet";

export class MobilenetWorker{

    videoSource = null;

    detectionInWork = false;

    model = null;

    constructor(callback = () => {}) {
        this.loadModel(callback);
    }

    async loadModel(callback){

        mobilenet.load().then((model) => {
            console.log('MobileNet finished loading');

            this.model = model;

            callback();
        })
    }

    async detect (source, callback = () => {}) {

        let result = await this.model.classify(source);

        console.log(result);

        callback();
    };

    detectVideo(vidSource){
        this.videoSource = vidSource;

        this.detectionInWork = true;

        /**
         * Function to detect every frame from video
         */
        let _startDetectFrame = async () => {

            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = vidSource.width;
            offscreenCanvas.height = vidSource.height;
            const ctx = offscreenCanvas.getContext('2d');

            ctx.drawImage(vidSource, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

            await this.detect(offscreenCanvas, () => {
                if(this.detectionInWork) {
                    setTimeout(_startDetectFrame, 1000);
                }
            });

        }
        _startDetectFrame();
    };


}
