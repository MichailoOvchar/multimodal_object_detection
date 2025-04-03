import * as tf from "@tensorflow/tfjs";
import labels from "./yamnet-labels.json";

export class YamnetWorker{
    numClass = labels.length;

    model = null;
    detectionInWork = false;

    #eventTarget = null;

    constructor() {
        this.#eventTarget = new EventTarget();

        this.loadModel();
    }

    loadModel(){
        tf.ready().then(async () => {
            console.log('Start model loading')
            this.model = await tf.loadGraphModel(
                `/yamnet-tfjs-tfjs-v1/model.json`,
                {
                    onProgress: (fractions) => {
                        console.log('Finished vggish modal loading')
                    },
                    onError: (err) => {
                        console.warn(err);
                    }
                }
            );
        });
    }

    async detect (buffer) {

        const inputTensor = tf.tensor1d(buffer).reshape([-1]);
        // resampled = tf.image.resizeBilinear(resampled.expandDims(2), [16000, 1]);

        // Робимо передбачення
        const [scores, embeddings, spectrogram] = await this.model.predict({ 'waveform': inputTensor });
        const predictionArray = (await scores.array())[0];

        const topClasses = predictionArray
            .map((score, index) => ({ name: labels[index] || `Class ${index}`, score, index }))
            .sort((a, b) => b.score - a.score) // Сортуємо від більшого до меншого
            .slice(0, 3);

        this._trigger('detected', topClasses);

        // resampled.dispose();
        scores.dispose();
        embeddings.dispose();
        spectrogram.dispose();
    };

    async realTimeDetector(context, source){

        const analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        this.detectionInWork = true;

        let dataArray = new Float32Array(analyser.fftSize);

        let timer = setTimeout(() => {}, 1)

        let _startDetectFrame = async () => {
            analyser.getFloatTimeDomainData(dataArray);

            if((this.model??false) && this.detectionInWork) {
                await this.detect(dataArray);
            }

            timer = setTimeout(_startDetectFrame, 50);
        }
        _startDetectFrame();
    }

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
