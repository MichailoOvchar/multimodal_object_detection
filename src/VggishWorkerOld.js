import * as tf from "@tensorflow/tfjs";
import { renderBoxes } from "./utils/renderBox.js";
import labels from "./utils/labels.json";

export class VggishWorker{
    numClass = labels.length;

    model = null;
    detectionInWork = false;

    constructor() {
        this.loadModel();
    }

    loadModel(){
        tf.ready().then(async () => {
            console.log('Start model loading')
            this.model = await tf.loadGraphModel(
                `/vggish/model.json`,
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

    async _preprocess(analyser){
        const bufferLength = analyser.frequencyBinCount;
        const frequencyData = new Float32Array(bufferLength);
        analyser.getFloatFrequencyData(frequencyData);  // Отримуємо спектр

        const melSpectrogram = [];
        for (let i = 0; i < 64; i++) {
            const index = Math.floor(i * (bufferLength / 64));
            melSpectrogram.push(frequencyData[index]);
        }

        return melSpectrogram;
    }

    async detect (melHistory) {

        const inputTensor = tf.tensor(melHistory);

        // Робимо передбачення
        const prediction = this.model.predict(inputTensor);
        const predictionArray = await prediction.array();

        console.log(this.model.outputShape);

        console.log('Результат передбачення:', predictionArray);

        inputTensor.dispose();
        prediction.dispose();
    };

    async realTimeDetector(context, source){

        const analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        this.detectionInWork = true;

        const melHistory = [];

        let _startDetectFrame = async () => {
            const melSpectrogram = await this._preprocess(analyser);
            if(!(melSpectrogram[0] === Infinity || melSpectrogram[1] === -Infinity)) {
                melHistory.push(melSpectrogram);
            }

            if (melHistory.length > 96) {
                melHistory.shift(); // Видаляємо найстаріший кадр
            }

            if((this.model??false) && this.detectionInWork) {
                if (melHistory.length === 96) {
                    await this.detect([melHistory], [1, 96, 64]);
                }
            }

            requestAnimationFrame(_startDetectFrame);
        }
        _startDetectFrame();
    }
}
