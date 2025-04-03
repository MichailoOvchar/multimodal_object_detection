import {YoloWorker} from "./YoloWorker.js";
import {YamnetWorker} from "./YamnetWorker.js";

let video = null;

let audioContext = null;
let audioSource = null;

let videoVisibility = true;

let canvas = document.querySelector('#canvas');
let localCanvas = document.querySelector('#local-canvas');
let ctx = null;

let yoloWorker = new YoloWorker(yoloModelLoad);
let yamnetWorker = new YamnetWorker();

let soundRes = document.querySelector('.sound-result');

async function startVideo(mode = 'environment') {
    video = document.createElement('video');
    video.muted = true;
    document.querySelector('.container').appendChild(video);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: {facingMode: mode} });
    video.srcObject = stream;
    ctx = localCanvas.getContext('2d');

    audioContext = new AudioContext();
    audioSource = audioContext.createMediaStreamSource(stream);

    // const analyzer = Meyda.createMeydaAnalyzer({
    //     audioContext: audioContext,
    //     source: source,
    //     bufferSize: 1024,
    //     featureExtractors: ["mfcc"],
    //     callback: features => {
    //         yamnetWorker.detect(features);
    //     }
    // });

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            // analyzer.start();
            // processFrame();
            resolve();
        };
    });
}

function stopDetection(){
    yoloWorker.detectionInWork = false;
    yamnetWorker.detectionInWork = false;
}

function startDetection(){
    canvas.width = video.width;
    canvas.height = video.height;
    localCanvas.width = video.width;
    localCanvas.height = video.height;
    yoloWorker.detectVideo(video, canvas);
    // yoloWorker.blackAndWhiteDetector(video, model, canvas, localCanvas);
    yamnetWorker.realTimeDetector(audioContext, audioSource);
}

function loadModel(){
    yamnetWorker.on('detected', function (e){
        // console.log(e.detail);
        let result = e.detail[0].name;

        soundRes.innerText = result;
    })
}

function yoloModelLoad(){
    canvas.width= yoloWorker.model.inputShape[1];
    canvas.height= yoloWorker.model.inputShape[2];

    video.width= yoloWorker.model.inputShape[1];
    video.height= yoloWorker.model.inputShape[2];
}

async function init() {
    await loadModel();

    try{
    await startVideo();
    }
    catch (e){
        await startVideo('user')
    }

    document.querySelector('#toggleDetectingEl').addEventListener('click', function(e){
        if(yoloWorker.detectionInWork){
            e.target.innerText = 'Start Detection';
            stopDetection();
        }
        else{
            e.target.innerText = 'Stop Detection';
            startDetection();
        }
    })
}

init();
