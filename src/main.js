import {YoloWorker} from "./YoloWorker.js";
import {YamnetWorker} from "./YamnetWorker.js";
// import {MobilenetWorker} from "./MobilenetWorker.js";
import {MobilenetWorker} from "./MobilenetWorkerSSD.js";
import { renderDetections } from './renderBox.js';
import yamnetGroup from "./utils/yamnet_clustered.json";

let video = null;

let audioContext = null;
let audioSource = null;

let videoVisibility = true;

let canvas = document.querySelector('#canvas');
let ctx = null;

let yoloWorker = new YoloWorker(yoloModelLoad);
let yamnetWorker = new YamnetWorker();
let mobilenetWorker = new MobilenetWorker();

let soundRes = document.querySelector('.sound-result');

let isMerge = false;
let isDebug = false;

let modelsResult = {
    yolo: null,
    mobilenet: null,
    yamnet: null,
}

async function startVideo(mode = 'environment') {
    video = document.createElement('video');
    video.muted = true;
    document.querySelector('.container').appendChild(video);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: {facingMode: mode} });
    video.srcObject = stream;

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
    mobilenetWorker.detectionInWork = false;
}

function startDetection(){
    canvas.width = video.width;
    canvas.height = video.height;

    yoloWorker.detectVideo(video);
    mobilenetWorker.detectVideo(video);
    yamnetWorker.realTimeDetector(audioContext, audioSource);
}

function loadModel(){
    yamnetWorker.on('detected', function (e){
        // console.log(e.detail);
        let result = e.detail[0].name.toLowerCase();

        modelsResult.yamnet = Object.keys(yamnetGroup).find(key => yamnetGroup[key].includes(result))??'other';

        soundRes.innerText = modelsResult.yamnet;
    });

    mobilenetWorker.on('detected', function(e){
        modelsResult.mobilenet = e.detail.map(fn => {
            fn.bbox = fixBlocksSize(...fn.bbox);

            return fn;
        });

        renderResult();
    })
    yoloWorker.on('detected', function(e){
        modelsResult.yolo = e.detail;
        renderResult()
    })
}
function renderResult(){
    if((modelsResult.yolo??false) && (modelsResult.mobilenet??false))
        renderDetections(ctx, modelsResult, {
            mode: isMerge?'merge':'split',
            debug: isDebug
        });
}

function yoloModelLoad(){
    canvas.width= yoloWorker.model.inputShape[1];
    canvas.height= yoloWorker.model.inputShape[2];

    video.width= yoloWorker.model.inputShape[1];
    video.height= yoloWorker.model.inputShape[2];
}

function fixBlocksSize(x, y, width, height){
    const videoWidth = video.width;
    const videoHeight = video.height;

    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;

    // const scaleX = displayWidth / videoWidth;
    // const scaleY = displayHeight / videoHeight;
    //
    // const offsetX = (displayWidth - videoWidth * scaleY) / 2;
    const offsetY = (displayHeight - videoHeight) / 2;
    //
    // console.log(y);
    //
    // x = x * scaleX + offsetX;
    y = y - offsetY;
    // width *= scaleX;
    height *= videoHeight/displayHeight;
    //
    // console.log(y)

    return [x, y, width, height];
}

async function init() {
    await loadModel();

    try{
    await startVideo();
    }
    catch (e){
        await startVideo('user')
    }

    ctx = canvas.getContext('2d');

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

    document.querySelector('input[name=merge]').addEventListener('change', function(){
        isMerge = !isMerge;
    })
    document.querySelector('input[name=debug]').addEventListener('change', function(){
        isDebug = !isDebug;
    })
}

init();
