import classes from "./utils/coco_audio_links.json";

function drawDetections(ctx, detections, audioClass, options = {color: 'lime', debug: false}) {
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';

    detections.forEach(({ bbox, score, modelScores, class: label }) => {
        const [x1, y1, width, height] = bbox;

        let debugText = '';

        let average = 0;
        if(modelScores??false){
            average = (parseFloat(modelScores.model1) + parseFloat(modelScores.model2))/2;

            debugText = `m1: ${parseFloat(modelScores.model1).toFixed(4)} | m2: ${parseFloat(modelScores.model2).toFixed(4)}`;
        }
        else {
            average = parseFloat(score);
            debugText = `m1: ${parseFloat(score).toFixed(4)}`;
        }

        if(classes[label]?.boost?.includes(audioClass)??false){
            average += 0.15;
            average = average < 1 || 1;

            debugText += ` | s: +0.15`;
        }
        if(classes[label]?.penalize?.includes(audioClass)??false){
            average -= 0.15;

            debugText += ` | s: -0.15`;

            if(average < 20) return;
        }

        const text = `${label} (${(average * 100).toFixed(1)}%)`;
        const textWidth = ctx.measureText(text).width;
        const textHeight = 16;

        ctx.fillStyle = options.color;
        ctx.fillRect(x1, y1 - textHeight, textWidth + 6, textHeight);
        ctx.fillStyle = 'black';
        ctx.fillText(text, x1 + 3, y1 - 3);

        if(options.debug){
            const textWidth = ctx.measureText(debugText).width;
            ctx.fillStyle = options.color;
            ctx.fillRect((x1 + width - (textWidth + 6)), y1, textWidth + 6, textHeight);
            ctx.fillStyle = 'black';
            ctx.fillText(debugText, (x1 + width - (textWidth + 3)), (y1 + textHeight) - 3);
        }

        ctx.strokeStyle = options.color;
        ctx.strokeRect(x1, y1, width, height);
    });
}


function mergeDetections(detections1, detections2, iouThreshold = 0.5) {
    const merged = [];

    // Функція для обчислення IoU між двома box
    function iou(boxA, boxB) {
        const [ax1, ay1, ax2, ay2] = boxA;
        const [bx1, by1, bx2, by2] = boxB;

        const x1 = Math.max(ax1, bx1);
        const y1 = Math.max(ay1, by1);
        const x2 = Math.min(ax2, bx2);
        const y2 = Math.min(ay2, by2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (ax2 - ax1) * (ay2 - ay1);
        const areaB = (bx2 - bx1) * (by2 - by1);
        const union = areaA + areaB - intersection;

        return intersection / union;
    }

    const used2 = new Set();

    for (let i = 0; i < detections1.length; i++) {
        const d1 = detections1[i];
        let matched = false;

        for (let j = 0; j < detections2.length; j++) {
            if (used2.has(j)) continue;
            const d2 = detections2[j];

            const sameClass = d1.class === d2.class;
            const boxIoU = iou(d1.bbox, d2.bbox);

            if (sameClass && boxIoU > iouThreshold) {
                merged.push({
                    bbox: [
                        (d1.bbox[0] + d2.bbox[0]) / 2,
                        (d1.bbox[1] + d2.bbox[1]) / 2,
                        (d1.bbox[2] + d2.bbox[2]) / 2,
                        (d1.bbox[3] + d2.bbox[3]) / 2,
                    ],
                    class: d1.class,
                    score: ((d1.score + d2.score) / 2).toFixed(2),
                    modelScores: {
                        model1: d1.score.toFixed(2),
                        model2: d2.score.toFixed(2)
                    }
                });
                used2.add(j);
                matched = true;
                break;
            }
        }

        if (!matched) {
            merged.push({
                ...d1,
                modelScores: {
                    model1: d1.score.toFixed(2),
                    model2: "-"
                }
            });
        }
    }

    // Add remaining unmatched from detections2
    detections2.forEach((d2, j) => {
        if (!used2.has(j)) {
            merged.push({
                ...d2,
                modelScores: {
                    model1: "-",
                    model2: d2.score.toFixed(2)
                }
            });
        }
    });

    return merged;
}

function mergeDetectionsSmart(detections1, detections2, iouThreshold = 0.2, proximity = 40) {
    const all = [...detections1, ...detections2];
    const merged = [];
    const used = new Array(all.length).fill(false);

    function getCenter(box) {
        const [x1, y1, x2, y2] = box;
        return [(x1 + x2) / 2, (y1 + y2) / 2];
    }

    function areClose(boxA, boxB) {
        const [cx1, cy1] = getCenter(boxA);
        const [cx2, cy2] = getCenter(boxB);
        const dx = cx1 - cx2;
        const dy = cy1 - cy2;
        return Math.sqrt(dx * dx + dy * dy) < proximity;
    }

    function iou(boxA, boxB) {
        const [ax1, ay1, ax2, ay2] = boxA;
        const [bx1, by1, bx2, by2] = boxB;

        const x1 = Math.max(ax1, bx1);
        const y1 = Math.max(ay1, by1);
        const x2 = Math.min(ax2, bx2);
        const y2 = Math.min(ay2, by2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const areaA = (ax2 - ax1) * (ay2 - ay1);
        const areaB = (bx2 - bx1) * (by2 - by1);
        const union = areaA + areaB - intersection;

        return intersection / union;
    }

    for (let i = 0; i < all.length; i++) {
        if (used[i]) continue;

        const cluster = [all[i]];
        used[i] = true;

        for (let j = i + 1; j < all.length; j++) {
            if (used[j]) continue;

            if (areClose(all[i].bbox, all[j].bbox) || iou(all[i].bbox, all[j].bbox) > iouThreshold) {
                cluster.push(all[j]);
                used[j] = true;
            }
        }

        // Об’єднуємо кластер в один блок
        const allBoxes = cluster.map(d => d.bbox);
        const allScores = cluster.map(d => d.score);
        const allClasses = [...new Set(cluster.map(d => d.class))];

        const minX = Math.min(...allBoxes.map(b => b[0]));
        const minY = Math.min(...allBoxes.map(b => b[1]));
        const maxX = Math.max(...allBoxes.map(b => b[2]));
        const maxY = Math.max(...allBoxes.map(b => b[3]));

        merged.push({
            bbox: [minX, minY, maxX, maxY],
            score: (allScores.reduce((a, b) => a + b, 0) / allScores.length).toFixed(2),
            class: allClasses.length === 1 ? allClasses[0] : allClasses.join(', '),
            mixed: allClasses.length > 1,
            count: cluster.length,
            modelScores: allScores.length > 1 ? {
                model1: allScores[0].toFixed(2),
                model2: allScores[1].toFixed(2)
            }: false
        });
    }

    return merged;
}



/**
 * Обчислює Intersection over Union для двох bbox
 */
function calcIOU(boxA, boxB) {
    const [x1, y1, x2, y2] = boxA;
    const [x1b, y1b, x2b, y2b] = boxB;

    const xI1 = Math.max(x1, x1b);
    const yI1 = Math.max(y1, y1b);
    const xI2 = Math.min(x2, x2b);
    const yI2 = Math.min(y2, y2b);

    const interArea = Math.max(0, xI2 - xI1) * Math.max(0, yI2 - yI1);
    const boxAArea = (x2 - x1) * (y2 - y1);
    const boxBArea = (x2b - x1b) * (y2b - y1b);
    const union = boxAArea + boxBArea - interArea;

    return interArea / union;
}

export function renderDetections(ctx, modelData, options = {mode: 'merge', debug: false}) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // console.log('yolo', yoloDetections[0].bbox);
    // console.log('mobilenet', mobilenetDetections[0].bbox);

    if (options?.mode === 'merge') {
        const merged = mergeDetectionsSmart(modelData.yolo, modelData.mobilenet);
        drawDetections(ctx, merged, modelData.yamnet, {
            color: 'cyan',
            debug: options.debug
        });
    } else {
        drawDetections(ctx, modelData.yolo, modelData.yamnet, {
            color: 'lime',
            debug: options.debug
        });
        drawDetections(ctx, modelData.mobilenet, modelData.yamnet, {
            color: 'orange',
            debug: options.debug
        });
    }
}
