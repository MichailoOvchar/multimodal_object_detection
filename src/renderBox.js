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
            average = average > 1 || 1;

            debugText += ` | s: +0.15`;
        }
        if(classes[label]?.penalize?.includes(audioClass)??false){
            average -= 0.15;

            debugText += ` | s: -0.15`;

            if(average < 20) return;
        }

        let text = `${label} (${(average * 100).toFixed(1)}%)`;
        const textWidth = ctx.measureText(text).width;
        const textHeight = 16;

        text = text.replaceAll('toothbrush', 'pen');

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


function mergeDetections(model1Detections, model2Detections, iouThreshold = 0.2) {
    const merged = [];
    const used1 = new Set();
    const used2 = new Set();

    function iou(boxA, boxB) {
        const [xA1, yA1, xA2, yA2] = boxA;
        const [xB1, yB1, xB2, yB2] = boxB;

        const interX1 = Math.max(xA1, xB1);
        const interY1 = Math.max(yA1, yB1);
        const interX2 = Math.min(xA2, xB2);
        const interY2 = Math.min(yA2, yB2);

        const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
        const boxAArea = (xA2 - xA1) * (yA2 - yA1);
        const boxBArea = (xB2 - xB1) * (yB2 - yB1);

        return interArea / (boxAArea + boxBArea - interArea);
    }

    const tryMerge = (preferSameClass = true) => {
        for (let i = 0; i < model1Detections.length; i++) {
            if (used1.has(i)) continue;

            const det1 = model1Detections[i];
            let bestMatch = null;
            let bestIndex = -1;
            let bestIoU = 0;

            for (let j = 0; j < model2Detections.length; j++) {
                if (used2.has(j)) continue;

                const det2 = model2Detections[j];
                const sameClass = det1.class === det2.class;
                const currentIoU = iou(det1.bbox, det2.bbox);

                const isValid = currentIoU >= iouThreshold &&
                    (!preferSameClass || sameClass);

                if (isValid && currentIoU > bestIoU) {
                    bestMatch = det2;
                    bestIndex = j;
                    bestIoU = currentIoU;
                }
            }

            if (bestMatch) {
                used1.add(i);
                used2.add(bestIndex);
                merged.push({
                    bbox: [
                        (det1.bbox[0] + bestMatch.bbox[0]) / 2,
                        (det1.bbox[1] + bestMatch.bbox[1]) / 2,
                        (det1.bbox[2] + bestMatch.bbox[2]) / 2,
                        (det1.bbox[3] + bestMatch.bbox[3]) / 2
                    ],
                    score: (det1.score + bestMatch.score) / 2,
                    class: preferSameClass ? det1.class : `${det1.class}+${bestMatch.class}`,
                    modelScores: {
                        model1: det1.score,
                        model2: bestMatch.score
                    }
                });
            }
        }
    };

    // Крок 1: об'єднуємо тільки ті, що з однаковими класами
    tryMerge(true);

    // Крок 2: об'єднуємо всі інші, що поруч (навіть якщо класи різні)
    tryMerge(false);

    // Додаємо не використані блоки
    model1Detections.forEach((det, i) => {
        if (!used1.has(i)) merged.push(det);
    });

    model2Detections.forEach((det, i) => {
        if (!used2.has(i)) merged.push(det);
    });

    return merged;
}

function mergeDetections2(detections1, detections2, iouThreshold = 0.2, proximity = 40) {
    const merged = [];
    const used1 = new Array(detections1.length).fill(false);
    const used2 = new Array(detections2.length).fill(false);

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

    for (let i = 0; i < detections1.length; i++) {
        if (used1[i]) continue;

        let bestMatchIndex = -1;
        let bestScore = 0;

        for (let j = 0; j < detections2.length; j++) {
            if (used2[j]) continue;

            const sameClass = detections1[i].class === detections2[j].class;
            const closeEnough = areClose(detections1[i].bbox, detections2[j].bbox);
            const iouVal = iou(detections1[i].bbox, detections2[j].bbox);

            let matchScore = 0;
            if (sameClass && (closeEnough || iouVal > iouThreshold)) {
                matchScore = 2; // ідеальний матч
            } else if (closeEnough || iouVal > iouThreshold) {
                matchScore = 1; // допустимий матч
            }

            if (matchScore > bestScore) {
                bestScore = matchScore;
                bestMatchIndex = j;
            }
        }

        if (bestMatchIndex !== -1) {
            const d1 = detections1[i];
            const d2 = detections2[bestMatchIndex];
            used1[i] = true;
            used2[bestMatchIndex] = true;

            const allBoxes = [d1.bbox, d2.bbox];
            const allScores = [d1.score, d2.score];
            const allClasses = [d1.class, d2.class];

            const minX = Math.min(...allBoxes.map(b => b[0]));
            const minY = Math.min(...allBoxes.map(b => b[1]));
            const maxX = Math.max(...allBoxes.map(b => b[2]));
            const maxY = Math.max(...allBoxes.map(b => b[3]));

            merged.push({
                bbox: [minX, minY, maxX, maxY],
                score: ((allScores[0] + allScores[1]) / 2).toFixed(2),
                class: allClasses[0] === allClasses[1] ? allClasses[0] : allClasses.join(', '),
                mixed: allClasses[0] !== allClasses[1],
                count: 2,
                modelScores: {
                    model1: allScores[0].toFixed(2),
                    model2: allScores[1].toFixed(2)
                }
            });
        }
    }

    detections1.forEach((det, i) => {
        if (!used1[i]) merged.push({ ...det, count: 1 });
    });
    detections2.forEach((det, i) => {
        if (!used2[i]) merged.push({ ...det, count: 1 });
    });

    return merged;
}


export function renderDetections(ctx, modelData, options = {mode: 'merge', debug: false}) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // console.log('yolo', yoloDetections[0].bbox);
    // console.log('mobilenet', mobilenetDetections[0].bbox);

    if (options?.mode === 'merge') {
        const merged = mergeDetections2(modelData.yolo, modelData.mobilenet);
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
