const path = require("path");
const TF = require("@tensorflow/tfjs-node");
const { performance } = require("perf_hooks");
const { loadImage, Canvas, createImageData } = require("canvas");
const { writeFileSync } = require("fs");

(async() => {
    let now = performance.now();
    const handler = TF.io.fileSystem(`${__dirname}/model/model.json`);
    const model = await TF.loadGraphModel(handler);
    console.log(`Model loaded (time elapsed: ${(performance.now() - now).toFixed(1)} ms)`);

    const img = await loadImage(path.resolve(__dirname, "test.jpg"));
    const canvas = new Canvas(128, 128);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, 128, 128);

    now = performance.now();
    const out = await TF.tidy(() => model.predict(
            TF.expandDims(TF.browser.fromPixels(canvas, 3)
                                    .toFloat()
                                    .div(TF.scalar(255)), 0))
            .reshape([128, 128, 3]));

    ctx.clearRect(0, 0, 128, 128);
    const bitmap = await TF.browser.toPixels(out);
    const data = createImageData(bitmap, 128, 128);
    ctx.putImageData(data, 0, 0);

    writeFileSync("test-waifu.jpg", canvas.toBuffer("image/jpeg"));
    console.log(`Predict (time elapsed: ${(performance.now() - now).toFixed(1)} ms)`);
})();