/* Inference stays in a dedicated CPU worker; no WebGPU or remote image service. */
/* global importScripts, ort, AeroVisionCore */
importScripts('./vendor/1.22.0/ort.wasm.min.js', './core.js');
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = new URL('./vendor/1.22.0/', self.location.href).href;
ort.env.logLevel = 'error';
const MODEL_HASH = '663af645fd72248371fda473759aba2fcb1b69f84c3bfca7d36190b30af1918d';
let sessionPromise;
async function session() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      postMessage({ status: 'Loading model… first use downloads about 22 MB.' });
      const response = await fetch('./models/aerovision-' + MODEL_HASH.slice(0, 12) + '.onnx');
      if (!response.ok) throw new Error('Model download failed. Please try again.');
      const buffer = await response.arrayBuffer();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), v => v.toString(16).padStart(2, '0')).join('');
      if (digest !== MODEL_HASH) throw new Error('Model verification failed. Refresh and try again.');
      return ort.InferenceSession.create(buffer, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    })().catch(error => { sessionPromise = undefined; throw error; });
  }
  return sessionPromise;
}
self.onmessage = async ({ data: { pixels, width, height, confidence } }) => {
  try {
    const model = await session();
    postMessage({ status: 'Inspecting your aerial image…' });
    const start = performance.now();
    const ratio = Math.min(1280 / height, 1280 / width);
    const w = Math.round(width * ratio), h = Math.round(height * ratio);
    const padW = (1280 - w) % 32, padH = (1280 - h) % 32;
    const left = Math.round(padW / 2 - 0.1), top = Math.round(padH / 2 - 0.1);
    const inputW = w + padW, inputH = h + padH;
    const original = new OffscreenCanvas(width, height);
    original.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
    const canvas = new OffscreenCanvas(inputW, inputH);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.fillStyle = 'rgb(114,114,114)'; context.fillRect(0, 0, inputW, inputH);
    context.drawImage(original, 0, 0, width, height, left, top, w, h);
    const rgba = context.getImageData(0, 0, inputW, inputH).data;
    const plane = inputW * inputH, tensor = new Float32Array(plane * 3);
    for (let i = 0; i < plane; i++) {
      tensor[i] = rgba[i * 4] / 255;
      tensor[plane + i] = rgba[i * 4 + 1] / 255;
      tensor[plane * 2 + i] = rgba[i * 4 + 2] / 255;
    }
    const output = await model.run({ [model.inputNames[0]]: new ort.Tensor('float32', tensor, [1, 3, inputH, inputW]) });
    const result = output[model.outputNames[0]];
    const detections = AeroVisionCore.decode(result.data, result.dims[2], ratio, left, top, width, height, confidence);
    const latency = performance.now() - start;
    for (const value of Object.values(output)) value.dispose();
    postMessage({ result: { detections, latency_ms: latency, backend: 'ONNX Runtime Web / WASM CPU', confidence, input_shape: [1, 3, inputH, inputW], image_size: [width, height], model_version: '1.0.0', onnx_sha256: MODEL_HASH } });
  } catch (error) { postMessage({ error: error.message || 'Inference could not finish. Try a smaller aerial image.' }); }
};
