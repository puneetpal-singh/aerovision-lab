(() => {
  const $ = id => document.getElementById(id);
  const file = $('image-file'), detect = $('detect'), slider = $('confidence');
  const canvas = $('preview'), ctx = canvas.getContext('2d', { willReadFrequently: true });
  const colors = ['#55d5ce', '#eabf68', '#a6b4fa', '#6ed99c', '#f18d91', '#82bcec', '#d6a0de', '#b4d978', '#f8aa71', '#7ad8df'];
  let image, result, worker, busy = false, fileVersion = 0;
  function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
  function setBusy(value) { busy = value; detect.disabled = busy || !image; file.disabled = busy; slider.disabled = busy; }
  function render() {
    if (!image) return;
    ctx.drawImage(image, 0, 0);
    if (!result) return;
    const size = Math.max(12, Math.min(24, canvas.width / 90));
    ctx.font = `600 ${size}px Arial`; ctx.lineWidth = Math.max(2, canvas.width / 700);
    for (const row of [...result.detections].reverse()) {
      const [x1, y1, x2, y2] = row.box_xyxy, color = colors[row.class_id];
      ctx.strokeStyle = color; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      if ($('labels').checked) {
        const text = `${row.label} ${row.confidence.toFixed(2)}`, y = Math.max(size + 4, y1);
        const x = Math.min(x1, Math.max(0, canvas.width - ctx.measureText(text).width - 6));
        ctx.fillStyle = '#10201b'; ctx.fillRect(x, y - size - 3, ctx.measureText(text).width + 6, size + 5);
        ctx.fillStyle = color; ctx.fillText(text, x + 3, y - 2);
      }
    }
  }
  async function choose(selected) {
    if (busy || !selected) return;
    const version = ++fileVersion;
    result = undefined; image?.close(); image = undefined; detect.disabled = true;
    $('result').hidden = true; canvas.hidden = true; $('empty').hidden = false;
    $('filename').textContent = ''; $('image-size').textContent = 'AERIAL INPUT';
    if (!['image/jpeg', 'image/png'].includes(selected.type)) { status('Choose an aerial JPG or PNG image.', true); return; }
    if (selected.size > 10 * 1024 * 1024) { status('Choose a file smaller than 10 MB.', true); return; }
    try {
      const decoded = await createImageBitmap(selected, { imageOrientation: 'from-image' });
      if (version !== fileVersion) { decoded.close(); return; }
      if (decoded.width * decoded.height > 20_000_000) { decoded.close(); throw new Error('Choose an image below 20 megapixels.'); }
      image = decoded; canvas.width = image.width; canvas.height = image.height;
      canvas.hidden = false; $('empty').hidden = true;
      $('filename').textContent = selected.name;
      $('image-size').textContent = `${image.width} × ${image.height}`;
      render(); detect.disabled = false; status('Image ready. Select Detect objects.');
    } catch (error) { status(error.message.startsWith('Choose') ? error.message : 'This file could not be read. Try another JPG or PNG.', true); }
  }
  file.addEventListener('change', () => choose(file.files[0]));
  const drop = $('drop-zone');
  for (const event of ['dragenter', 'dragover']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.add('drag'); });
  for (const event of ['dragleave', 'drop']) drop.addEventListener(event, e => { e.preventDefault(); drop.classList.remove('drag'); });
  drop.addEventListener('drop', e => choose(e.dataTransfer.files[0]));
  slider.addEventListener('input', () => {
    $('confidence-value').value = Number(slider.value).toFixed(2);
    result = undefined; $('result').hidden = true; render();
    if (image) status('Threshold changed. Select Detect objects to update.');
  });
  $('labels').addEventListener('change', render);
  detect.addEventListener('click', () => {
    if (!image || busy) return;
    setBusy(true); status('Preparing detection…');
    if (!worker) {
      worker = new Worker(new URL('./worker.js', location.href));
      worker.onerror = () => {
        status('The browser worker could not start. Refresh and try again in a current browser.', true);
        worker.terminate(); worker = undefined; setBusy(false);
      };
      worker.onmessage = ({ data }) => {
        if (data.status) { status(data.status); return; }
        setBusy(false);
        if (data.error) { status(data.error, true); return; }
        result = data.result;
        $('result').hidden = false; $('count').textContent = result.detections.length;
        const counts = new Map();
        for (const row of result.detections) counts.set(row.label, (counts.get(row.label) || 0) + 1);
        $('classes').textContent = counts.size;
        $('latency').textContent = `${(result.latency_ms / 1000).toFixed(2)} s`;
        $('class-counts').replaceChildren(...[...counts].sort((a, b) => b[1] - a[1]).map(([label, count]) => {
          const chip = document.createElement('span'); chip.textContent = `${label} · ${count}`; return chip;
        }));
        render(); status(result.detections.length ? 'Done. Explore your detections or download the results.' : 'No objects above this threshold. Try lowering confidence.');
      };
    }
    const original = new OffscreenCanvas(image.width, image.height), context = original.getContext('2d', { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, image.width, image.height).data;
    worker.postMessage({ pixels: pixels.buffer, width: image.width, height: image.height, confidence: Number(slider.value) }, [pixels.buffer]);
  });
  function download(blob, name) {
    const link = document.createElement('a'), url = URL.createObjectURL(blob);
    link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  $('download-image').addEventListener('click', () => canvas.toBlob(blob => download(blob, 'aerovision-detections.png'), 'image/png'));
  $('download-json').addEventListener('click', () => download(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }), 'aerovision-predictions.json'));
})();
