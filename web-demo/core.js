/* Pure postprocessing shared by the browser worker and Node checks. */
(() => {
  const labels = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motorbike'];
  function iou(a, b) {
    const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const area = box => Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
    return w * h / Math.max(1e-12, area(a) + area(b) - w * h);
  }
  function decode(data, n, ratio, left, top, width, height, confidence) {
    const candidates = [];
    for (let j = 0; j < n; j++) {
      let score = -Infinity, cls = 0;
      for (let c = 0; c < 10; c++) {
        const value = data[(c + 4) * n + j];
        if (value > score) { score = value; cls = c; }
      }
      if (!(score > confidence)) continue;
      const x = data[j], y = data[n + j], w = data[2 * n + j], h = data[3 * n + j];
      candidates.push({ class_id: cls, label: labels[cls], confidence: score, box_xyxy: [x - w / 2, y - h / 2, x + w / 2, y + h / 2] });
    }
    candidates.sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    for (const row of candidates.slice(0, 30000)) {
      if (kept.some(other => other.class_id === row.class_id && iou(other.box_xyxy, row.box_xyxy) > 0.7)) continue;
      kept.push(row);
      if (kept.length === 1000) break;
    }
    for (const row of kept) {
      row.box_xyxy = row.box_xyxy.map((v, k) => Math.min(k % 2 ? height : width, Math.max(0, (v - (k % 2 ? top : left)) / ratio)));
    }
    return kept;
  }
  globalThis.AeroVisionCore = { labels, iou, decode };
})();
