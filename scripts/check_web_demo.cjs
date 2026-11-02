// Pure geometry checks: no browser, GPU, dataset or third-party dependency.
const assert = require('node:assert/strict');
require('../web-demo/core.js');
const { decode, iou } = globalThis.AeroVisionCore;
assert.equal(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1);
assert.equal(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0);
const data = new Float32Array(14 * 3);
for (let j = 0; j < 3; j++) for (let k = 0; k < 4; k++) data[k * 3 + j] = 16;
data[4 * 3] = .9; data[4 * 3 + 1] = .8; data[5 * 3 + 2] = .7;
const rows = decode(data, 3, 1.6, 0, 8, 20, 10, .55);
assert.equal(rows.length, 2); // same-class duplicate suppressed; different class kept
assert.deepEqual(rows.map(r => r.class_id), [0, 1]);
assert.deepEqual(rows[0].box_xyxy, [5, 0, 15, 10]);
assert.equal(decode(data, 3, 1.6, 0, 8, 20, 10, .95).length, 0);
console.log('Browser postprocessing checks passed.');
