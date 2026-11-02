"""Generate the frozen PyTorch held-out test-dev experiment."""

import json
import runpy
from pathlib import Path

root = Path(__file__).resolve().parent.parent
runpy.run_path(str(root / "scripts/make_kaggle_notebook.py"))
base = json.loads((root / "notebooks/AeroVision_Kaggle.ipynb").read_text(encoding="utf-8"))
cells = []


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


cells = [
    cell(
        "markdown",
        """# AeroVision Lab — frozen held-out test-dev evaluation
Import into a **private** Kaggle notebook. Attach the same private full-training
results Dataset; enable Internet and T4 GPU. Enable Files-only persistence.
No training, threshold search or model selection occurs here. This runs the
saved PyTorch checkpoint, not the ONNX model. The ONNX checks were a separate
small CPU parity test.

Frozen candidate: whole-image 1280, confidence 0.30 for operating-point metrics,
NMS IoU 0.7, prediction/evaluation limit 1000. AP retains predictions down to
0.001 for curve integration. Whole-image 640 is a predeclared baseline.
Scoring remains COCO-style with a crowd approximation for ignored regions;
these are **not official VisDrone challenge scores**. Keep test results separate
from validation and do not tune the configuration after seeing them.
""",
    ),
    *base["cells"][1:4],
]
helper = "".join(base["cells"][4]["source"])
helper = helper[helper.index("def run_evaluation(") :]
cell_source = """from aerovision.data import prepare_validation, digest
from datetime import datetime, timezone
TEST_DATA = prepare_validation(PROJECT / 'data', count=1610, split='test-dev')
TEST_ROOT = TEST_DATA.parent
test_manifest = json.loads((TEST_ROOT / 'manifest.json').read_text())
assert len(test_manifest['splits']['test-dev']) == 1610
seen = {r['sha256'] for split in original['splits'].values() for r in split}
assert not any(r['sha256'] in seen for r in test_manifest['splits']['test-dev']), 'Identical train/validation image found in test-dev.'
FINAL = PROJECT / 'runs/final-testdev'
FINAL.mkdir(parents=True, exist_ok=True)
plan = dict(weights_sha256=EXPECTED, candidate='whole-1280', baseline='whole-640',
    confidence=0.30, iou=0.7, max_det=1000, eval_max_det=1000,
    ap_confidence=0.001, split='test-dev', count=1610)
plan_path = FINAL / 'frozen-plan.json'
if plan_path.exists():
    assert json.loads(plan_path.read_text())['settings'] == plan
else:
    plan_path.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), settings=plan), indent=2))
print('Verified 1610 held-out images, with no exact image hashes shared with train/validation.')
"""
cells.append(cell("code", cell_source + "\n" + helper))
cells.append(
    cell(
        "code",
        """from aerovision.confidence import sweep
truth = json.loads((TEST_ROOT / 'test-dev.coco.json').read_text())
ignored = json.loads((TEST_ROOT / 'test-dev.ignored.json').read_text())
summary = []
DESTINATIONS = []
for label, size in [('whole-640', 640), ('whole-1280', 1280)]:
    destination = FINAL / label
    report_path = destination / 'comparison.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['weights_sha256'] == EXPECTED
        assert report['manifest_sha256'] == digest(TEST_ROOT / 'manifest.json')
        assert report['settings']['split'] == 'test-dev' and report['settings']['imgsz'] == size
        assert report['settings']['max_det'] == 1000 and report['settings']['eval_max_det'] == 1000
        assert len(report['image_ids']) == 1610
        print(label + ': reusing saved evaluation')
    else:
        run_evaluation(label, 'compare', '--data', TEST_DATA, '--weights', BEST,
            '--mode', 'whole', '--split', 'test-dev', '--imgsz', size,
            '--max-det', 1000, '--eval-max-det', 1000, '--output', destination, '--device', '0')
        report = json.loads(report_path.read_text())
    predictions = json.loads((destination / 'whole.predictions.json').read_text())
    # One predetermined confidence, not a threshold sweep on test-dev.
    operating = sweep(truth, predictions, ignored, thresholds=[0.30])[0]
    (destination / 'operating-point.json').write_text(json.dumps(operating, indent=2))
    metrics = report['modes']['whole']
    summary.append(dict(condition=label, AP=metrics['ap'], AP50=metrics['ap50'],
        AP_small=metrics['ap_small'], AR_small=metrics['recall_small'],
        p50_ms=metrics['latency_p50_ms'], precision_at_030=operating['precision'],
        recall_at_030=operating['recall'], micro_F1_at_030=operating['micro_f1'],
        macro_F1_at_030=operating['macro_f1']))
    DESTINATIONS.append(destination)
    print(label, summary[-1], flush=True)
with (FINAL / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
""",
    )
)
cells.append(
    cell(
        "code",
        """export = pathlib.Path('/kaggle/working/aerovision-testdev-results.zip')
with zipfile.ZipFile(export, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in FINAL.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(FINAL))
    for name in ['manifest.json', 'test-dev.coco.json', 'test-dev.ignored.json']:
        archive.write(TEST_ROOT / name, 'inputs/' + name)
    for log in PROJECT.glob('whole-*.log'):
        archive.write(log, 'logs/' + log.name)
print('Download from Kaggle files/output panel:', export)
print('The ZIP includes predictions and annotations so review can continue on CPU.')
""",
    )
)
for i, item in enumerate(cells):
    item["id"] = f"testdev-{i:02}"
base["cells"] = cells
out = root / "notebooks/AeroVision_TestDev.ipynb"
out.write_text(json.dumps(base, indent=2), encoding="utf-8")
print(out)
