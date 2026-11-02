"""Generate a readable, self-contained cell for the existing Kaggle session."""

from pathlib import Path

root = Path(__file__).resolve().parent.parent
matching = (root / "src/aerovision/diagnostics.py").read_text(encoding="utf-8").split("def review(")[0]
confidence = (root / "src/aerovision/confidence.py").read_text(encoding="utf-8")
confidence = confidence.replace("from .diagnostics import match_errors", "")
body = """
# Run in the original Kaggle notebook: the downloaded summary ZIP has no predictions.
ROOT = Path('/kaggle/working/aerovision-lab')
SUBSET = ROOT / 'data/validation-v548-s42'
COMPARISON = ROOT / 'runs/whole-1280'
PREDICTIONS = COMPARISON / 'whole.predictions.json'
assert PREDICTIONS.exists(), (
    'Saved predictions are missing. Use the original Kaggle session. '
    'The resolution-results ZIP contains summaries, not predictions.'
)
comparison = json.loads((COMPARISON / 'comparison.json').read_text())
assert comparison['weights_sha256'] == '9ea4e947efe74f75703ab2c5a32bf5a6af70124945a6ee9bd366c7cd83172840'
assert comparison['settings']['imgsz'] == 1280
truth = json.loads((SUBSET / 'val.coco.json').read_text())
predictions = json.loads(PREDICTIONS.read_text())
ignored = json.loads((SUBSET / 'val.ignored.json').read_text())
print('Reviewing saved predictions on CPU; no model is loaded.', flush=True)
rows = sweep(truth, predictions, ignored)
summary = [{k: v for k, v in row.items() if k != 'per_class'} for row in rows]
print('Confidence | Precision | Recall | Micro F1 | Macro F1')
for row in summary:
    print(f"{row['confidence']:10.2f} | {row['precision']:9.1%} | {row['recall']:6.1%} | "
          f"{row['micro_f1']:8.1%} | {row['macro_f1']:8.1%}")
report = dict(protocol='Validation-only greedy same-class IoU>=0.5; crowd-overlap approximation; not COCO AP',
    image_count=len(truth['images']), weights_sha256=comparison['weights_sha256'],
    settings=comparison['settings'], rows=rows,
    micro_f1_candidate=max(summary, key=lambda r: r['micro_f1'])['confidence'],
    macro_f1_candidate=max(summary, key=lambda r: r['macro_f1'])['confidence'])
OUT = Path('/kaggle/working/aerovision-confidence-results.zip')
import zipfile
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('confidence.json', json.dumps(report, indent=2))
print('Download from the Kaggle files/output panel:', OUT)
"""
output = root / "notebooks/AeroVision_Confidence_Cell.py"
output.write_text(matching + "\n" + confidence + "\n" + body, encoding="utf-8")
print(output)
