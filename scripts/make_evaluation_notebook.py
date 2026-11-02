"""Generate an evaluation-only notebook from the same embedded source snapshot."""

import json
import runpy
from pathlib import Path

root = Path(__file__).resolve().parent.parent
runpy.run_path(str(root / "scripts/make_colab_notebook.py"))
base = json.loads((root / "notebooks/AeroVision_Colab.ipynb").read_text(encoding="utf-8"))


def cell(kind, source):
    c = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


cells = [
    cell(
        "markdown",
        """# AeroVision Lab — saved-model evaluation
This notebook does **not train**. It evaluates the saved `colab-pilot-v1` model
at prediction limits 300, 600 and 1000 using the same 200 validation images.
Select a hosted GPU runtime, mount the same Google Drive, then run all cells.
Keep private: source code is embedded. No GitHub token is needed.
""",
    ),
    *base["cells"][1:4],
]
cells.append(
    cell(
        "code",
        """# Verify the model before preparing data or doing inference.
from aerovision.data import digest
BEST = PERSIST / 'runs/colab-pilot-v1/weights/best.pt'
assert BEST.exists(), f'Checkpoint not found: {BEST}'
EXPECTED_MODEL = '3ecd854c6b6607edabc455128057b3a77a2816898a84ea405c178fd08e307194'
assert digest(BEST) == EXPECTED_MODEL, 'This is not the checkpoint from the pilot reviewed in chat.'
saved_manifest = PERSIST / 'manifest.json'
assert saved_manifest.exists(), 'Original subset manifest is missing from Drive.'
original = json.loads(saved_manifest.read_text())
run('prepare', '--train', 800, '--val', 200)
DATA = PROJECT / 'data/mini-t800-v200-s42/dataset.yaml'
subset = DATA.parent
current = json.loads((subset / 'manifest.json').read_text())
assert original['splits'] == current['splits'], 'Image membership or content differs from the pilot.'
(PERSIST / 'runs').mkdir(exist_ok=True)
if not pathlib.Path('runs').exists():
    pathlib.Path('runs').symlink_to(PERSIST / 'runs', target_is_directory=True)
assert pathlib.Path('runs').resolve() == (PERSIST / 'runs').resolve()
""",
    )
)
cells.append(
    cell(
        "markdown",
        """## Controlled prediction-limit sweep
The COCO evaluator limit stays at 1000 per category for every run. The prediction
limit varies jointly at the detector output (per tile for tiled mode) and after
global merging. This does not isolate those two stages from each other.
The 300 control uses the original prediction limits, but a different evaluator
limit from the original report. Compare within this new sweep.
""",
    )
)
cells.append(
    cell(
        "code",
        """import pandas as pd
rows = []
DESTINATIONS = []
for cap in [300, 600, 1000]:
    destination = pathlib.Path('runs') / f'colab-pilot-v1-cap-{cap}'
    print(f'\\nEvaluating prediction cap {cap}', flush=True)
    run('compare', '--data', DATA, '--weights', BEST, '--max-det', cap,
        '--eval-max-det', 1000, '--output', destination, '--device', '0')
    report = json.loads((destination / 'comparison.json').read_text())
    for mode, values in report['modes'].items():
        rows.append(dict(cap=cap, mode=mode, AP=values['ap'], AP50=values['ap50'],
            AP_small=values['ap_small'], AR_small=values['recall_small'],
            p50_ms=values['latency_p50_ms'], p95_ms=values['latency_p95_ms'],
            capped_images=values['images_at_prediction_cap']))
    DESTINATIONS.append(destination)
table = pd.DataFrame(rows)
display(table)
SWEEP = PERSIST / 'runs/colab-pilot-v1-cap-sweep'
SWEEP.mkdir(exist_ok=True)
table.to_csv(SWEEP / 'summary.csv', index=False)
""",
    )
)
cells.append(
    cell(
        "code",
        """# Fixed confidence diagnostic, separate from COCO AP.
from aerovision.diagnostics import review
from IPython.display import display, Image as DisplayImage
for destination in DESTINATIONS:
    errors = review(subset, destination, confidence=0.25, examples=6)
    print(destination.name, {m: v['totals'] for m, v in errors['modes'].items()})
# Same first six validation images across modes; these are not cherry-picked successes.
for path in sorted((DESTINATIONS[-1] / 'error-review').glob('*.jpg'))[:4]:
    display(DisplayImage(filename=str(path), width=1000))
""",
    )
)
cells.append(
    cell(
        "code",
        """# Save metrics and visual examples, excluding bulky prediction JSON and model duplicates.
export = PERSIST / 'colab-pilot-v1-evaluation-v2.zip'
with zipfile.ZipFile(export, 'w', zipfile.ZIP_DEFLATED) as archive:
    for directory in [SWEEP, *DESTINATIONS]:
        for path in directory.rglob('*'):
            if path.is_file() and not path.name.endswith('.predictions.json'):
                archive.write(path, path.resolve().relative_to((PERSIST / 'runs').resolve()))
print('Saved to Drive:', export)
from google.colab import files
files.download(str(export))
""",
    )
)
cells.append(
    cell(
        "markdown",
        """Send back `colab-pilot-v1-evaluation-v2.zip`. This is a validation diagnostic,
not a final held-out test or evidence that a higher cap is better. No local GPU is used.
""",
    )
)
for i, c in enumerate(cells):
    c["id"] = f"evaluation-{i:02}"
base["cells"] = cells
base["metadata"]["colab"]["name"] = "AeroVision_Evaluation.ipynb"
out = root / "notebooks/AeroVision_Evaluation.ipynb"
out.write_text(json.dumps(base, indent=2), encoding="utf-8")
print(out)
