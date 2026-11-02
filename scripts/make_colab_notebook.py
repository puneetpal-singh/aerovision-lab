"""Bundle our source into a Colab notebook; private GitHub authentication is unnecessary."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
buffer = io.BytesIO()
files = [ROOT / "pyproject.toml", ROOT / "README.md", *sorted((ROOT / "src/aerovision").glob("*.py"))]
with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, path.read_bytes())
payload = buffer.getvalue()
sha = hashlib.sha256(payload).hexdigest()
encoded = base64.b64encode(payload).decode()
cells = []


def cell(kind, source):
    item = dict(cell_type=kind, id=f"cell-{len(cells):02}", metadata={}, source=source.splitlines(True))
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)


cell(
    "markdown",
    """# AeroVision Lab — free Colab GPU pilot
1. Choose **Runtime → Change runtime type → GPU** (a free GPU option, if available).
2. Run the cells in order. Approve Google Drive access when prompted.
3. Keep this notebook private: it includes a snapshot of your private project's source.

Training runs on Google's GPU. Do not connect to a local runtime.
This notebook contains the complete pipeline, so no GitHub token is needed.
Checkpoints save to `My Drive/AeroVisionLab/runs/` every epoch. Colab resources
are not guaranteed; if a session ends, rerun the notebook to prepare the same data
and resume from the saved checkpoint. A failure during the current epoch can lose
that epoch. Select another saved epoch checkpoint if `last.pt` is damaged.

First milestone: YOLO11n, 800 training / 200 validation images, 640-pixel input,
batch 2, 20 epochs. Batch stays small to preserve the planned pilot setup.
""",
)
cell(
    "code",
    """import os, sys, pathlib, subprocess, json, csv, zipfile, io, base64, hashlib
from google.colab import drive
import torch
assert torch.cuda.is_available(), 'Select a hosted GPU runtime, then rerun this cell.'
print('Hosted GPU:', torch.cuda.get_device_name(0))
drive.mount('/content/drive')
PERSIST = pathlib.Path('/content/drive/MyDrive/AeroVisionLab')
PERSIST.mkdir(parents=True, exist_ok=True)
PROJECT = pathlib.Path('/content/aerovision-lab')
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT / '.state/ultralytics')
""",
)
cell(
    "markdown",
    f"Source snapshot SHA-256: `{sha}`. The next cell installs this snapshot into the hosted runtime.",
)
cell(
    "code",
    f"""bundle = base64.b64decode('{encoded}')
assert hashlib.sha256(bundle).hexdigest() == '{sha}'
with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
    for member in archive.namelist():
        assert (PROJECT / member).resolve().is_relative_to(PROJECT)
    archive.extractall(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)
# Save the source that produced the run alongside its outputs.
(PERSIST / 'source-{sha[:12]}.zip').write_bytes(bundle)
def run(*args):
    subprocess.run([sys.executable, '-u', '-m', 'aerovision.cli', *map(str, args)], check=True)
run('doctor')
""",
)
cell(
    "code",
    """# Dataset stays on Colab scratch disk for fast image loading.
run('prepare', '--train', 800, '--val', 200)
DATA = PROJECT / 'data/mini-t800-v200-s42/dataset.yaml'
subset = DATA.parent
for filename in ['manifest.json', 'dataset.yaml']:
    (PERSIST / filename).write_bytes((subset / filename).read_bytes())
# Only small run artifacts/checkpoints go to Drive, not thousands of dataset images.
(PERSIST / 'runs').mkdir(exist_ok=True)
if not pathlib.Path('runs').exists():
    pathlib.Path('runs').symlink_to(PERSIST / 'runs', target_is_directory=True)
assert pathlib.Path('runs').resolve() == (PERSIST / 'runs').resolve()
""",
)
cell(
    "code",
    """# Keep the same name to resume. Choose a NEW name for a separate experiment.
RUN_NAME = 'colab-pilot-v1'
EPOCHS = 20
run_dir = pathlib.Path('runs') / RUN_NAME
last = run_dir / 'weights/last.pt'
best = run_dir / 'weights/best.pt'
completed_epochs = 0
if (run_dir / 'results.csv').exists():
    with (run_dir / 'results.csv').open() as f:
        records = list(csv.DictReader(f))
    if records:
        completed_epochs = int(float(records[-1]['epoch']))
if completed_epochs >= EPOCHS and best.exists():
    print('Training already completed; proceeding to evaluation.')
elif last.exists():
    print('Resuming saved checkpoint; last recorded epoch:', completed_epochs)
    run('train', '--data', DATA, '--name', RUN_NAME, '--resume', last, '--device', '0')
else:
    if run_dir.exists():
        raise RuntimeError('An incomplete run folder has no checkpoint. Choose a new RUN_NAME.')
    run('train', '--data', DATA, '--epochs', EPOCHS, '--batch', 2,
        '--workers', 2, '--name', RUN_NAME, '--device', '0')
""",
)
cell(
    "code",
    """# Both modes use the same checkpoint, images and prediction settings.
COMPARISON = pathlib.Path('runs') / (RUN_NAME + '-comparison')
run('compare', '--data', DATA, '--weights', best, '--output', COMPARISON, '--device', '0')
report = json.loads((COMPARISON / 'comparison.json').read_text())
for mode, metrics in report['modes'].items():
    print(mode, json.dumps(metrics, indent=2))
""",
)
cell(
    "code",
    """# Compact result bundle, already stored in Drive; download a copy if desired.
export = PERSIST / (RUN_NAME + '-results.zip')
with zipfile.ZipFile(export, 'w', zipfile.ZIP_DEFLATED) as archive:
    for directory in [run_dir, COMPARISON]:
        for path in directory.rglob('*'):
            if path.is_file() and (path.suffix in {'.json', '.csv', '.yaml', '.png'}
                                  or path.name in {'best.pt', 'last.pt'}):
                # Exclude dataset preview images; results.png is a learning curve.
                if path.name.startswith(('train_batch', 'val_batch')) or path.name == 'labels.jpg':
                    continue
                archive.write(path, path.relative_to('runs'))
print('Saved:', export)
from google.colab import files
files.download(str(export))
""",
)
cell(
    "markdown",
    """## Interpretation
These are exploratory COCO-style validation metrics, not official VisDrone scores.
Ignored regions use an explicit crowd approximation. The validation set is fixed;
test-dev has not been used. Small-object recall gains must be weighed against latency
and false positives. Record the GPU actually assigned; do not compare its latency
directly with laptop timings.

Free GPU type and session duration vary: https://research.google.com/colaboratory/faq.html
After completing the experiment, disconnect the runtime to release its resources.
""",
)
notebook = dict(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "colab": {"name": "AeroVision_Colab.ipynb", "provenance": []},
        "accelerator": "GPU",
    },
    nbformat=4,
    nbformat_minor=5,
)
out = ROOT / "notebooks/AeroVision_Colab.ipynb"
out.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(out)
