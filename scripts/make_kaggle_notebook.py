"""Build a self-contained Kaggle saved-model inference experiment."""

import json
import runpy
from pathlib import Path

root = Path(__file__).resolve().parent.parent
runpy.run_path(str(root / "scripts/make_evaluation_notebook.py"))
base = json.loads((root / "notebooks/AeroVision_Colab.ipynb").read_text(encoding="utf-8"))
source = "".join(base["cells"][3]["source"])
source = source.replace("(PERSIST / 'source-", "(PROJECT / 'source-")
source = source[: source.index("def run(*args):")]
cells = []


def cell(kind, text):
    value = dict(cell_type=kind, id=f"kaggle-{len(cells):02}", metadata={}, source=text.splitlines(True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell(
    "markdown",
    """# AeroVision Lab — Kaggle resolution evaluation
Keep this notebook and its input Dataset **private**. Upload the completed
`colab-full-v1-60epochs-b16-results.zip` as a private Kaggle Dataset and attach it.
Enable Internet and an available GPU accelerator. This notebook does not train.
It uses one GPU even if T4 x2 is assigned, keeping inference comparable to Colab.
Download the final results ZIP before ending the session. No Google Drive or
GitHub token is needed; the source snapshot is embedded below.
""",
)
cell(
    "code",
    """import os, sys, pathlib, subprocess, json, csv, zipfile, io, base64, hashlib
assert pathlib.Path('/kaggle/working').exists(), 'Run this notebook on Kaggle.'
PROJECT = pathlib.Path('/kaggle/working/aerovision-lab')
PROJECT.mkdir(exist_ok=True)
os.chdir(PROJECT)
os.environ['YOLO_CONFIG_DIR'] = str(PROJECT / '.state/ultralytics')
(PROJECT / '.state/ultralytics').mkdir(parents=True, exist_ok=True)
""",
)
cell("code", source)
cell(
    "code",
    """import torch
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator, then rerun this cell.'
print('GPU:', torch.cuda.get_device_name(0))
EXPECTED = '9ea4e947efe74f75703ab2c5a32bf5a6af70124945a6ee9bd366c7cd83172840'
BEST = PROJECT / 'full-model-best.pt'
original = None
INPUT = pathlib.Path('/kaggle/input')
for archive_path in INPUT.rglob('*.zip'):
    if not zipfile.is_zipfile(archive_path):
        continue
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [n for n in archive.namelist() if n.endswith('/weights/best.pt')]
        for name in candidates:
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() == EXPECTED:
                BEST.write_bytes(payload)
                original = json.loads(archive.read('full-data-manifest.json'))
                break
    if original is not None:
        break
# Kaggle may unpack uploaded ZIPs automatically.
if original is None:
    for checkpoint in INPUT.rglob('best.pt'):
        payload = checkpoint.read_bytes()
        if hashlib.sha256(payload).hexdigest() == EXPECTED:
            BEST.write_bytes(payload)
            for manifest in INPUT.rglob('full-data-manifest.json'):
                candidate = json.loads(manifest.read_text())
                if len(candidate.get('splits', {}).get('val', [])) == 548:
                    original = candidate
                    break
            break
assert original is not None, 'Attach your full-data results ZIP as a private Dataset input.'
print('Verified full-data checkpoint.')
""",
)
cell(
    "code",
    """from aerovision.data import prepare_validation
DATA = prepare_validation(PROJECT / 'data', count=548)
current = json.loads((DATA.parent / 'manifest.json').read_text())
assert current['splits']['val'] == original['splits']['val'], 'Validation images differ from the original run.'
print('Verified all 548 validation images. Training data was not downloaded.')

def run_evaluation(label, *args):
    import re
    command = [sys.executable, '-u', '-m', 'aerovision.cli', *map(str, args)]
    log_path = PROJECT / (label + '.log')
    print(label + ': starting', flush=True)
    with log_path.open('w') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors='replace', bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                if 'images, cap=' in line:
                    progress = re.search(r'([0-9]+)/([0-9]+) images', line)
                    if progress and (int(progress[1]) % 100 == 0 or progress[1] == progress[2]):
                        print(line.strip(), flush=True)
            code = process.wait()
    if code:
        print('\\n'.join(log_path.read_text(errors='replace').splitlines()[-20:]))
        raise RuntimeError(label + ' failed; see the saved log.')
    print(label + ': complete', flush=True)
""",
)
cell(
    "code",
    """# Same checkpoint and scoring protocol in all four conditions.
conditions = [('whole-640', 'whole', 640), ('whole-960', 'whole', 960),
              ('whole-1280', 'whole', 1280), ('tiled-640', 'tiled', 640)]
rows = []
DESTINATIONS = []
for label, mode, size in conditions:
    destination = PROJECT / 'runs' / label
    run_evaluation(label, 'compare', '--data', DATA, '--weights', BEST,
        '--mode', mode, '--imgsz', size, '--tile', 640, '--overlap', 0.2,
        '--max-det', 1000, '--eval-max-det', 1000,
        '--output', destination, '--device', '0')
    result = json.loads((destination / 'comparison.json').read_text())
    metrics = result['modes'][mode]
    rows.append(dict(condition=label, AP=metrics['ap'], AP50=metrics['ap50'],
        AP_small=metrics['ap_small'], AR_small=metrics['recall_small'],
        p50_ms=metrics['latency_p50_ms'], p95_ms=metrics['latency_p95_ms']))
    DESTINATIONS.append(destination)
    print(f"{label}: AP={metrics['ap']:.2%}; small AP={metrics['ap_small']:.2%}; p50={metrics['latency_p50_ms']:.1f} ms")
with (PROJECT / 'summary.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
""",
)
cell(
    "code",
    """export = pathlib.Path('/kaggle/working/aerovision-resolution-results.zip')
with zipfile.ZipFile(export, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.write(PROJECT / 'summary.csv', 'summary.csv')
    archive.write(DATA.parent / 'manifest.json', 'validation-manifest.json')
    for destination in DESTINATIONS:
        archive.write(destination / 'comparison.json', destination.name + '/comparison.json')
    for log in PROJECT.glob('*.log'):
        archive.write(log, 'logs/' + log.name)
print('Download this file from the Kaggle output/files panel:', export)
print('Prediction JSON files remain under /kaggle/working/aerovision-lab/runs if needed.')
""",
)
cell(
    "markdown",
    """Send back `aerovision-resolution-results.zip`. No model training or final test-dev
evaluation was performed. Compare latency within this Kaggle run because GPU and
CPU conditions may differ from Colab. Higher inference resolution is an experiment,
not a guaranteed improvement for a checkpoint trained at 640 pixels.
""",
)
base["cells"] = cells
base["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
out = root / "notebooks/AeroVision_Kaggle.ipynb"
out.write_text(json.dumps(base, indent=2), encoding="utf-8")
print(out)
