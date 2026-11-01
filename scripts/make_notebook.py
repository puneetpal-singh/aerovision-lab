"""Generate the portable notebook using only the Python standard library."""
import json
from pathlib import Path

cells = []


def cell(kind, source):
    c = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    cells.append(c)


cell("markdown", """# AeroVision Lab — small GPU pilot
Run in Colab or Kaggle with a GPU enabled. Uses the same code as the local project.
No notebook outputs or credentials are saved in this template. The private repo needs
a GitHub token with read access to this repo, entered through a hidden prompt.
Keep this notebook private. Hosted runtimes are temporary: download the final ZIP
and save `last.pt` periodically to persistent storage for interruption recovery.
""")
cell("code", """import os, sys, subprocess, pathlib, zipfile, io, getpass, json
import requests
token = getpass.getpass('GitHub token (read access to private AeroVision Lab): ')
response = requests.get('https://api.github.com/repos/puneetpal-singh/aerovision-lab/zipball/main',
    headers={'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json'}, timeout=120)
del token
response.raise_for_status()
archive = zipfile.ZipFile(io.BytesIO(response.content))
destination = pathlib.Path('aerovision-source').resolve()
destination.mkdir(exist_ok=True)
for item in archive.infolist():
    target = (destination / item.filename).resolve()
    if not target.is_relative_to(destination):
        raise ValueError('Unsafe archive member')
archive.extractall(destination)
project = next(p.parent for p in destination.glob('*/pyproject.toml'))
os.chdir(project)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
subprocess.run([sys.executable, '-m', 'aerovision.cli', 'doctor'], check=True)
""")
cell("markdown", """## Prepare a fixed pilot subset
All ten classes; random sample of 800 official training and 200 official validation
images, seed 42. Train and val membership is hashed. No test-dev tuning.
The original train/val ZIP downloads are larger than the selected subset.
""")
cell("code", """def run(*args):
    subprocess.run([sys.executable, '-m', 'aerovision.cli', *map(str, args)], check=True)
run('prepare', '--train', 800, '--val', 200)
data = pathlib.Path('data/mini-t800-v200-s42/dataset.yaml')
""")
cell("code", """# Short fit check. Run names are unique; existing runs are never overwritten.
run('train', '--data', data, '--epochs', 2, '--batch', 2, '--name', 'hosted-smoke')
""")
cell("code", """# Independent pilot training from pretrained weights, not from the smoke checkpoint.
run('train', '--data', data, '--epochs', 20, '--batch', 2, '--name', 'hosted-pilot')
""")
cell("code", """run('compare', '--data', data, '--weights', 'runs/hosted-pilot/weights/best.pt',
    '--output', 'runs/hosted-comparison')
print(pathlib.Path('runs/hosted-comparison/comparison.json').read_text())
""")
cell("code", """# Download immediately: hosted scratch storage is temporary.
import shutil
result_zip = shutil.make_archive(str(pathlib.Path.cwd() / 'aerovision-results'), 'zip', 'runs')
try:
    from google.colab import files
    files.download(result_zip)
except ImportError:
    from IPython.display import FileLink, display
    display(FileLink(result_zip))
""")
cell("markdown", """## Resume an interrupted run
Restore the entire saved run directory, including `weights/last.pt`, then call:
`run('train', '--data', data, '--name', 'hosted-pilot', '--resume',
'runs/hosted-pilot/weights/last.pt')`.
The checkpoint retains the original training schedule. Completed runs cannot be
extended through resume; create a new experiment instead.

These are pilot COCO-style metrics with an ignored-region approximation, not the
official VisDrone evaluation. Laptop and hosted timings are separate measurements.
""")
out = Path('notebooks/aerovision_pilot.ipynb')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5}, indent=2), encoding='utf-8')
print(out)
