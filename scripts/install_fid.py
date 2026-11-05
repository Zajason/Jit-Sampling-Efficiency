"""Explicit optional installation of the pinned official FID fork."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
lock=json.loads((root/'configs/fid-upstream.json').read_text())
subprocess.run([sys.executable,'-m','pip','install',f'git+{lock["repository"]}.git@{lock["revision"]}'],check=True)
