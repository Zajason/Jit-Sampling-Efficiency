from argparse import Namespace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[2]

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def verify_source(path):
    path = Path(path).resolve()
    lock = json.loads((ROOT / 'configs/upstream.json').read_text())
    for name, digest in lock['files'].items():
        if sha256(path / name) != digest:
            raise RuntimeError(f'Upstream source differs from pinned revision: {name}')
    return lock

def denoiser_class(path):
    sys.path.insert(0, str(Path(path).resolve()))
    return importlib.import_module('denoiser').Denoiser

def load_model(path, checkpoint, device, cfg=3.0):
    """Mirror main_jit.py and engine_jit.py: model buffers + first EMA weights."""
    verify_source(path)
    cls = denoiser_class(path)
    args = SimpleNamespace(model='JiT-B/16', img_size=256, class_num=1000,
        attn_dropout=0., proj_dropout=0., label_drop_prob=0.1,
        P_mean=-0.8, P_std=0.8, t_eps=0.05, noise_scale=1.,
        ema_decay1=0.9999, ema_decay2=0.9996,
        sampling_method='heun', num_sampling_steps=50,
        cfg=cfg, interval_min=0.1, interval_max=1.)
    # weights_only prevents executable checkpoint payloads; tensor dictionaries only.
    # Official training checkpoints also save argparse.Namespace metadata.
    with torch.serialization.safe_globals([Namespace]):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not {'model', 'model_ema1'} <= checkpoint.keys():
        raise ValueError('Expected official checkpoint with model and model_ema1; refusing random/raw weights.')
    model = cls(args)
    model.load_state_dict(checkpoint['model'], strict=True)
    parameters = dict(model.named_parameters())
    ema = checkpoint['model_ema1']
    missing = set(parameters) - set(ema)
    if missing:
        raise ValueError(f'Incomplete first EMA: {sorted(missing)[:5]}')
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(ema[name])
    del checkpoint
    return model.eval().requires_grad_(False).to(device)
