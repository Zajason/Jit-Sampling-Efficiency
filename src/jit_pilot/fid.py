"""Separate, equal-count FID screen using the same evaluator API as official JiT."""
import argparse
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from .upstream import ROOT, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stats',type=Path,default=ROOT/'vendor/official-jit/fid_stats/jit_in256_stats.npz')
    args=p.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    settings=manifest['settings'];count=settings['samples']
    if manifest['status']!='complete':p.error('Generation run is not complete')
    if count<2000 or count%1000:p.error('FID screen requires >=2000 samples and an equal number per ImageNet class')
    if not args.stats.is_file():p.error('Reference statistics missing; fetch the official jit_in256_stats.npz file')
    pairs=json.loads((args.run/'pairs.json').read_text())
    from collections import Counter
    counts=Counter(pair['label'] for pair in pairs)
    if len(pairs)!=count or len(counts)!=1000 or set(counts.values())!={count//1000}:
        p.error('Saved class labels are not balanced')
    folders=[]
    for name in settings['configurations']:
        folder=args.run/'samples'/name
        files=sorted(folder.glob('*.png'))
        if [f.name for f in files]!=[f'{i:06d}.png' for i in range(count)]:
            p.error(f'{name}: image IDs/count differ from manifest')
        folders.append((name,folder))
    import torch
    if not torch.cuda.is_available():p.error('CUDA GPU required for this FID screen')
    import torch_fidelity
    # Upstream uses a fork exposing fid_statistics_file, absent in stock PyPI versions.
    source=inspect.getsource(importlib.import_module('torch_fidelity.metrics'))
    if 'fid_statistics_file' not in source:
        p.error('Install the author-maintained LTH14/torch-fidelity fork, not the unmodified PyPI package')
    lock=json.loads((ROOT/'configs/fid-upstream.json').read_text())
    dist=importlib.metadata.distribution('torch-fidelity')
    installed_source=json.loads(dist.read_text('direct_url.json') or '{}')
    if installed_source.get('vcs_info',{}).get('commit_id')!=lock['revision']:
        p.error('FID evaluator revision is unverified; run python scripts/install_fid.py')
    out=args.run/'fid_screen.json'
    if out.exists():p.error('fid_screen.json already exists; refusing to overwrite')
    # Match official JiT: ISC + FID avoids the fork's FID-only path,
    # which does not consume the reference-statistics file.
    report={'sample_count_per_configuration':count,'stats_sha256':sha256(args.stats),
        'evaluator_version':importlib.metadata.version('torch-fidelity'),
        'evaluator_source': installed_source,
        'inception_batch_size':32,
        'warning':'Small-sample screening only; not confirmation that FID is within 10% of reference.',
        'metrics':{}}
    for name,folder in folders:
        report['metrics'][name]=torch_fidelity.calculate_metrics(
            input1=str(folder),input2=None,fid_statistics_file=str(args.stats),
            cuda=True,isc=True,fid=True,kid=False,prc=False,verbose=True,batch_size=32)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
