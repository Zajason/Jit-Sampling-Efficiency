"""Print the pilot's cost accounting without PyTorch, GPU, or checkpoint."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
c=json.loads((root/'configs/pilot.json').read_text())
total=0
print('Configuration       Steps  Network passes / image')
for name in c['configurations']:
    method,steps=name.rsplit('_',1);steps=int(steps)
    passes=2*(2*steps-1 if method=='heun' else steps)
    total+=passes*c['samples']
    print(f'{name:20} {steps:4}  {passes:5}')
print(f'\n{c["samples"]} paired images x {len(c["configurations"])} configurations = {c["samples"]*len(c["configurations"])} images')
print(f'{total:,} sample-equivalent network passes for image generation, excluding warmup, timing and checks.')
extra_per_config=c['batch_size']*(c['warmup']+c['repeats']+1)
full=total+sum(2*(2*int(name.rsplit('_',1)[1])-1 if name.startswith('heun_') else int(name.rsplit('_',1)[1]))*extra_per_config for name in c['configurations'])
print(f'{full:,} sample-equivalent network passes including default warmup, timing and checks.')
print('No GPU results have been measured by this command.')
