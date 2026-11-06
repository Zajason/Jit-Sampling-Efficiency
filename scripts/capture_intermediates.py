"""Untimed conditional clean predictions for frozen finalists, using saved pairs."""
import argparse
from dataclasses import asdict
import json
import numpy as np
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from jit_pilot.samplers import Configuration, sample
from jit_pilot.run import noise_for,to_images
from jit_pilot.upstream import load_model,verify_source,ROOT,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    selection=json.loads((args.study/'selection.json').read_text())
    manifest=json.loads((args.study/'final-1/manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Final generation must complete first')
    pairs=json.loads((args.study/'final-1/pairs.json').read_text())[:4]
    args.output.mkdir(parents=True,exist_ok=False)
    verify_source(ROOT/'vendor/official-jit')
    model=load_model(ROOT/'vendor/official-jit',args.checkpoint,torch.device('cuda'),manifest['settings']['cfg'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    labels=torch.tensor([p['label'] for p in pairs],device='cuda')
    z=noise_for([p['id'] for p in pairs],manifest['settings']['seed']).cuda()
    records=[]
    for item in selection['finalists']:
        cfg=Configuration(**item)
        grid=torch.linspace(0.,1.,cfg.steps+1).pow(cfg.schedule_power)
        # Last time is never used by the final Euler update.
        targets=sorted(set(float(grid[:-1][torch.argmin(abs(grid[:-1]-t))]) for t in [0,.25,.5,.75,.95]))
        captured={}
        def hook(module,inputs,output):
            t=float(inputs[1][0])
            if int(inputs[2][0])==1000:return
            target=next((s for s in targets if abs(t-s)<1e-6),None)
            if target is not None and target not in captured:
                captured[target]=to_images(output.detach())
        handle=model.net.register_forward_hook(hook)
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=to_images(sample(model,z,labels,cfg))
        finally:handle.remove()
        for pair,generated in zip(pairs,result):
            with Image.open(args.study/'final-1/samples'/cfg.name/f'{pair["id"]:06d}.png') as saved:
                if not np.array_equal(np.asarray(saved),generated):
                    raise ValueError(f'Untimed capture differs from saved final output: {cfg.name}, {pair["id"]}')
        tile=192;header=44;left=80
        image=Image.new('RGB',(left+tile*(len(targets)+1),header+tile*len(pairs)),'white');draw=ImageDraw.Draw(image)
        for col,t in enumerate(targets):
            draw.text((left+col*tile+8,12),f'conditional x_pred, t={t:.3f}',fill='black')
            for row,pair in enumerate(pairs):image.paste(Image.fromarray(captured[t][row]).resize((tile,tile)),(left+col*tile,header+row*tile))
        draw.text((left+len(targets)*tile+8,12),'Final image',fill='black')
        for row,pair in enumerate(pairs):
            draw.text((4,header+row*tile+10),f'id {pair["id"]}\nclass {pair["label"]}',fill='black')
            image.paste(Image.fromarray(result[row]).resize((tile,tile)),(left+len(targets)*tile,header+row*tile))
        image.save(args.output/f'{cfg.name}.png')
        records.append({'configuration':asdict(cfg),'captured_times':targets})
    metadata={'timed':False,'description':'Raw conditional network clean-image predictions, clipped for display; first conditional evaluation at each selected time. Heun intermediate times may be evaluated on its Euler predictor state. Not ground truth or additional quality measurements.','pairs':pairs,'checkpoint_sha256':sha256(args.checkpoint),'captures':records}
    (args.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(args.output)

if __name__=='__main__':main()
