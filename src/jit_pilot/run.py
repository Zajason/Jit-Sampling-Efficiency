"""Single-GPU paired sampling and timing. No training and no fake benchmark mode."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
from .samplers import CONFIGS, Configuration, expected_network_passes, sample
from .upstream import ROOT, load_model, sha256, verify_source

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')

def labels_for(count, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(1000, generator=generator).tolist()
    return [permutation[i % 1000] for i in range(count)]

def noise_for(ids, seed, size=256):
    return torch.stack([torch.randn(3, size, size,
        generator=torch.Generator().manual_seed(seed+i)) for i in ids])

def to_images(samples):
    if not torch.isfinite(samples).all():
        raise RuntimeError('Non-finite generated samples; sampler is unstable.')
    # Matches upstream quantization: clip then round (not truncate).
    x = ((samples.float().cpu() + 1) / 2).clamp(0, 1)
    return (x * 255).round().to(torch.uint8).permute(0,2,3,1).numpy()

def percentile(values, fraction):
    return float(np.quantile(values, fraction))

def make_grid(output, configs, labels, count=8):
    count = min(count, len(labels))
    tile, header, left = 150, 40, 110
    grid = Image.new('RGB', (left + tile*len(configs), header+tile*count), 'white')
    draw = ImageDraw.Draw(grid)
    for c, cfg in enumerate(configs):
        draw.text((left+c*tile+8, 12), cfg.name, fill='black')
    for row in range(count):
        draw.text((8,header+row*tile+12),f'id {row}\nclass {labels[row]}',fill='black')
        for col,cfg in enumerate(configs):
            with Image.open(output/'samples'/cfg.name/f'{row:06d}.png') as image:
                grid.paste(image.resize((tile,tile)),(left+col*tile,header+row*tile))
    grid.save(output/'paired_grid.png')

def paired_metrics(output, configs, count, use_lpips):
    perceptual = None
    if use_lpips:
        import lpips
        # CPU avoids interfering with sampler timing and GPU memory measurements.
        perceptual = lpips.LPIPS(net='alex').eval()
    rows=[]
    for config in configs:
        squared, perceptual_values=[],[]
        for i in range(count):
            with Image.open(output/'samples'/'heun_50'/f'{i:06d}.png') as im:
                ref=np.array(im,dtype=np.float32)/255
            with Image.open(output/'samples'/config.name/f'{i:06d}.png') as im:
                candidate=np.array(im,dtype=np.float32)/255
            squared.append(float(np.mean((ref-candidate)**2)))
            if perceptual is not None:
                a=torch.from_numpy(ref).permute(2,0,1)[None]*2-1
                b=torch.from_numpy(candidate).permute(2,0,1)[None]*2-1
                with torch.no_grad():
                    perceptual_values.append(float(perceptual(a,b).item()))
        rows.append({'configuration':config.name,
            'paired_pixel_rmse_0_1':float(np.sqrt(np.mean(squared))),
            'paired_lpips_alex': statistics.mean(perceptual_values) if perceptual_values else None})
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pilot.json')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,default=ROOT/'vendor/official-jit')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--lpips',action='store_true',help='Optional paired perceptual agreement; downloads LPIPS weights on first use')
    args=parser.parse_args()
    settings=json.loads(args.config.read_text())
    count,batch=settings['samples'],settings['batch_size']
    if count<batch or batch<1 or settings['warmup']<1 or settings['repeats']<3:
        parser.error('Require samples >= batch >= 1, warmup >= 1, repeats >= 3')
    lookup={c.name:c for c in CONFIGS}
    for item in settings.get('additional_configurations', []):
        config=Configuration(**item)
        if config.name in lookup or not config.name or config.name in {'.','..'} or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_.' for c in config.name):
            parser.error('Additional configuration names must be unique safe path components')
        lookup[config.name]=config
    configs=[lookup[name] for name in settings['configurations']]
    if len({c.name for c in configs})!=len(configs) or 'heun_50' not in settings['configurations']:
        parser.error('Configurations must be distinct and include heun_50 reference')
    if settings['precision'] not in {'bf16','fp32'}:
        parser.error('precision must be bf16 or fp32')
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required for official JiT. CPU tests are available via unittest; no benchmark has run.')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint file does not exist. Download JiT-B/16 from the official README.')
    device=torch.device('cuda:0')
    if settings['precision']=='bf16' and not torch.cuda.is_bf16_supported():
        parser.error('This GPU does not support bf16; explicitly set precision=fp32 in a copied config.')
    lock=verify_source(args.upstream)
    output=args.output or ROOT/'results'/datetime.now(timezone.utc).strftime('pilot-%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True,exist_ok=False)
    # Checkpoint hash and software/hardware provenance make runs identifiable.
    metadata={'status':'running','settings':settings,'upstream':lock,
        'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha256(args.checkpoint),
        'torch':torch.__version__,'cuda':torch.version.cuda,'platform':platform.platform(),
        'gpu':torch.cuda.get_device_name(0),'gpu_capability':torch.cuda.get_device_capability(0),
        'timing_boundary':'Sampler with initial noise/labels already on GPU; includes network and solver; excludes RNG, transfers, loading, metrics and saving',
        'compile':False,'tf32':False,'lpips_requested':args.lpips,
        'resolved_configurations':[asdict(c) for c in configs],
        'config_sha256':sha256(args.config),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_diff':subprocess.check_output(['git','diff'],cwd=ROOT,text=True),
        'packages':subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True).splitlines()}
    try:
        metadata['nvidia_smi']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        metadata['nvidia_smi']=None
    write_json(output/'manifest.json',metadata)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    model=load_model(args.upstream,args.checkpoint,device,settings['cfg'])
    labels=labels_for(count,settings['seed'])
    write_json(output/'pairs.json',[{'id':i,'label':v,'noise_seed':settings['seed']+i} for i,v in enumerate(labels)])
    def autocast():
        return torch.autocast('cuda',dtype=torch.bfloat16) if settings['precision']=='bf16' else nullcontext()
    timing_ids=list(range(batch))
    initial=noise_for(timing_ids,settings['seed']).to(device)
    timing_labels=torch.tensor(labels[:batch],device=device)
    def generate(config):
        with torch.inference_mode(),autocast():
            return sample(model,initial,timing_labels,config)
    # Instrument one untimed batch, then remove hooks before warmup and timing.
    passes={}
    for config in configs:
        calls=[0]
        def hook(module,inputs): calls[0]+=1
        handle=model.net.register_forward_pre_hook(hook)
        try:
            result=generate(config)
            if not torch.isfinite(result).all():
                raise RuntimeError(f'{config.name}: non-finite output')
            del result
        finally:
            handle.remove()
        assert calls[0]==expected_network_passes(config.method,config.steps),(config.name,calls[0])
        passes[config.name]=calls[0]
        print(f'{config.name}: verified {calls[0]} network calls per batch ({calls[0]} sample-equivalent passes per image)',flush=True)
        for _ in range(settings['warmup']):
            result=generate(config)
            del result
    torch.cuda.synchronize()
    timings={c.name:[] for c in configs};peaks={c.name:[] for c in configs}
    order_log=[];rng=random.Random(settings['seed'])
    for repeat in range(settings['repeats']):
        order=list(configs);rng.shuffle(order);order_log.append([c.name for c in order])
        for config in order:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize();start=time.perf_counter()
            result=generate(config)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            timings[config.name].append(elapsed)
            peaks[config.name].append(torch.cuda.max_memory_allocated())
            del result
        print(f'Timing round {repeat+1}/{settings["repeats"]}',flush=True)
    write_json(output/'timings.json',{'batch_ms':timings,'peak_allocated_bytes':peaks,'orders':order_log})
    # Generate the same per-image noise independently of batching, for every solver.
    for config in configs:
        folder=output/'samples'/config.name;folder.mkdir(parents=True)
        for start in range(0,count,batch):
            ids=list(range(start,min(count,start+batch)))
            z=noise_for(ids,settings['seed']).to(device)
            y=torch.tensor([labels[i] for i in ids],device=device)
            with torch.inference_mode(),autocast():
                images=to_images(sample(model,z,y,config))
            for i,image in zip(ids,images):
                Image.fromarray(image).save(folder/f'{i:06d}.png')
            del z,y,images
            if (start+batch) % 100 == 0:
                print(f'{config.name}: saved {min(start+batch,count)}/{count}',flush=True)
        print(f'Saved {count} images: {config.name}',flush=True)
    agreement={r['configuration']:r for r in paired_metrics(output,configs,count,args.lpips)}
    reference=statistics.median(timings['heun_50'])
    rows=[]
    for config in configs:
        ms=timings[config.name]
        rows.append({'configuration':config.name,'steps':config.steps,
            'schedule_power':config.schedule_power,
            'actual_forward_calls_per_batch':passes[config.name],
            'network_passes_per_image':passes[config.name],
            'guided_velocity_evaluations':passes[config.name]//2,
            'batch_size':batch,'median_batch_ms':statistics.median(ms),
            'batch_ms_p25':percentile(ms,.25),'batch_ms_p75':percentile(ms,.75),
            'amortized_ms_per_image':statistics.median(ms)/batch,
            'speedup_vs_heun_50':reference/statistics.median(ms),
            'peak_allocated_gib':max(peaks[config.name])/1024**3,
            **agreement[config.name]})
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    make_grid(output,configs,labels)
    report=['# JiT pilot results','',f'Samples per configuration: {count}; timing batch size: {batch}.',
        '', '| Sampler | Passes/image | Batch ms | Amortized ms/image | Speedup | Paired pixel RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append(f'| {row["configuration"]} | {row["network_passes_per_image"]} | {row["median_batch_ms"]:.2f} | {row["amortized_ms_per_image"]:.2f} | {row["speedup_vs_heun_50"]:.2f}x | {row["paired_pixel_rmse_0_1"]:.4f} |')
    report+=['','Timing excludes initial-noise creation and transfer, model loading, file saving and metrics.',
        'Per-image time is amortized batch cost, not single-image latency unless batch size is 1.',
        'Paired distances measure agreement with Heun-50, not image quality. No FID has been computed by this command.',
        'Inspect paired_grid.png before selecting candidates for the separate 2000-image FID screen.']
    (output/'report.md').write_text('\n'.join(report)+'\n')
    metadata['status']='complete';write_json(output/'manifest.json',metadata)
    print(f'Completed: {output.resolve()}')

if __name__=='__main__': main()
