"""Execute the proposal study sequentially; save selection before final seeds."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from jit_pilot.samplers import CONFIGS, Configuration
from jit_pilot.upstream import ROOT, verify_source

VALIDATION_SEED=20271106
FINAL_SEEDS=[20281106,20291106]

def verify_noise_ranges(settings):
    ranges=[(settings['seed'],settings['seed']+settings['samples']-1),
            (VALIDATION_SEED,VALIDATION_SEED+1999),
            *[(seed,seed+4999) for seed in FINAL_SEEDS]]
    for i,(start,end) in enumerate(ranges):
        for other_start,other_end in ranges[i+1:]:
            if max(start,other_start)<=min(end,other_end):
                raise ValueError('Screen, validation and final noise seed ranges overlap')
    return ranges


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Reuse completed generation/FID stages; never overwrite partial generation')
    p.add_argument('--select-only',action='store_true',help='Freeze selection from completed screens without launching final CUDA runs')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    state=json.loads((args.output/'status.json').read_text()) if args.resume else {'started':datetime.now(timezone.utc).isoformat(),'completed':[]}
    state.update(status='running');state.pop('error',None)
    def save():
        (args.output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def execute(command,log,sentinel,status_required=False):
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        completed_at=None
        while process.poll() is None:
            complete=False
            if sentinel.exists():
                try:
                    content=json.loads(sentinel.read_text())
                    complete=content.get('status')=='complete' if status_required else 'metrics' in content
                except (OSError,json.JSONDecodeError):pass
            if complete:
                completed_at=completed_at or time.monotonic()
                if time.monotonic()-completed_at>90:
                    # Only after all generation/metric artifacts are saved. Some
                    # Python/CUDA combinations can stall in interpreter cleanup.
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            raise RuntimeError('CUDA process cannot exit after completed output; possible driver kernel wait. Recover the CUDA environment before resuming.')
                    with (args.output/'cleanup-events.jsonl').open('a') as events:
                        events.write(json.dumps({'command':command,'sentinel':str(sentinel),'reason':'Interpreter failed to exit for 90 seconds after complete artifacts were saved; stopped process group outside all sampler timing.','returncode':process.returncode})+'\n')
                    return
            time.sleep(1)
        if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
    def run_stage(name,settings):
        verify_source(ROOT/'vendor/official-jit')
        state['stage']=name;save()
        cfg=args.output/f'{name}.json'
        out=args.output/name
        existing=out.exists()
        if existing:
            manifest=json.loads((out/'manifest.json').read_text())
            if not args.resume or manifest['status']!='complete' or manifest['settings']!=settings:
                raise ValueError(f'Refusing partial or mismatched stage: {out}')
        else:
            cfg.write_text(json.dumps(settings,indent=2)+'\n')
        with (args.output/f'{name}.log').open('a' if existing else 'w') as log:
            if not existing:
                execute([sys.executable,'-m','jit_pilot.run','--checkpoint',str(args.checkpoint),'--config',str(cfg),'--output',str(out)],log,out/'manifest.json',True)
            if not (out/'fid_screen.json').exists():
                execute([sys.executable,'-m','jit_pilot.fid','--run',str(out)],log,out/'fid_screen.json')
        if name not in state['completed']:state['completed'].append(name)
        save()
        rows=list(csv.DictReader((out/'summary.csv').open()))
        metrics=json.loads((out/'fid_screen.json').read_text())['metrics']
        for row in rows:
            row['fid']=metrics[row['configuration']]['frechet_inception_distance']
        return rows
    def make_settings(configs,seed,samples,description):
        base=json.loads((ROOT/'configs/quality-screen.json').read_text())
        known={c.name for c in CONFIGS}
        base.update(description=description,seed=seed,samples=samples,configurations=[c.name for c in configs],additional_configurations=[asdict(c) for c in configs if c.name not in known])
        return base
    try:
        uniform=json.loads((ROOT/'configs/solver-budgets.json').read_text())
        ranges=verify_noise_ranges(uniform)
        (args.output/'noise-seed-ranges.json').write_text(json.dumps(ranges,indent=2)+'\n')
        rows=run_stage('uniform-screen',uniform)
        ref=next(r for r in rows if r['configuration']=='heun_50')
        candidates=[r for r in rows if r['configuration']!='heun_50']
        eligible=[r for r in candidates if r['fid']<=1.1*ref['fid']]
        chosen=min(eligible,key=lambda r:float(r['median_batch_ms'])) if eligible else min(candidates,key=lambda r:r['fid'])
        fixed=min((r for r in candidates if int(r['network_passes_per_image'])==50),key=lambda r:r['fid'])
        lookup={c.name:c for c in CONFIGS}
        lookup.update({c['name']:Configuration(**c) for c in uniform['additional_configurations']})
        solver=lookup[fixed['configuration']]
        schedules=[lookup['heun_50'],solver]
        for power in [.5,2.]:
            schedules.append(Configuration(f'{solver.name}_p{str(power).replace(".","_")}',solver.method,solver.steps,power))
        validation=run_stage('schedule-validation',make_settings(schedules,VALIDATION_SEED,2000,'Validation only: fixed solver and 50-pass budget; t=u^p with p=0.5,1,2.'))
        best=min((r for r in validation if r['configuration']!='heun_50'),key=lambda r:r['fid'])
        schedule_lookup={c.name:c for c in schedules}
        finalists=[lookup['heun_50'],lookup[chosen['configuration']]]
        if best['configuration'] not in {c.name for c in finalists}:finalists.append(schedule_lookup[best['configuration']])
        decision={'uniform_candidate':chosen,'screen_reference':ref,'uniform_selection_rule':'fastest within 1.10x reference screen FID, otherwise lowest FID','schedule_solver':asdict(solver),'schedule_validation_winner':best,'finalists':[asdict(c) for c in finalists],'final_seeds':FINAL_SEEDS,'samples_per_configuration_per_repeat':5000,'frozen_at':datetime.now(timezone.utc).isoformat()}
        selection=args.output/'selection.json'
        if selection.exists():
            previous=json.loads(selection.read_text())
            if {k:v for k,v in previous.items() if k!='frozen_at'}!={k:v for k,v in decision.items() if k!='frozen_at'}:
                raise ValueError('Frozen selection differs; refusing to overwrite')
        else:selection.write_text(json.dumps(decision,indent=2)+'\n')
        if args.select_only:
            state.update(status='awaiting_cuda_recovery',stage='final-evaluation-pending');save();return
        for repeat,seed in enumerate(decision['final_seeds'],1):
            run_stage(f'final-{repeat}',make_settings(finalists,seed,5000,'Frozen finalists; independent final seeds; resource-limited 5000-image evaluation.'))
        state['status']='complete';state['finished']=datetime.now(timezone.utc).isoformat();save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));save();raise

if __name__=='__main__':main()
