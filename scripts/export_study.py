"""Export measured study tables and presentation figures; never synthesize results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'heun':'#334155','euler':'#D97706','ab2':'#007F86'}
plt.rcParams.update({'font.size':16,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

def load(run):
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError(f'Incomplete run: {run}')
    rows=list(csv.DictReader((run/'summary.csv').open()))
    fid=run/'fid_screen.json'
    metrics=json.loads(fid.read_text())['metrics'] if fid.exists() else {}
    for row in rows:
        row.update(run=run.name,seed=manifest['settings']['seed'],samples=manifest['settings']['samples'])
        row['fid']=metrics.get(row['configuration'],{}).get('frechet_inception_distance')
    return rows

def save(fig,out,name):
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    pilot=load(args.pilot)
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    labels=[r['configuration'].replace('_','-') for r in pilot]
    colors=[COLORS[r['configuration'].split('_')[0]] for r in pilot]
    axs[0].barh(labels,[float(r['amortized_ms_per_image']) for r in pilot],color=colors)
    axs[0].set_xlabel('Amortized sampler time (ms/image)');axs[0].invert_yaxis()
    axs[0].set_title('Measured latency · batch 4')
    axs[1].barh(labels,[float(r['paired_lpips_alex']) for r in pilot],color=colors)
    axs[1].set_xlabel('Paired AlexNet LPIPS (lower = closer)');axs[1].invert_yaxis()
    axs[1].set_title('Agreement with generated Heun-50')
    fig.suptitle('64 paired images · RTX 5080 · bf16',fontsize=22)
    save(fig,args.output,'pilot-latency-agreement')
    stages=['uniform-screen','schedule-validation','final-1','final-2']
    rows=[];by_stage={}
    for stage in stages:
        run=args.study/stage
        if not run.exists():continue
        data=load(run);rows.extend(data);by_stage[stage]=data
    if rows:
        fields=list(rows[0])
        with (args.output/'all-measurements.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    for stage in ['uniform-screen','schedule-validation']:
        data=by_stage.get(stage,[])
        if not data or any(r['fid'] is None for r in data):continue
        fig,ax=plt.subplots(figsize=(13,7))
        for method in ['euler','heun','ab2']:
            group=[r for r in data if r['configuration'].startswith(method+'_')]
            if not group:continue
            group.sort(key=lambda r:float(r['amortized_ms_per_image']))
            ax.plot([float(r['amortized_ms_per_image']) for r in group],[r['fid'] for r in group],marker='o',linestyle='-' if stage=='uniform-screen' else 'none',label=method.upper(),color=COLORS[method])
            for r in group:
                if stage=='uniform-screen':
                    if method!='ab2' and r['configuration']!='heun_50':continue
                    label=r['network_passes_per_image']+' passes'
                    offset=(-85,10) if r['configuration']=='heun_50' else (5,-18)
                else:
                    label='Heun-50 reference' if r['configuration']=='heun_50' else {0.5:'AB2: t = √u',1.0:'AB2: uniform t = u',2.0:'AB2: t = u²'}[float(r['schedule_power'])]
                    offset=(-130,10) if r['configuration']=='heun_50' else (5,7)
                ax.annotate(label,(float(r['amortized_ms_per_image']),r['fid']),xytext=offset,textcoords='offset points',fontsize=11)
        frontier=[r for r in data if not any(float(s['amortized_ms_per_image'])<=float(r['amortized_ms_per_image']) and s['fid']<=r['fid'] and (float(s['amortized_ms_per_image'])<float(r['amortized_ms_per_image']) or s['fid']<r['fid']) for s in data)]
        frontier.sort(key=lambda r:float(r['amortized_ms_per_image']))
        ax.plot([float(r['amortized_ms_per_image']) for r in frontier],[r['fid'] for r in frontier],color='#94A3B8',linestyle='--',label='Measured Pareto frontier',zorder=0)
        ax.set_xlabel('Amortized sampler time (ms/image, batch 4)');ax.set_ylabel('FID ↓ (2,000 images; screening only)')
        ax.set_title('Uniform solver comparison' if stage=='uniform-screen' else 'Fixed-solver schedule validation',fontsize=22)
        if stage=='uniform-screen':
            ax.set_ylim(bottom=min(r['fid'] for r in data)-1.2)
            ax.text(.98,.70,'Points: nominal 10, 20, 30, 50, 100 passes/image\nHeun realizes 18 and 98 at budgets 20 and 100',ha='right',transform=ax.transAxes,fontsize=11)
        ax.legend();ax.grid(alpha=.15)
        save(fig,args.output,stage+'-frontier')
    final=[r for stage in ['final-1','final-2'] for r in by_stage.get(stage,[])]
    references={r['run']:r['fid'] for r in final if r['configuration']=='heun_50'}
    for r in final:
        r['fid_ratio_to_reference']=r['fid']/references[r['run']] if r['fid'] is not None else None
    summary=[]
    for name in dict.fromkeys(r['configuration'] for r in final):
        group=[r for r in final if r['configuration']==name]
        fids=[r['fid'] for r in group]
        if any(v is None for v in fids):continue
        summary.append(dict(configuration=name,repeats=len(group),samples_per_repeat=group[0]['samples'],fid_mean=statistics.mean(fids),fid_sd=statistics.stdev(fids) if len(fids)>1 else None,fid_per_repeat=fids,fid_ratio_per_repeat=[r['fid_ratio_to_reference'] for r in group],amortized_ms_per_image_per_repeat=[float(r['amortized_ms_per_image']) for r in group],speedup_per_repeat=[float(r['speedup_vs_heun_50']) for r in group],observed_target_met_per_repeat=[r['fid_ratio_to_reference']<=1.10 and float(r['speedup_vs_heun_50'])>=2 for r in group]))
    (args.output/'final-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if summary:
        fig,ax=plt.subplots(figsize=(12,7))
        for r in summary:
            x=statistics.mean(r['amortized_ms_per_image_per_repeat']);y=r['fid_mean']
            ax.errorbar(x,y,xerr=statistics.stdev(r['amortized_ms_per_image_per_repeat']) if r['repeats']>1 else None,yerr=r['fid_sd'],fmt='o',capsize=6,color=COLORS[r['configuration'].split('_')[0]])
            ax.annotate(r['configuration'],(x,y),xytext=(8,8),textcoords='offset points')
        ax.set_xlabel('Mean amortized sampler time (ms/image, batch 4)');ax.set_ylabel('FID ↓ (5,000 images per repeat)')
        ax.set_title('Frozen finalists · two independent seeds',fontsize=22);ax.grid(alpha=.15)
        ax.text(.02,.02,'Error bars: SD across two repeats, not confidence intervals',transform=ax.transAxes,fontsize=11)
        save(fig,args.output,'final-quality-latency')
    lines=['# Measured JiT experiment results','','All charts use actual pretrained JiT-B/16 outputs. No training.','', 'Timing: synchronized sampler only; excludes model loading, noise generation/transfers, warm-up, call-count checks, image saving and metrics. Batch 4 bf16 on RTX 5080; per-image times are amortized, not single-image latency.','', 'Pilot: 64 paired images/configuration. LPIPS and pixel RMSE measure agreement with generated Heun-50, not distributional quality.','', 'Screen/validation: 2,000 balanced images/configuration. Schedule tuning uses separate validation seeds. Final protocol: two independent seeds, 5,000 balanced images/configuration/repeat. FID variability from two repeats is descriptive; no population-level 10% claim.','']
    if not summary:
        lines+=['**Final evaluations have not run.** CUDA driver recovery is required. The completed screen and validation results below do not establish the proposal target.','']
    for r in summary:
        lines.append(f'- {r["configuration"]}: FID {r["fid_mean"]:.3f} ± {r["fid_sd"]:.3f} across {r["repeats"]} repeats; individual FIDs {r["fid_per_repeat"]}; speedups {r["speedup_per_repeat"]}; FID/reference ratios {r["fid_ratio_per_repeat"]}. Observed 2x/1.10x target met per repeat: {r["observed_target_met_per_repeat"]}. These are finite-sample observations, not a population-level tolerance guarantee.')
    screen=by_stage.get('uniform-screen',[])
    if screen:
        lines+=['','## Matched-pass AB2 versus Euler (screening only)','']
        for ab in [r for r in screen if r['configuration'].startswith('ab2_')]:
            eu=next(r for r in screen if r['configuration']=='euler_'+ab['configuration'].split('_')[1])
            lines.append(f'- {ab["network_passes_per_image"]} passes/image: Euler FID {eu["fid"]:.3f}; AB2 FID {ab["fid"]:.3f}.')
    validation=by_stage.get('schedule-validation',[])
    if validation:
        candidates=[r for r in validation if r['configuration']!='heun_50']
        baseline=next(r for r in candidates if float(r['schedule_power'])==1)
        best=min(candidates,key=lambda r:r['fid'])
        lines+=['','## Schedule validation (separate seeds)','',f'Fixed solver {baseline["configuration"]}; uniform FID {baseline["fid"]:.3f}. Lowest validation FID: {best["configuration"]}, {best["fid"]:.3f}. Selection uses these validation results only.']
    lines+=['','Full settings, exact labels/noise seeds, checkpoint/source hashes, timing repetitions and evaluator provenance are saved in each run directory.','PNG figures are suitable for slides; SVG/PDF versions are provided for resizing.']
    (args.output/'presentation-notes.md').write_text('\n'.join(lines)+'\n')
    print(args.output.resolve())

if __name__=='__main__':main()
