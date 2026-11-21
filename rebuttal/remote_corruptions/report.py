"""Audit and summarize separate family/sigma results; never pool corruptions."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
NAMES={'eot_eta0.001':'Constrained compact','free_eta0.01':'Free tuned','map_pca':'PCA','sc_positive':'Positive SC','sc_signed':'Signed SC','eot_k3':'Constrained hard-k3','free_k3':'Free hard-k3'}
COLORS={'eot_eta0.001':'#0072B2','free_eta0.01':'#D55E00','map_pca':'#777777','sc_positive':'#CC79A7','sc_signed':'#009E73'}
def one(stage,primary_family_size,allow_pilot=False,bootstrap_reps=20000):
 done=json.loads((stage/'completion.json').read_text());assert done['exit_code']==0 and ((done['N']==540 and not done['pilot']) or (allow_pilot and done['pilot']));N=done['N'];rows=[];vec={};count=0;labels=None
 for folder in sorted(stage.iterdir()):
  if not folder.is_dir() or not (folder/'complete.json').exists():continue
  case=json.loads((folder/'complete.json').read_text());ms=json.loads((folder/'metrics.json').read_text());assert len(ms)==450;count+=1
  for name in {m['name'] for m in ms}:
   z=np.load(folder/f'{name}.npz');assert z['predictions'].shape==(30,N) and all(np.isfinite(z[k]).all() for k in z.files);labels=z['labels'];assert np.array_equal(np.unique(labels,return_counts=True)[1],np.full(6,N//6))
   model=[m for m in ms if m['name']==name];recipe=model[0]['recipe']
   for m in model:
    q=m['q'];assert abs(np.mean(z['predictions'][q-1]==labels)-m['accuracy'])<1e-12
    if q==30:assert abs(z['w2'].mean()-m['w2_squared'])<1e-12
    if q in [1,3,5,30]:
     j=[1,3,5,30].index(q);assert abs(z['chamfer'][j].mean()-m['chamfer'])<1e-12;assert abs(float(z['map_loss'][j].mean())-m['map_loss'])<1e-7
    rows.append(m)
   corr=z['predictions']==labels[None,:];vec.setdefault((case['family'],case['sigma'],recipe),[]).append(dict(accuracy=corr[-1].astype(float),accuracy_q1to5=corr[:5].mean(0),w2_squared=z['w2']))
 assert count==done['conditions'];R=stage/'report';R.mkdir(exist_ok=True);df=pd.DataFrame(rows);df.to_csv(R/'metrics.csv',index=False);seed=df.groupby(['family','sigma','recipe','training_seed','q'])[['accuracy','w2_squared','chamfer','map_loss']].mean().reset_index();seed.to_csv(R/'seed_means.csv',index=False)
 v={k:{m:np.mean([a[m] for a in xs],0) for m in ['accuracy','accuracy_q1to5','w2_squared']} for k,xs in vec.items()};rng=np.random.default_rng(20261205);ix=np.concatenate([rng.choice(np.flatnonzero(labels==i),(bootstrap_reps,N//6)) for i in range(6)],1);intervals=[];families=sorted(f for f in df.family.unique() if f!='clean');sigmas=sorted(s for s in df.sigma.unique() if s>0)
 for f in families:
  for sigma in sigmas:
   for comp in ['map_pca','sc_positive','free_eta0.01','sc_signed']:
    for metric in ['accuracy','accuracy_q1to5','w2_squared']:
     d=v[f,sigma,'eot_eta0.001'][metric]-v[f,sigma,comp][metric];cl=v['clean',0.,'eot_eta0.001'][metric]-v['clean',0.,comp][metric]
     for adjusted in [False,True]:
      delta=d-(cl if adjusted else 0);b=delta[ix].mean(1);r=dict(family=f,sigma=sigma,comparator=comp,metric=metric,clean_adjusted=adjusted,difference=float(delta.mean()),pointwise95=np.quantile(b,[.025,.975]).tolist(),n_shapes=N,conditional_on_models_and_draws=True)
      if comp in ['map_pca','sc_positive'] and metric=='w2_squared' and not adjusted:r['family_bonferroni']=np.quantile(b,[.05/(2*primary_family_size),1-.05/(2*primary_family_size)]).tolist()
      intervals.append(r)
 (R/'paired_intervals.json').write_text(json.dumps(intervals,indent=2));lines=['# Frozen noise results','', 'Numerical execution/audit complete; automatic report, visual/scientific review pending. Each family/sigma is separate; within-cell draw means then training-seed mean/sample SD. Neural3 dictionaries, classical1. Heldout540 objects already inspected in preceding studies; exploratory, not untouched confirmation.','']
 plt.rcParams.update({'font.size':16,'axes.labelsize':17,'figure.facecolor':'white'})
 for f in families:
  lines+=['## '+f,'']
  for metric,title,qs,scale,digits in [('accuracy','Full classification (%)',[30],100,2),('w2_squared','Full W2 squared (lower is better)',[30],1,6),('accuracy','Mean accuracy across q1–5 (%)',list(range(1,6)),100,2)]:
   lines+=['### '+title,'','| Model | Clean | '+ ' | '.join(f'σ={s:g}' for s in sigmas)+' |','|---|'+ '|'.join(['---:']*(len(sigmas)+1))+'|']
   for recipe,name in NAMES.items():
    cells=[]
    for sigma in [0,*sigmas]:
     sub=seed[(seed.family==('clean' if sigma==0 else f))&(seed.sigma==sigma)&(seed.recipe==recipe)&seed.q.isin(qs)].groupby('training_seed')[metric].mean();cells.append(f'{scale*sub.mean():.{digits}f}'+(f' ± {scale*sub.std():.{digits}f}' if len(sub)>1 else ''))
    lines.append('| '+name+' | '+' | '.join(cells)+' |')
   lines+=['']
  fig,axs=plt.subplots(2,1,figsize=(7.2,10),layout='constrained');fig2,axes=plt.subplots(len(sigmas),1,figsize=(7.2,4*len(sigmas)),layout='constrained')
  for recipe,color in COLORS.items():
   for ax,metric,scale in [(axs[0],'accuracy',100),(axs[1],'w2_squared',1)]:
    ss=seed[(seed.family.isin(['clean',f]))&(seed.recipe==recipe)&(seed.q==30)].groupby('sigma')[metric].mean();ax.plot(ss.index,scale*ss,'-o',color=color,label=NAMES[recipe],lw=2.4)
   for sigma,ax in zip(sigmas,np.atleast_1d(axes)):
    a=seed[(seed.family==f)&(seed.sigma==sigma)&(seed.recipe==recipe)].groupby('q').accuracy.mean();ax.plot(a.index,100*a,color=color,lw=2.4,label=NAMES[recipe]);ax.set(title=f'σ={sigma:g}',xlabel='Retained atoms',ylabel='Accuracy (%)')
  floors=[]
  for sigma in [0,*sigmas]:
   folders=[p for p in stage.iterdir() if p.is_dir() and (p/'complete.json').exists() and (c:=json.loads((p/'complete.json').read_text()))['family']==('clean' if sigma==0 else f) and c['sigma']==sigma];floors.append(float(np.mean([np.load(p/'input_floor.npz')['w2'].mean() for p in folders])))
  mean=float(np.load(stage/'mean_floor.npz')['w2'].mean());axs[1].plot([0,*sigmas],floors,':',color='#222222',label='Corrupted input');axs[1].axhline(mean,color='#888888',ls='--',label='Mean shape');axs[0].set_ylabel('Accuracy (%)');axs[1].set_ylabel('Clean-target W2 squared');axs[0].legend(fontsize=11,ncol=2);axs[1].legend(fontsize=11)
  for ax in axs:ax.set_xlabel('Isotropic-equivalent noise σ');ax.grid(alpha=.2)
  np.atleast_1d(axes)[0].legend(fontsize=11,ncol=2);fig.suptitle(f,fontsize=22);fig2.suptitle(f+' · activity curves',fontsize=21);fig.savefig(R/f'{f}_native.png',dpi=150);fig2.savefig(R/f'{f}_budgets.png',dpi=150);plt.close(fig);plt.close(fig2)
  lines+=[f'[Native curves]({f}_native.png) · [All-sigma atom-budget curves]({f}_budgets.png)','']
 lines+=['## Interpretation and review','', 'No automatic winner claim: inspect absolute errors against BOTH PCA and positive SC, family-adjusted bounds, clean-relative degradation, raw-input/mean floors and useful retained classification. Low error growth alone is not a crossover; regression to an average at chance prediction is not recovered geometry. Keep negative outcomes and all controls visible. StageA and stageB have separate prespecified8/24-contrast reconstruction families. Other endpoints are companions, not post-hoc replacements.','', 'Numerical audit covers saved predictions/error means and finite arrays. Inspect plots and fixed-example reconstructions before calling this scientifically reviewed. Protocol, raw field/maps/codes/predictions/exit records remain in the parent directory; source asset checksums and exact command/commit are in protocol.json.']
 (R/'REPORT.md').write_text(('INTEGRATION ONLY — NOT SCIENTIFIC RESULTS\n\n' if done['pilot'] else '')+'\n'.join(lines));(R/'audit.json').write_text(json.dumps(dict(conditions=count,metric_rows=len(rows),sample_size=N,all_native_metrics_reproduced=True,visual_review_pending=True),indent=2));(R/'SHA256SUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'},indent=2));return R
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);reports=[]
 for name,family in [('exp007-iid',8),('exp008-depth',24)]:reports.append(one(root/name,family))
 (root/'REPORT.md').write_text('# Remote noise study\n\nNumerical execution/audit complete; visual/scientific review pending.\n\n'+ '\n'.join(f'- [{r.parent.name}]({r.parent.name}/report/REPORT.md)' for r in reports)+'\n');print('Both stages numerically audited; visual review pending')
