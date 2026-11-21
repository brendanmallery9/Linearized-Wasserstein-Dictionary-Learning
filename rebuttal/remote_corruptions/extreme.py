"""EXP-015 bounded extension and numerical/visual artifact generation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,subprocess,json,time,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
cmd=[sys.executable,'-u',str(HERE/'run.py'),'--device','cuda','--workers','8','--families','iid','--sigmas','.25','.30','.40','--seeds','0','1','2','--max-hours','6','--out',str(out)]
r=subprocess.run(cmd)
if r.returncode:raise SystemExit(r.returncode)
import report
report.one(out,6)
f=out/'report/REPORT.md';s=f.read_text().replace('StageA and stageB have separate prespecified8/24-contrast reconstruction families.','EXP-015 has a prespecified6-contrast reconstruction family.');f.write_text(s)
import numpy as np,torch,joblib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
bank=joblib.load(HERE/'assets/frozen_models.joblib');data=torch.load(HERE/'assets/holdout_540.pt',weights_only=False);indices=[0,90,180,270,360,450]
models=['eot_eta0.001_seed0','free_eta0.01_seed0','map_pca','sc_positive'];lookup={x['name']:x for x in bank}
for sigma in [0,.25,.30,.40]:
 folder=out/('clean_sigma0_seed0' if sigma==0 else f'iid_sigma{sigma:g}_seed0');inputs=torch.load(folder/'inputs.pt',weights_only=False);rec=[]
 for name in models:
  rr=lookup[name];z=torch.from_numpy(np.load(folder/f'{name}.npz')['codes']);rec.append((rr['B']+torch.einsum('bm,mnd->bnd',z,rr['D'])).numpy())
 fig=plt.figure(figsize=(18,15));names=['Clean','Observed','Constrained s0','Tuned free s0','PCA','Positive SC']
 for row,i in enumerate(indices):
  clouds=[data['raw'][i].numpy(),inputs['clouds'][i],*[x[i] for x in rec]];allpts=np.concatenate(clouds);lo=allpts.min(0);hi=allpts.max(0);c=(lo+hi)/2;rad=max(hi-lo)/2
  for col,(cloud,name) in enumerate(zip(clouds,names)):
   ax=fig.add_subplot(6,6,row*6+col+1,projection='3d');ax.scatter(*np.asarray(cloud).T,s=.5);ax.set(xlim=(c[0]-rad,c[0]+rad),ylim=(c[1]-rad,c[1]+rad),zlim=(c[2]-rad,c[2]+rad));ax.set_box_aspect((1,1,1));ax.view_init(20,40);ax.set_axis_off()
   if row==0:ax.set_title(name)
 fig.suptitle(f'Fixed examples, draw0, sigma={sigma:g}; common limits per object');fig.tight_layout();fig.savefig(out/'report'/f'fixed_sigma{sigma:g}.png',dpi=120);plt.close(fig)
# Regenerate checksum manifest after adding figures/amending the experiment label.
import hashlib
R=out/'report';(R/'SHA256SUMS.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'},indent=2))
(out/'finalizer.json').write_text(json.dumps(dict(exit_code=0,state='numerically_complete_visual_review_pending',epoch=time.time()),indent=2))
