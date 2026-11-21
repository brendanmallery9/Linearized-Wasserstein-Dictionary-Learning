"""Sequential EXP-007/008 supervisor entrypoint; no outcome-based stage selection."""
import argparse,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out-root',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--workers',type=int,default=8);p.add_argument('--resume',action='store_true');a=p.parse_args();root=Path(a.out_root);root.mkdir(parents=True,exist_ok=a.resume)
def status(**kw):(root/'stages.json').write_text(json.dumps(dict(epoch=time.time(),**kw),indent=2))
for stage,families in [('exp007-iid',['iid']),('exp008-depth',['depth_iid','depth_smooth','scan_stripes'])]:
 out=root/stage
 if (out/'completion.json').exists():
  assert a.resume and json.loads((out/'completion.json').read_text())['exit_code']==0;continue
 cmd=[sys.executable,'-u',str(HERE/'run.py'),'--out',str(out),'--device',a.device,'--workers',str(a.workers),'--families',*families,'--sigmas','.075','.10','.15','.20','--seeds','0','1','2','--max-hours','12']
 if out.exists():
  assert a.resume,'Existing partial stage: inspect and explicitly --resume';cmd+=['--resume']
 with (root/f'{stage}.log').open('ab',buffering=0) as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);status(state='running',stage=stage,worker_pid=proc.pid,command=cmd);code=proc.wait()
 if code:status(state='failed',stage=stage,exit_code=code);raise SystemExit(code)
status(state='analyzing')
with (root/'analysis.log').open('ab',buffering=0) as log:r=subprocess.run([sys.executable,str(HERE/'report.py'),'--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
status(state='numerically_complete_review_pending' if r.returncode==0 else 'analysis_failed',exit_code=r.returncode);raise SystemExit(r.returncode)
