#!/usr/bin/env python3
"""Portable frozen-model evaluation. No Mac paths, dataset training, or custom pickles."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time,sys,hashlib,threading,warnings,subprocess,platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch,joblib,ot
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
A=Path(__file__).resolve().parent/'assets';SIGMAS=[.075,.10,.15,.20,.25,.30,.40]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def gate(z,D,q):
 if not q:return z
 score=z.abs()*D.square().sum(-1).mean(-1).sqrt();return z*torch.zeros_like(z).scatter(1,score.topk(min(q,z.shape[1]),dim=1).indices,1)
def infer(r,maps,device):
 if r['type']=='linear':return torch.from_numpy(r['linear_model'].transform((maps-r['B']).reshape(len(maps),-1).double().numpy()/np.sqrt(1000))).float()
 with torch.no_grad():
  D=r['D'].to(device);B=r['B'].to(device);w={k:v.to(device) for k,v in r['encoder_state'].items()};codes=[]
  for chunk in maps.split(128):
   inner=torch.einsum('bnd,mnd->bm',chunk.to(device)-B,D)/1000
   if 'gain' in w:inner=inner*w['gain']
   z=torch.relu(inner+w['biases.0'])
   for step in range(1,len([k for k in w if k.startswith('biases.')])):z=torch.relu(inner+z@w[f'laterals.{step-1}'].T+w[f'biases.{step}'])
   codes.append(z.cpu())
  return gate(torch.cat(codes),r['D'],r['k'])
def generated(raw,ids,family,sigma,draw):
 clouds=[];metadata=[]
 for mesh,p in zip(ids,raw.numpy().astype('float64')):
  center=p.mean(0);radius=np.linalg.norm(p-center,axis=1).max();u=(p-center)/radius;key=int(hashlib.sha256(mesh.encode()).hexdigest()[:8],16)
  if family=='iid':
   # Matches the declared local EXP-007 draws exactly; legacy RandomState is deliberate.
   rng=np.random.RandomState(np.random.SeedSequence([20261202,draw,SIGMAS.index(sigma)+1,key]).generate_state(1)[0]);delta=sigma*rng.randn(*u.shape);meta={'namespace':20261202}
  else:
   view_rng=np.random.default_rng(np.random.SeedSequence([20261204,draw,key]));v=view_rng.normal(size=3);v/=np.linalg.norm(v);a=np.cross(v,np.eye(3)[np.argmin(abs(v))]);a/=np.linalg.norm(a);b=np.cross(v,a);coords=np.column_stack([u@a,u@b])
   rng=np.random.default_rng(np.random.SeedSequence([20261204,draw,key,{'depth_iid':1,'depth_smooth':2,'scan_stripes':3}[family]]));meta={'view_direction':v.tolist(),'transverse_a':a.tolist(),'transverse_b':b.tolist(),'namespace':20261204}
   if family=='depth_iid':f=rng.normal(size=len(u))
   elif family=='depth_smooth':
    omega=rng.normal(size=(32,2))/.25;phase=rng.uniform(0,2*np.pi,32);coeff=rng.normal(size=32);f=np.cos(coords@omega.T+phase)@coeff;meta.update(correlation_length=.25,omega=omega.tolist(),phase=phase.tolist(),coefficients=coeff.tolist())
   elif family=='scan_stripes':
    phase=float(rng.uniform(0,2*np.pi));f=np.sin(2*np.pi*coords[:,0]/.25+phase);meta.update(period=.25,phase=phase)
   else:raise ValueError(family)
   f=f-f.mean();rms=np.sqrt(np.mean(f*f));assert rms>1e-8,'Degenerate depth field; preserve failure, do not drop mesh';f=f/rms;delta=np.sqrt(3)*sigma*f[:,None]*v;meta.update(empirical_centered=True,depth_rms=float(np.sqrt(3)*sigma),actual_displacement_rms=float(np.linalg.norm(delta,axis=1).dot(np.linalg.norm(delta,axis=1))**.5/np.sqrt(len(u))))
  clouds.append(((u+delta)*radius+center).astype('float32'));metadata.append(dict(mesh=mesh,center=center.tolist(),radius=float(radius),**meta))
 return clouds,metadata

def mapping(clouds,X,pool):
 def one(p):
  p=p.astype('float64');cost=ot.dist(X,p,metric='sqeuclidean');plan=ot.emd(np.ones(len(X))/len(X),np.ones(len(p))/len(p),cost,numThreads=1);return torch.from_numpy(len(X)*plan@p).float()
 return torch.stack(list(pool.map(one,clouds)))
def geometry(clean,pred,pool,exact):
 def one(ab):
  a,b=(np.asarray(x,dtype='float64') for x in ab);ch=cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean();w=ot.emd2(np.ones(len(a))/len(a),np.ones(len(b))/len(b),ot.dist(a,b,metric='sqeuclidean'),numThreads=1) if exact else 0.;return float(ch),float(w)
 a=np.array(list(pool.map(one,zip(clean,pred))));return a[:,0],a[:,1]
def verify():
 for name,h in json.loads((A/'SHA256SUMS.json').read_text()).items():assert sha(A/name)==h,name

def smoke(bank,device,pool):
 t=torch.load(A/'smoke_reference.pt',weights_only=False);errors=[]
 for r in bank:
  z=infer(r,t['maps'],'cpu');expected=t['expected'][r['name']];err=float(np.max(abs(z.numpy()-expected['codes'])));assert np.allclose(z,expected['codes'],atol=2e-5,rtol=2e-4),(r['name'],err)
  for q in range(1,31):
   sc,clf,_=r['probes'][q-1];pr=clf.predict(sc.transform(gate(z,r['D'],q).numpy()));assert np.array_equal(pr,expected['predictions'][q-1]),(r['name'],q,'CPU probe mismatch')
  if device=='cuda':
   zz=infer(r,t['maps'],device);assert torch.allclose(zz,z,atol=2e-5,rtol=2e-4),(r['name'],'CUDA code mismatch')
   for q in range(1,31):
    sc,clf,_=r['probes'][q-1];assert np.array_equal(clf.predict(sc.transform(gate(zz,r['D'],q).numpy())),expected['predictions'][q-1]),(r['name'],q,'CUDA label mismatch; use CPU pending investigation')
  p=r['B']+torch.einsum('bm,mnd->bnd',z[:2],r['D']);_,w=geometry(t['raw'][:2].numpy(),p.numpy(),pool,True);assert np.allclose(w,expected['w2'][:2],atol=1e-7)
  errors.append(dict(name=r['name'],max_cpu_code_error=err))
 data=torch.load(A/'holdout_540.pt',weights_only=False);X=torch.load(A/'reference_X.pt',weights_only=False).double().numpy();m=mapping(list(data['raw'][:2].numpy()),X,pool);assert torch.allclose(m,data['maps'][:2],atol=2e-5)
 for family in ['iid','depth_iid','depth_smooth','scan_stripes']:
  p,meta=generated(data['raw'][:2],data['identities'][:2],family,.075,0);q,_=generated(data['raw'][:2],data['identities'][:2],family,.075,0);assert all(np.array_equal(x,y) for x,y in zip(p,q))
  if family!='iid':assert all(abs(x['actual_displacement_rms']-np.sqrt(3)*.075)<1e-9 for x in meta)
 return dict(status='passed',device=device,models=errors,exact_geometry_and_mapping=True,corruption_reproducibility=True,depth_rms_matching=True)

def main(args):
 verify();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if args.device=='cuda':assert torch.cuda.is_available(),'CUDA requested but unavailable'
 bank=joblib.load(A/'frozen_models.joblib');pool=ThreadPoolExecutor(args.workers)
 if args.smoke:
  result=smoke(bank,args.device,pool);print(json.dumps(result,indent=2));pool.shutdown();return
 out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume);data=torch.load(A/'holdout_540.pt',weights_only=False)
 if args.pilot:
  ix=np.array([np.flatnonzero(data['labels']==i)[0] for i in range(6)]);data={**data,**{k:data[k][ix] for k in ['raw','maps','labels']},'identities':[data['identities'][i] for i in ix]}
 N=len(data['labels']);settings=dict(families=args.families,sigmas=args.sigmas,seeds=args.seeds,device=args.device,workers=args.workers,pilot=args.pilot,N=N,asset_manifest_sha256=sha(A/'SHA256SUMS.json'),source_sha256=sha(__file__))
 protocol=out/'protocol.json'
 if args.resume:
  meta=json.loads(protocol.read_text());assert meta['settings']==settings,'Resume configuration/source mismatch'
 else:
  meta=dict(settings=settings,start_epoch=time.time(),deadline_epoch=time.time()+args.max_hours*3600,command=sys.argv,python=sys.version,platform=platform.platform(),torch=torch.__version__,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_diff=subprocess.check_output(['git','diff','HEAD'],text=True),identities=data['identities']);save(protocol,meta)
 def watch():
  while not (out/'completion.json').exists():
   if time.time()>meta['deadline_epoch']:save(out/'budget_termination.json',dict(exit_code=124,time=time.time()));os._exit(124)
   time.sleep(10)
 threading.Thread(target=watch,daemon=True).start();X=torch.load(A/'reference_X.pt',weights_only=False).double().numpy();clean=data['raw'].numpy();conditions=[('clean',0.,0)]+[(f,s,d) for f in args.families for s in args.sigmas for d in args.seeds]
 try:
  for i,(family,sigma,draw) in enumerate(conditions):
   folder=out/f'{family}_sigma{sigma:g}_seed{draw}';folder.mkdir(exist_ok=True)
   if (folder/'complete.json').exists():continue
   import shutil
   assert shutil.disk_usage(out).free>2*2**30,'2GiB disk reserve';start=time.time();case=dict(family=family,sigma=sigma,draw=draw,N=N,index=i+1,total=len(conditions));save(out/'status.json',dict(state='mapping',**case))
   if (folder/'inputs.pt').exists():a=torch.load(folder/'inputs.pt',weights_only=False);clouds,maps=a['clouds'],a['maps']
   else:
    clouds,fields=(list(clean),[]) if family=='clean' else generated(data['raw'],data['identities'],family,sigma,draw);maps=data['maps'] if family=='clean' else mapping(clouds,X,pool);torch.save(dict(clouds=clouds,maps=maps,fields=fields),folder/'inputs.pt')
   if not (folder/'input_floor.npz').exists():ch,w=geometry(clean,clouds,pool,True);np.savez_compressed(folder/'input_floor.npz',chamfer=ch,w2=w)
   rows=[]
   for r in bank:
    name=r['name'];mark=folder/f'{name}.json'
    if mark.exists():rows.extend(json.loads(mark.read_text()));continue
    save(out/'status.json',dict(state='evaluating',current=name,**case));z=infer(r,maps,args.device);pr=[];chs=[];losses=[];w2=None;records=[]
    for q in range(1,31):
     zz=gate(z,r['D'],q);sc,clf,val=r['probes'][q-1];pred=clf.predict(sc.transform(zz.numpy()));pr.append(pred);row=dict(**case,name=name,recipe=r['recipe'],training_seed=r['seed'],q=q,accuracy=float(np.mean(pred==data['labels'])),validation_accuracy=val,C=float(clf.C))
     if q in [1,3,5,30]:
      p=r['B']+torch.einsum('bm,mnd->bnd',zz,r['D']);ch,w=geometry(clean,p.numpy(),pool,q==30);loss=.5*(p-data['maps']).square().sum((1,2))/1000;chs.append(ch);losses.append(loss.numpy());row.update(chamfer=float(ch.mean()),map_loss=float(loss.mean()))
      if q==30:w2=w;row['w2_squared']=float(w.mean())
     records.append(row)
    np.savez_compressed(folder/f'{name}.npz',codes=z.numpy(),predictions=np.array(pr),chamfer=np.array(chs),map_loss=np.array(losses),w2=w2,labels=data['labels']);save(mark,records);rows.extend(records)
   save(folder/'metrics.json',rows);save(folder/'complete.json',dict(**case,seconds=time.time()-start,rows=len(rows)));print('COMPLETE',family,sigma,draw,round(time.time()-start,1),flush=True)
  ch,w=geometry(clean,bank[0]['B'].unsqueeze(0).repeat(N,1,1).numpy(),pool,True);np.savez_compressed(out/'mean_floor.npz',chamfer=ch,w2=w);save(out/'completion.json',dict(state='completed',exit_code=0,N=N,pilot=args.pilot,conditions=len(conditions),seconds=time.time()-meta['start_epoch']));save(out/'status.json',dict(state='completed',conditions=len(conditions)))
 except BaseException as ex:
  import traceback
  save(out/'failure.json',dict(type=type(ex).__name__,traceback=traceback.format_exc()));raise
 finally:pool.shutdown()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--families',nargs='+',choices=['iid','depth_iid','depth_smooth','scan_stripes'],default=['iid']);p.add_argument('--sigmas',nargs='+',type=float,choices=SIGMAS,default=SIGMAS);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--max-hours',type=float,default=12);p.add_argument('--smoke',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if not a.smoke and not a.out:p.error('--out required')
 warnings.filterwarnings('error',message='numItermax reached.*|Problem infeasible.*|Problem unbounded.*')
 with threadpool_limits(1):main(a)
