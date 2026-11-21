#!/usr/bin/env python3
"""EXP-004 fresh-data preparation, locked-recipe evaluation and uncertainty."""
import argparse
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import ot
import torch
from threadpoolctl import threadpool_limits

import pointcloud_fixed_width_ablation as a
from pointcloud.pipeline.prepare_pointcloud_ot import parse_off, triangulate, sample_surface, normalize_to_unit_cube

s = a.s
ROOT, OUT = a.ROOT, a.OUT
FRESH = OUT/'fresh_holdout'
PRIOR = ROOT/'pointcloud/results/rebuttal_exp002_transport_20261120/fresh_holdout/data.pt'


def prepare():
    FRESH.mkdir(exist_ok=True)
    if (FRESH/'data.pt').exists():
        return
    old = json.loads((a.OLD/'data.json').read_text())
    classes = old['metadata']['classes']
    seen = torch.load(PRIOR, weights_only=False)['identities']
    X = torch.load(a.OLD/'geometry.pt', weights_only=False)['X']
    archive = ROOT/'pointcloud_raw/ModelNet10.zip'
    # Audit the concurrent study's actual inputs, not only its prose description.
    manifest_path = ROOT/'pointcloud/results/rebuttal_exp003_corruptions_20261120/provenance/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert str(PRIOR) in manifest['inputs']
    assert manifest['inputs'][str(PRIOR)] == s.sha(PRIOR)
    selected = []
    with zipfile.ZipFile(archive) as z:
        for c in classes:
            names = sorted(n for n in z.namelist() if n.startswith(f'ModelNet10/{c}/test/') and n.endswith('.off'))
            selected.extend((c, n) for n in names if n not in seen)
    identities = [n for c, n in selected]
    original = {r['mesh'] for r in old['identities']}
    assert len(selected) == 540 and len(set(identities)) == 540
    assert not set(identities).intersection(seen) and not set(identities).intersection(original)
    # Search saved result metadata for test identities beyond the known60.
    # Rows inside this new run are excluded because preparation itself records them.
    import re
    observed = set(); audited = []
    pattern = re.compile(r'ModelNet10/[^/"\\\s]+/test/[^"\\\s]+\.off')
    for run in sorted((ROOT/'pointcloud/results').glob('rebuttal_exp*')):
        if run == OUT or 'exp004' in run.name:
            continue
        for p in run.rglob('*.json'):
            if p.stat().st_size > 30*2**20:
                continue
            hits = set(pattern.findall(p.read_text(errors='replace')))
            if hits:
                observed.update(hits); audited.append(dict(path=str(p.relative_to(ROOT)), sha256=s.sha(p), test_identities=len(hits)))
    unexpected = observed-set(seen)
    assert not unexpected, ('Other official test meshes already referenced in metadata', sorted(unexpected))
    protocol = dict(start=s.now(), mesh_selection='all remaining official-test meshes in the six existing classes',
                    identities=identities, classes=classes, rows=540, old_fresh_excluded=seen,
                    surface_seed_base=20261129, reference='original1000 fixed source points',
                    raw_points=1024, normalization='unchanged existing unit-cube/padding.05 pipeline',
                    archive_sha256=s.sha(archive), prior_fresh_sha256=s.sha(PRIOR),
                    concurrent_manifest_sha256=s.sha(manifest_path),
                    other_saved_test_identities=sorted(observed), audited_metadata=audited,
                    disjoint_original_and_previous_fresh=True, outcomes_evaluated=False)
    s.save_json(FRESH/'protocol.json', protocol)
    raw=[]; maps=[]; labels=[]
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as tmp:
        for j, (c, name) in enumerate(selected):
            path=Path(tmp)/'mesh.off'; path.write_bytes(z.read(name))
            vertices, faces=parse_off(path)
            cloud=normalize_to_unit_cube(sample_surface(vertices, triangulate(faces), 1024,
                                                        np.random.default_rng(20261129+j)))
            cost=ot.dist(X.double().numpy(), cloud.astype('float64'), metric='sqeuclidean')
            plan=ot.emd(np.ones(1000)/1000, np.ones(1024)/1024, cost)
            raw.append(torch.from_numpy(cloud).float())
            maps.append(torch.from_numpy(1000*plan@cloud).float())
            labels.append(classes.index(c))
            if j%90 == 0:
                print('prepared', j+1, '/540', name, flush=True)
    raw=torch.stack(raw); maps=torch.stack(maps); labels=np.array(labels)
    assert torch.isfinite(raw).all() and torch.isfinite(maps).all()
    centroid_error=float((raw.mean(1)-maps.mean(1)).abs().max())
    assert centroid_error < 2e-5
    rng=np.random.default_rng(20261130)
    geometry_indices=np.concatenate([rng.choice(np.flatnonzero(labels==c), 10, replace=False) for c in range(6)]).tolist()
    torch.save(dict(raw=raw, maps=maps, labels=labels, identities=identities,
                    classes=classes, geometry_indices=geometry_indices), FRESH/'data.pt')
    shutil.copy2(__file__, FRESH/'preparation_source.py')
    s.save_json(FRESH/'checks.json', dict(end=s.now(), rows=540, class_counts=np.bincount(labels).tolist(),
                data_sha256=s.sha(FRESH/'data.pt'), centroid_error=centroid_error,
                no_mesh_overlap=True, geometry_indices=geometry_indices, outcomes_evaluated=False))
    print('Prepared540 independent meshes; no classifier outcomes evaluated', flush=True)


def selected_names(selection):
    return {name for r in selection['recipes'].values() for name in r.values()}


def frozen_probe(z, labels, split):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler=StandardScaler().fit(z[split['train']])
    best=-1
    for C in [.1,1.,10.]:
        clf=LogisticRegression(C=C,max_iter=1500).fit(scaler.transform(z[split['train']]),labels[split['train']])
        score=clf.score(scaler.transform(z[split['validation']]),labels[split['validation']])
        if score>best:
            chosen=clf;best=score
    return scaler,chosen


def evaluate():
    prepare()
    complete=json.loads((OUT/'training_complete.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    lock_inputs={str(p.relative_to(ROOT)):s.sha(p) for p in [OUT/'selection.json', OUT/'training_complete.json', FRESH/'data.pt', OUT/'protocol.json']}
    shutil.copy2(__file__, OUT/'provenance'/Path(__file__).name)
    s.save_json(FRESH/'evaluation_lock.json', dict(time=s.now(), command=[sys.executable,*sys.argv],
                source_sha256=s.sha(__file__), inputs=lock_inputs, training_completed=complete,
                selection=selection, test_selection_prohibited=True))
    fresh=torch.load(FRESH/'data.pt', weights_only=False)
    oldmeta=json.loads((a.OLD/'data.json').read_text()); classes=oldmeta['metadata']['classes']
    labels=np.array([classes.index(r['class']) for r in oldmeta['identities']])
    all_labels=np.concatenate([labels, fresh['labels']]); split=json.loads((OUT/'split.json').read_text())
    fresh_split=dict(split, test=list(range(1800,2340)))
    selected=selected_names(selection)
    curves=[]; metrics=[]; predictions={}; start=time.monotonic()
    geometry_ids=fresh['geometry_indices']
    for path in sorted(OUT.glob('*/model.pt')):
        name=path.parent.name; cfg=json.loads((path.parent/'config.json').read_text())
        base_name=name.rsplit('seed',1)[0]+'seed0'
        chosen=base_name in selected
        ck=torch.load(path, weights_only=False); model=a.build(cfg, ck['X'], ck['baseline'])
        mismatch=model.load_state_dict(ck['state'],strict=False)
        allowed={'X','displacement_center','atoms_module.X','atoms_module.Y','atoms_module.cost'}
        assert set(mismatch.missing_keys)<=allowed and not mismatch.unexpected_keys, mismatch
        saved=torch.load(path.parent/'outputs.pt',weights_only=False)
        z, D=a.infer(model,fresh['maps'],ck['baseline'])
        assert torch.allclose(D,saved['directions'],atol=2e-6)
        B=ck['baseline']; all_z=torch.cat([saved['codes'],z])
        q_values=list(range(1,31)) if chosen else a.QS
        kinds=['adaptive','global'] if chosen else ['adaptive']
        scores=all_z.abs()*D.square().sum(-1).mean(-1).sqrt()
        global_order=torch.argsort(scores[split['train']].mean(0),descending=True,stable=True)
        this_predictions={}; model_curves=[]
        for kind in kinds:
            for q in q_values:
                # Global filtering is a companion control with q fixed coordinates.
                if kind=='global':
                    features=all_z[:,global_order[:q]]
                else:
                    features=a.truncate(all_z,D,q)
                probe, pred=s.fit_probe(features.numpy(),all_labels,fresh_split,[.1,1.,10.])
                key=f'{kind}_q{q}'; this_predictions[key]=pred.tolist()
                row=dict(name=name, family=cfg['family'], grid_side=cfg['grid_side'],
                         c=cfg['sparsity_coeff'], seed=cfg['seed'], kind=kind, q=q,
                         selected_recipe=chosen, **probe)
                curves.append(row); model_curves.append(row)
        if chosen:
            # Companion diagnostics: how much information is in support identity,
            # and whether filtering works without retraining the full-code probe.
            scaler,clf=frozen_probe(all_z.numpy(),all_labels,split)
            for q in [1,2,3,4,5,10,15,20,30]:
                features=a.truncate(all_z,D,q)
                pr=clf.predict(scaler.transform(features[-540:].numpy()))
                curves.append(dict(name=name,family=cfg['family'],grid_side=cfg['grid_side'],
                    c=cfg['sparsity_coeff'],seed=cfg['seed'],kind='adaptive_frozen_full_probe',q=q,
                    selected_recipe=True,accuracy=float((pr==fresh['labels']).mean()),C=float(clf.C)))
                this_predictions[f'adaptive_frozen_full_probe_q{q}']=pr.tolist()
                if q<=5:
                    probe,pr=s.fit_probe((features!=0).float().numpy(),all_labels,fresh_split,[.1,1.,10.])
                    curves.append(dict(name=name,family=cfg['family'],grid_side=cfg['grid_side'],
                        c=cfg['sparsity_coeff'],seed=cfg['seed'],kind='support_only',q=q,
                        selected_recipe=True,**probe))
                    this_predictions[f'support_only_q{q}']=pr.tolist()
        p=B+torch.einsum('bm,mnd->bnd',z,D)
        geom,w2,ch=s.geometry(fresh['raw'],p,geometry_ids)
        native=json.loads((path.parent/'metrics.json').read_text())
        row=dict(name=name, config=cfg, selected_recipe=chosen,
                 trainable_parameters=native['trainable_parameters'], checkpoint_bytes=native['checkpoint_bytes'],
                 fit_seconds=native['fit_seconds'], epochs_completed=native['epochs_completed'],
                 best_epoch=native['best_epoch'], termination_reason=native['termination_reason'],
                 validation_low_activity_accuracy=native['validation_low_activity_accuracy'],
                 validation_full_accuracy=native['validation_full_accuracy'],
                 full_accuracy=next(r['accuracy'] for r in model_curves if r['kind']=='adaptive' and r['q']==30),
                 low_activity_accuracy=float(np.mean([r['accuracy'] for r in model_curves if r['kind']=='adaptive' and r['q']<=5])),
                 map_half_L2rho=float(.5*(p-fresh['maps']).square().sum((1,2)).mean()/1000),
                 mean_active=float((z.abs()>1e-4).sum(1).float().mean()),
                 eval_w2_squared=w2, eval_chamfer=ch, **geom)
        if chosen:
            row['sparse_geometry']={}
            for q in [3,5]:
                zz=a.truncate(z,D,q); pp=B+torch.einsum('bm,mnd->bnd',zz,D)
                g,ww,cc=s.geometry(fresh['raw'],pp,geometry_ids)
                row['sparse_geometry'][str(q)]=dict(**g,eval_w2_squared=ww,eval_chamfer=cc)
        metrics.append(row); predictions[name]=this_predictions
        torch.save(dict(codes=z,directions=D,baseline=B,identities=fresh['identities']),FRESH/f'{name}_codes.pt')
        s.save_json(FRESH/'metrics.json',metrics); s.save_json(FRESH/'curves.json',curves)
        s.save_json(FRESH/'predictions.json',predictions)
        print('evaluated',len(metrics),name,'selected',chosen,'full',row['full_accuracy'],'low',row['low_activity_accuracy'],flush=True)
        del model
    s.save_csv(FRESH/'metrics.csv',[{k:v for k,v in r.items() if k not in ['config','sparse_geometry','eval_w2_squared','eval_chamfer']}|r['config'] for r in metrics])
    s.save_csv(FRESH/'curves.csv',curves)
    s.save_json(FRESH/'evaluation_complete.json',dict(time=s.now(),seconds=time.monotonic()-start,models=len(metrics),curves=len(curves),exit_status=0))
    summarize()


def summarize():
    selection=json.loads((OUT/'selection.json').read_text())['recipes']
    metrics=json.loads((FRESH/'metrics.json').read_text()); lookup={r['name']:r for r in metrics}
    predictions=json.loads((FRESH/'predictions.json').read_text())
    fresh=torch.load(FRESH/'data.pt',weights_only=False); labels=fresh['labels']
    rng=np.random.default_rng(20261201)
    draws=np.stack([np.concatenate([rng.choice(np.flatnonzero(labels==c),90,replace=True) for c in range(6)]) for _ in range(5000)])
    geometry_labels=labels[fresh['geometry_indices']]
    geometry_draws=np.stack([np.concatenate([rng.choice(np.flatnonzero(geometry_labels==c),10,replace=True) for c in range(6)]) for _ in range(5000)])
    def names(base):
        return sorted(n for n in lookup if n.rsplit('seed',1)[0]==base.rsplit('seed',1)[0])
    groups={}; contrasts=[]
    for key,recipes in selection.items():
        for purpose,base in recipes.items():
            nn=names(base); rows=[lookup[n] for n in nn]
            group=dict(names=nn,seeds=[r['config']['seed'] for r in rows],purpose=purpose,
                       trainable_parameters=rows[0]['trainable_parameters'])
            for metric in ['low_activity_accuracy','full_accuracy','w2_squared','chamfer_unsquared','map_half_L2rho']:
                vals=np.array([r[metric] for r in rows]); group[metric]=dict(mean=float(vals.mean()),seed_sd=float(vals.std(ddof=1)) if len(vals)>1 else None,values=vals.tolist())
            groups[f'{key}__{purpose}']=group
    for grid in a.GRIDS:
        for purpose,qs in [('low_activity',range(1,6)),('full_accuracy',[30])]:
            cgroup=groups[f'eot_g{grid}__{purpose}']; fgroup=groups[f'free_g9__{purpose}']
            cn=cgroup['names']; fn=fgroup['names']
            seeds=sorted(set(cgroup['seeds']).intersection(fgroup['seeds']))
            if not seeds: continue
            cn=[next(n for n in cn if lookup[n]['config']['seed']==seed) for seed in seeds]
            fn=[next(n for n in fn if lookup[n]['config']['seed']==seed) for seed in seeds]
            def correct(nn):
                return np.stack([[np.array(predictions[n][f'adaptive_q{q}'])==labels for q in qs] for n in nn]).mean((0,1))
            delta=correct(cn)-correct(fn); boot=delta[draws].mean(1)
            c_w=np.mean([lookup[n]['eval_w2_squared'] for n in cn],axis=0)
            f_w=np.mean([lookup[n]['eval_w2_squared'] for n in fn],axis=0)
            ratio=c_w[geometry_draws].mean(1)/f_w[geometry_draws].mean(1)
            contrasts.append(dict(grid_side=grid,purpose=purpose,matched_seeds=seeds,
                primary=grid==9,constrained_names=cn,free_names=fn,
                accuracy_difference_pp=float(100*delta.mean()),
                conditional_mesh_ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),
                conditional_mesh_ci97_5_pp=(100*np.quantile(boot,[.0125,.9875])).tolist(),
                w2_squared_ratio=float(c_w.mean()/f_w.mean()),
                conditional_geometry_ratio_ci95=np.quantile(ratio,[.025,.975]).tolist(),
                uncertainty_scope='paired stratified meshes conditional on fitted seeds, not initialization-population guarantee'))
    summary=dict(time=s.now(),groups=groups,contrasts=contrasts,
                 primary_comparisons='grid9 activity and grid9 full_accuracy; other grids exploratory',
                 shape_count=540,geometry_shape_count=60,bootstrap_draws=5000,
                 working_accuracy_tolerance_pp=2,working_w2_ratio_tolerance=1.25)
    s.save_json(OUT/'analysis.json',summary)
    s.save_csv(OUT/'group_summary.csv',[dict(group=k,parameters=g['trainable_parameters'],seeds=len(g['seeds']),**{m:g[m]['mean'] for m in ['low_activity_accuracy','full_accuracy','w2_squared','chamfer_unsquared','map_half_L2rho']}) for k,g in groups.items()])
    s.save_json(OUT/'status.json',dict(status='evaluation_complete_pending_inspection',time=s.now()))
    print('Saved aggregate and paired conditional uncertainty',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        if args.prepare: prepare()
        elif args.summarize: summarize()
        else: evaluate()
