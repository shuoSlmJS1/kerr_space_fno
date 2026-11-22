"""Frozen T1200 Q400 comparison; no training, tuning, or normalization fitting."""
import argparse
import atexit
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
OUT=Path('/workspace/shanjinshuo/outputs/fno_kerr/priority1_q400_comparison_t1200')
DATA=Path('/workspace/shanjinshuo/datasets/fno_kerr/priority1_q400_comparison_t1200/q_1p6007-2p9993_n400_t1200')
TRAIN=Path('/workspace/shanjinshuo/datasets/fno_kerr/priority1_spline_t1200_val_20261011T0341Z')
SPLINE=Path('/workspace/shanjinshuo/outputs/fno_kerr/priority1_spline_t1200_val_20261011T0341Z')
MLP=Path('/workspace/shanjinshuo/outputs/fno_kerr/priority1_mlp_t1200_seed27_20261024T014339Z')
WAVE=ROOT/'outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912'
FNO=WAVE/'train_t1200/best_model.pt'
REF=WAVE/'eval_t1200/prediction_t1200.npz'
KEYS=['global_relative_l2','mean_per_q_relative_l2','global_mse']


def mark(stage):
    print(json.dumps({'stage':stage,'utc':dt.datetime.now(dt.timezone.utc).isoformat()}),file=sys.stderr,flush=True)


atexit.register(mark,'python_atexit_before_torch')
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import scipy
from src.training.benchmark_track_a import checkpoint_contract,predict_frozen
from src.training.benchmark_data import Trajectories,restore_normalization
from scripts.evaluate_plan_b_resolution_generalization_2d import build_canonical_field_from_artifact,compute_raw_metrics,_relative_l2
from src.data_generation.orbit_math import spherical_to_cartesian
atexit.register(mark,'python_atexit_after_torch')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rawsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def save(name,obj):
    with (OUT/name).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def loadmodule(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def load_contracts():
    hist=read(WAVE/'eval_t1200/metrics_t1200.json');prov=hist['dataset_provenance']
    assert sha(DATA/'dataset.npz')==prov['dataset_sha256']=='39c7b1ce9624123a5f2680d76abe8a7fa18e135ec166fa70262efb8d563c60db'
    assert sha(DATA/'meta.json')==prov['metadata_sha256']=='fbe816012a245c4cadf132de8cae5f042f8f796677e89b91af33b1fb377702d8'
    assert sha(FNO)=='11ccbebf1d0fd68c57d0895685adeb7d1386da723242ee8271b18286ebbb7ff1'
    assert sha(REF)=='4482c19f58f57b5102e33942f6a50fb04ce7a31729727b041457922edef3c0c3'
    fno=torch.load(FNO,map_location='cpu',weights_only=True)
    assert fno['model_name']=='fno1d' and fno['train_resolution']==1200
    assert fno['training']['seed']==27 and fno['training']['target_transform']=='raw'
    summary=read(WAVE/'train_t1200/summary.json')
    assert summary['completed_epochs']==fno['completed_epochs']==500
    assert summary['best_epoch']==fno['epoch'] and summary['best_checkpoint_sha256']==sha(FNO)
    for rel,h in fno['source_provenance']['source_sha256'].items():assert sha(ROOT/rel)==h,rel
    best=read(MLP/'run/BEST.json');mp=MLP/'run/checkpoints'/best['file']
    assert sha(mp)==best['sha256']=='91503281bd80e5106ab7fb051b5956c88e269d49b20b5749681b66824f78b4c0'
    assert read(mp.with_suffix('.json'))==best
    mlp=torch.load(mp,map_location='cpu',weights_only=True)
    mc=read(MLP/'frozen-config.json')
    assert mlp['epoch']==best['epoch']==494 and mlp['config']==mc
    assert mc['architecture']==[2,576,576,576,576,3] and mc['seed']==27 and mc['split_seed']==10
    assert mc['target_transform']=='raw'
    assert read(MLP/'run/completion.json')['status']=='COMPLETED'
    for path,h in mc['source_sha256'].items():assert sha(path)==h,path
    assert mlp['normalization']==read(MLP/'run/normalization.json')==fno['normalization']
    sp=read(SPLINE/'frozen-protocol.json')
    for name,h in sp['code_sha256'].items():assert sha(SPLINE/name)==h
    assert sp['training_Q_count']==1400 and sp['data_seed']==10
    assert sha(ROOT/'scripts/evaluate_plan_b_resolution_generalization_2d.py')==sp['historical_metric_source_sha256']
    tm=read(TRAIN/'data-manifest.json')
    assert sha(TRAIN/'data-manifest.json')==mc['data_manifest_sha256']==sp['data_manifest_sha256']
    assert sha(TRAIN/tm['data_file'])==tm['sha256']
    assert tm['source_sha256']==fno['dataset_provenance']['dataset_sha256']
    with np.load(TRAIN/tm['data_file'],allow_pickle=False) as loaded:train={k:loaded[k].copy() for k in loaded.files}
    for k,v in train.items():
        assert v.dtype==np.float64 and list(v.shape)==tm['arrays'][k]['shape']
        assert rawsha(v)==tm['arrays'][k]['raw_array_sha256'] and np.isfinite(v).all()
    for split in ['train','val']:assert rawsha(train['x_'+split])==fno['dataset_provenance']['split_q_sha256'][split]
    assert not np.intersect1d(train['x_train'],train['x_val']).size
    meta=read(DATA/'meta.json');trainmeta=read(TRAIN/'source-meta.json')
    assert sha(TRAIN/'source-meta.json')==tm['source_metadata_sha256']==fno['dataset_provenance']['metadata_sha256']
    assert meta['task_spec']['fixed_params']==trainmeta['task_spec']['fixed_params']
    assert meta['integration']==trainmeta['integration']=={'n_steps':1200,'step_size':.005,'lambda_max':5.995}
    assert meta['vary_params_order']==trainmeta['vary_params_order']==['Q']
    for k in ['orbit_solver','orbit_solver_version']:
        assert meta['task_spec']['metadata'][k]==trainmeta['task_spec']['metadata'][k]
    assert meta['generation_status']['fail_count']==0 and meta['generation_status']['completed']
    field=build_canonical_field_from_artifact(DATA)
    q=field.canonical_q;truth=field.canonical_truth;grid=field.lambda_grid
    assert q.shape==(400,) and truth.shape==(400,1200,3) and truth.dtype==np.float64
    assert np.array_equal(grid,np.arange(1200,dtype=np.float64)*.005)
    assert np.array_equal(grid,train['lambda_grid'])
    assert field.canonical_to_source_index.tolist()==prov['canonical_to_source_index']
    assert field.source_records==prov['source_records']
    with np.load(DATA/'dataset.npz',allow_pickle=False) as a:
        for split in ['train','val','test']:assert rawsha(a['x_'+split])==prov['split_q_sha256'][split]
    with np.load(REF,allow_pickle=False) as a:
        reference=a['prediction_xyz'].copy()
        assert np.array_equal(q,a['Q'].reshape(-1)) and np.array_equal(grid,a['lambda_grid'])
    assert reference.shape==truth.shape and reference.dtype==np.float32 and np.isfinite(reference).all()
    fixed=meta['task_spec']['fixed_params']
    initial=spherical_to_cartesian(fixed['r0'],fixed['theta0'],fixed['phi0'])
    assert np.array_equal(truth[:,0,:],np.broadcast_to(initial,(400,3)))
    assert np.array_equal(train['y_train'][:,0,:],np.broadcast_to(initial,(1400,3)))
    # 这里只核对旧预测和新恢复真值的指标口径，不据此选择任何模型或插值规则。
    recomputed=compute_raw_metrics(reference,truth,q)
    for key in KEYS:assert np.isclose(recomputed[key],hist[key],rtol=1e-12,atol=1e-15),(key,recomputed[key],hist[key])
    return {'hist':hist,'field':field,'train':train,'reference':reference,'fno':fno,'mlp':mlp,'mp':mp,
            'sp':sp,'fixed':fixed,'historical_recomputed':recomputed,'trainmeta':trainmeta,'meta':meta}


def precision():
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'matmul_precision':torch.get_float32_matmul_precision(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}


def audit():
    c=load_contracts();field=c['field'];train=c['train'];q=field.canonical_q;tq=train['x_train'].reshape(-1)
    paths=list(ROOT.joinpath('src').rglob('*.py'))+list(ROOT.joinpath('scripts').glob('*.py'))
    paths += [Path(__file__),SPLINE/'spline_in_q_priority1.py',SPLINE/'frozen-protocol.json',SPLINE/'source-transport-note.json',
              MLP/'code/p1_mlp_runner.py',MLP/'frozen-config.json',MLP/'run/normalization.json',MLP/'run/BEST.json',
              MLP/'run/completion.json',c['mp'],FNO,REF,WAVE/'eval_t1200/metrics_t1200.json',WAVE/'train_t1200/summary.json',
              DATA/'dataset.npz',DATA/'meta.json',TRAIN/'data-manifest.json',TRAIN/'source-meta.json',TRAIN/'train1400_val300_t1200.npz']
    source={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    inside=(q>=tq.min())&(q<=tq.max())
    protocol={'status':'FROZEN_BEFORE_NEW_PREDICTIONS','frozen_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'sources':source,'Q_sha256':rawsha(q),'truth_sha256':rawsha(field.canonical_truth),'lambda_sha256':rawsha(field.lambda_grid),
              'Q_count':400,'shape':[400,1200,3],'Q_range':[float(q.min()),float(q.max())],
              'field_caveat':'Established Q400 evaluation field previously used in historical research; not an untouched held-out test',
              'training_count':1400,'training_Q_hull':[float(tq.min()),float(tq.max())],'training_Q_sha256':rawsha(train['x_train']),
              'training_split_seed':10,'neural_training_seed':27,'inside_count':int(inside.sum()),'outside_count':int((~inside).sum()),
              'outside_indices':np.flatnonzero(~inside).tolist(),'outside_Q':q[~inside].tolist(),
              'train_Q_exact_overlap_with_Q400':int(np.intersect1d(tq,q).size),
              'spline_protocol_reference':str(SPLINE/'frozen-protocol.json'),'spline_protocol_changed':False,
              'neural_batch_size':32,'last_batch':16,'canonical_order':'Original stable ascending Q and original split/source-row mapping',
              'normalization':'Restore each checkpoint; never fit on Q400','normalization_exactly_matches_between_models':True,
              'target_transform':'raw','fixed_physics_and_initial_conditions':c['fixed'],
              'coordinate_definition':'x=r*sin(theta)*cos(phi), y=r*sin(theta)*sin(phi), z=r*cos(theta); original spherical_to_cartesian',
              'lambda':'arange(1200)*0.005; original Mino parameter grid',
              'metrics':{'function':'Original compute_raw_metrics for all methods and subsets','epsilon':1e-12,'reduction':'float64',
                         'global_relative_l2':'norm(pred-truth)/(norm(truth)+epsilon)',
                         'mean_per_q_relative_l2':'mean_i(norm(pred_i-truth_i)/(norm(truth_i)+epsilon))',
                         'global_mse':'mean over Q,lambda,xyz of squared error','per_q_max_absolute_error':'max over lambda and xyz'},
              'FNO_historical_comparison':{'atol':1e-5,'rtol':1e-4,'full_reference_count':400,'historical_metrics_recomputed_match':True},
              'models':{'Spline':'Original frozen float64 Natural Cubic Spline and two-node secant tails',
                        'MLP':{'architecture':[2,576,576,576,576,3],'activation':'GELU','best_epoch':494},
                        'FNO':{'model_name':'fno1d','config':c['fno']['model_config'],'best_epoch':c['fno']['epoch']}},
              'runtime':{'hostname':os.uname().nodename,'python':sys.version,'torch':str(torch.__version__),
                         'numpy':np.__version__,'scipy':scipy.__version__,'cuda_runtime':torch.version.cuda},
              'source_provenance':'Actual source SHA256 snapshot; remote is not a complete Git checkout',
              'safe_stop_utc':'2026-10-24T07:07:00Z','new_training':False,'tuning':False,'formal_timing_benchmark':False}
    save('canonical-identity.json',{'Q':q.tolist(),'canonical_to_source_index':field.canonical_to_source_index.tolist(),'source_records':field.source_records})
    save('frozen-protocol.json',protocol)
    save('comparability-audit.json',{'status':'PASS','historical_metrics_recomputed':{k:c['historical_recomputed'][k] for k in KEYS},
        'data_archive_matches_historical_sha256':True,'exact_Q_grid_and_source_mapping':True,'same_fixed_physics':True,
        'same_initial_xyz':True,'correct_training_split_identity':True,'checkpoint_normalization_restored':True,
        'normalization_not_refitted':True,'all_frozen_sources_verified':True,'outside_Q':q[~inside].tolist()})
    print(json.dumps({'status':'AUDIT_PASS_PROTOCOL_FROZEN','inside':int(inside.sum()),'outside':int((~inside).sum()),'outside_Q':q[~inside].tolist()}),flush=True)


def unified(pred,truth,q,mask):
    if not mask.any():return {'status':'N/A','count':0}
    a=pred[mask].astype(np.float64);b=truth[mask].astype(np.float64)
    metrics=compute_raw_metrics(a,b,q[mask])
    metrics['count']=int(mask.sum());metrics['status']='PASS'
    metrics['squared_error_sum']=float(np.sum((a-b)**2,dtype=np.float64))
    return metrics


def evaluate():
    p=read(OUT/'frozen-protocol.json')
    for path,entry in p['sources'].items():assert sha(path)==entry['sha256'],path
    c=load_contracts();field=c['field'];q=field.canonical_q;grid=field.lambda_grid;truth=field.canonical_truth
    assert rawsha(q)==p['Q_sha256'] and rawsha(truth)==p['truth_sha256'] and rawsha(grid)==p['lambda_sha256']
    allmask=np.ones(400,dtype=bool);lo,hi=p['training_Q_hull'];inside=(q>=lo)&(q<=hi);outside=~inside
    spline=loadmodule('frozen_spline',SPLINE/'spline_in_q_priority1.py')
    mlpcode=loadmodule('frozen_mlp',MLP/'code/p1_mlp_runner.py')
    start=time.perf_counter();mark('before_spline_fit')
    fitted=spline.SplineInQ(c['train']['x_train'].reshape(-1),c['train']['y_train'])
    fit_seconds=time.perf_counter()-start
    sp,spinside,left,right=fitted.predict(q);assert np.array_equal(spinside,inside)
    reconstruction=fitted.predict(c['train']['x_train'].reshape(-1))[0]
    node_error=float(np.max(np.abs(reconstruction-c['train']['y_train'])))
    assert node_error<1e-10
    predictions={'Spline':sp};compute={'Spline':{'device':'cpu','fit_seconds_diagnostic_only':fit_seconds,'training_node_max_abs':node_error}}
    # 恢复 MLP 训练进程的精度标志，仅作用于本子进程；不修改环境。
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    m=mlpcode.make_model();m.load_state_dict(c['mlp']['model_state_dict'],strict=True)
    assert sum(v.numel() for v in m.parameters())==1000515
    ms=restore_normalization(c['mlp']['normalization']);m.eval().requires_grad_(False).to('cuda:0')
    mark('before_MLP_forward')
    with torch.inference_mode():predictions['MLP'],mlpc=predict_frozen(m,Trajectories(q,grid,truth),ms,'cuda:0')
    torch.cuda.synchronize();mark('after_MLP_forward')
    compute['MLP']={'device':'cuda:0','precision':precision(),'batch_size':32,'last_batch':16,
                    'allocator_peak_bytes':mlpc['peak_gpu_memory_bytes'],'formal_timing_benchmark':False}
    assert all(v.grad is None for v in m.parameters());del m
    # FNO 沿用历史独立 evaluate 的默认精度标志、原始 predict_frozen 和完整 canonical batch。
    torch.backends.cudnn.deterministic=False
    f,fs=checkpoint_contract(c['fno']);f.eval().requires_grad_(False).to('cuda:0')
    mark('before_FNO_forward')
    with torch.inference_mode():predictions['FNO'],fnoc=predict_frozen(f,Trajectories(q,grid,truth),fs,'cuda:0')
    torch.cuda.synchronize();mark('after_FNO_forward')
    compute['FNO']={'device':'cuda:0','precision':precision(),'batch_size':32,'last_batch':16,
                   'allocator_peak_bytes':fnoc['peak_gpu_memory_bytes'],'formal_timing_benchmark':False}
    assert all(v.grad is None for v in f.parameters())
    for name,a in predictions.items():
        assert a.shape==(400,1200,3) and np.isfinite(a).all()
        with (OUT/('prediction_'+name.lower()+'.npz')).open('xb') as stream:
            np.savez_compressed(stream,prediction_xyz=a,Q=q,lambda_grid=grid)
    with (OUT/'canonical-truth.npz').open('xb') as stream:np.savez_compressed(stream,truth_xyz=truth,Q=q,lambda_grid=grid)
    actual=predictions['FNO'].astype(np.float64);reference=c['reference'].astype(np.float64);delta=actual-reference
    close=np.abs(delta)<=1e-5+1e-4*np.abs(reference)
    history={'bitwise_equal':bool(np.array_equal(predictions['FNO'],c['reference'])),'max_abs_diff':float(np.max(np.abs(delta))),
             'relative_l2':_relative_l2(actual,reference),'MSE':float(np.mean(delta**2)),
             'allclose':bool(close.all()),'outside_tolerance_scalars':int((~close).sum()),'scalars':int(delta.size),
             'atol':1e-5,'rtol':1e-4,'canonical_batch_sizes':[32]*12+[16]}
    save('fno-history-crosscheck.json',history)
    if not history['allclose']:raise RuntimeError('FNO historical prediction mismatch; stop comparison without protocol adaptation')
    metrics={};paired=[];ratios={};tails={}
    for name,a in predictions.items():
        metrics[name]={key:unified(a,truth,q,mask) for key,mask in [('all',allmask),('inside',inside),('outside',outside)]}
        error=a.astype(np.float64)-truth
        per=np.array([_relative_l2(a[i].astype(np.float64),truth[i]) for i in range(400)])
        maximum=np.max(np.abs(error),axis=(1,2))
        tails[name]={'outside_squared_error_fraction':float(np.sum(error[outside]**2)/np.sum(error**2)),
                     'outside_sum_per_Q_relative_error_fraction':float(per[outside].sum()/per.sum())}
        for i in range(400):
            if len(paired)<=i:paired.append({'canonical_index':i,'Q':float(q[i]),'in_training_hull':bool(inside[i]),'models':{}})
            paired[i]['models'][name]={'relative_l2':float(per[i]),'max_abs_error':float(maximum[i])}
    for subset in ['all','inside','outside']:
        ratios[subset]={}
        for numerator,denominator in [('MLP','Spline'),('FNO','Spline'),('FNO','MLP')]:
            ratios[subset][numerator+'_over_'+denominator]={k:metrics[numerator][subset][k]/metrics[denominator][subset][k] for k in KEYS}
    legacy={key:{'historical':c['hist'][key],'current':metrics['FNO']['all'][key],
                 'absolute_difference':abs(c['hist'][key]-metrics['FNO']['all'][key])} for key in KEYS}
    save('per-q-paired-errors.json',paired);save('metrics.json',metrics);save('error-ratios.json',ratios)
    save('train-sort-mapping.json',{'sorted_to_original_row':fitted.order.tolist(),'sorted_train_Q':fitted.q.tolist()})
    save('runtime.json',{'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'cuda':torch.version.cuda,
                        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'models':compute,
                        'memory_caveat':'Allocator counters only; HAMI physical memory reports are not quota evidence'})
    for path,entry in p['sources'].items():assert sha(path)==entry['sha256'],path
    result={'status':'COMPLETED','completed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'Q_count':400,'T':1200,'all_shapes_and_finiteness':'PASS','Q_alignment':'PASS','all_frozen_inputs_unchanged':True,
            'training_Q_hull':[lo,hi],'inside_count':int(inside.sum()),'outside_count':int(outside.sum()),
            'outside_Q':q[outside].tolist(),'metrics':metrics,'ratios':ratios,'tail_contribution':tails,
            'FNO_history':history,'historical_metric_comparison':legacy,'training_node_reconstruction_max_abs':node_error,
            'field_caveat':p['field_caveat'],'new_training':False,'hyperparameter_selection_on_Q400':False,
            'formal_timing_benchmark':False,'scope':'One frozen evaluation per model; not a multi-seed or untouched-test result'}
    save('result.json',result);mark('all_results_saved')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['audit','evaluate'],required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    assert os.uname().nodename=='job-u15c7468-d8ms2','Container changed; recheck task time'
    assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat('2026-10-24T07:07:00+00:00')
    mark('start_'+args.phase)
    if args.phase=='audit':audit()
    else:evaluate()
    mark('script_end')
