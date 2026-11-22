"""Execute one pre-frozen CPU-only T1200 train1400/validation300 experiment."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any
import numpy as np
import scipy
from spline_in_q_priority1 import SplineInQ,unit_tests

R=Path(__file__).resolve().parent
ROOT=Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
freeze=json.loads((R/'frozen-protocol.json').read_text())
D=Path(freeze['data_directory'])


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name,data):
    with (R/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


start=datetime.datetime.now(datetime.timezone.utc)
assert start<datetime.datetime.fromisoformat(freeze['safety_deadline_utc'])
for n,h in freeze['code_sha256'].items():assert sha(R/n)==h,'Frozen source changed'
manifest=json.loads((D/'data-manifest.json').read_text())
assert sha(D/'data-manifest.json')==freeze['data_manifest_sha256']
assert sha(D/manifest['data_file'])==manifest['sha256']
assert sha(D/'source-meta.json')==manifest['source_metadata_sha256']
with np.load(D/manifest['data_file'],allow_pickle=False) as f:
    assert set(f.files)=={'x_train','y_train','x_val','y_val','lambda_grid'}
    arrays={k:f[k].copy() for k in f.files}
for n,arr in arrays.items():
    info=manifest['arrays'][n]
    assert arr.dtype==np.float64 and list(arr.shape)==info['shape']
    assert hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()==info['raw_array_sha256']
    assert np.isfinite(arr).all(),'Nonfinite data: '+n
q=arrays['x_train'][:,0];y=arrays['y_train'];v=arrays['x_val'][:,0];truth=arrays['y_val'];grid=arrays['lambda_grid']
assert len(q)==1400 and len(v)==300 and len(np.unique(q))==1400 and len(np.unique(v))==300
assert not np.intersect1d(q,v).size
assert np.array_equal(grid,np.arange(1200,dtype=np.float64)*.005)
assert json.loads((D/'source-meta.json').read_text())['task_spec']['seed']==10

# 仅编译历史指标函数的原始 AST，不导入评估入口及其 torch/GPU 依赖。
metric_path=ROOT/'scripts/evaluate_plan_b_resolution_generalization_2d.py'
assert sha(metric_path)==freeze['historical_metric_source_sha256']
tree=ast.parse(metric_path.read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ['_relative_l2','compute_raw_metrics']]
assert len(nodes)==2
namespace={'np':np,'Any':Any,'EPSILON':1e-12}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(metric_path),'exec'),namespace)
metrics=namespace['compute_raw_metrics']
tests=unit_tests()
synthetic=np.ones((2,4,3));changed=synthetic*2
assert abs(metrics(changed,synthetic,np.array([1.,2.]))['global_relative_l2']-np.linalg.norm(synthetic)/(np.linalg.norm(synthetic)+1e-12))<1e-15
write('unit-tests.json',tests)
write('data-validation.json',{'status':'PASS','train_val_disjoint':True,'raw_array_hashes_verified':True,
                            'original_row_order':True,'finite':True,'lambda_grid_exact':True,'test_data_present':False})
print(json.dumps({'stage':'before_fit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
t=time.perf_counter();model=SplineInQ(q,y);fit_seconds=time.perf_counter()-t
assert np.array_equal(model.q,q[model.order]) and np.all(np.diff(model.q)>0)
t=time.perf_counter();prediction,inside,left,right=model.predict(v);prediction_seconds=time.perf_counter()-t
assert prediction.shape==(300,1200,3) and prediction.dtype==np.float64 and np.isfinite(prediction).all()
outside=left|right
assert inside.sum()+outside.sum()==300
write('train-sort-mapping.json',{'sorted_to_original_row':model.order.tolist(),'Q_sorted':model.q.tolist()})
groups={}
for name,mask in [('all',np.ones(300,dtype=bool)),('inside',inside),('outside',outside)]:
    groups[name]={'status':'N/A','count':0} if not mask.any() else {'status':'evaluated','count':int(mask.sum()),**metrics(prediction[mask],truth[mask],v[mask])}
errors=np.array([np.linalg.norm(prediction[i]-truth[i])/(np.linalg.norm(truth[i])+1e-12) for i in range(300)])
write('per-validation-Q.json',[{'original_val_row':i,'Q':float(v[i]),'region':'left_linear_tail' if left[i] else 'right_linear_tail' if right[i] else 'inside_cubic',
                              'relative_l2':float(errors[i]),'max_abs_error':float(np.max(np.abs(prediction[i]-truth[i]))),
                              'MSE':float(np.mean((prediction[i]-truth[i])**2))} for i in range(300)])
# 节点重建只检验插值实现，不构成泛化性能证据。
reconstructed=model.predict(q)[0]
reconstruction=float(np.max(np.abs(reconstructed-y)))
assert np.allclose(reconstructed,y,atol=1e-10,rtol=1e-12)
boundary={}
for name,endpoint,slope in [('left',model.q[0],model.left_slope),('right',model.q[-1],model.right_slope)]:
    knot=model.xyz[0] if name=='left' else model.xyz[-1]
    assert np.allclose(model.predict(np.array([endpoint]))[0][0],knot,atol=1e-10,rtol=1e-12)
    mask=left if name=='left' else right
    if mask.any():
        expected=knot+(v[mask]-endpoint)[:,None,None]*slope
        assert np.array_equal(prediction[mask],expected)
    boundary[name]={'count':int(mask.sum()),'secant_formula_verified':True,'endpoint_continuity_verified':True}
with (R/'validation-predictions.npz').open('xb') as f:
    np.savez_compressed(f,Q=v[:,None],lambda_grid=grid,prediction_xyz=prediction,inside_hull=inside,
                        original_val_row=np.arange(300,dtype=np.int64))
result={'status':'COMPLETED_VALIDATION','started_utc':start.isoformat(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256':sha(R/'frozen-protocol.json'),'source_code_sha256':freeze['code_sha256'],
        'data_sha256':manifest['sha256'],'device':'CPU','python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'cpu_platform':platform.processor(),'train':{'count':1400,'Q_min':float(model.q[0]),'Q_max':float(model.q[-1]),
            'max_adjacent_Q_gap':float(np.max(np.diff(model.q)))},
        'validation':{'count':300,'Q_min':float(v.min()),'Q_max':float(v.max()),'inside_count':int(inside.sum()),'outside_count':int(outside.sum())},
        'metrics':groups,'timing':{'fit_seconds':fit_seconds,'prediction_seconds':prediction_seconds,
                                 'scope':'single CPU wall-clock; fit includes stable sorting and secant slopes; prediction includes region masks and output allocation; excludes disk IO and metrics; no performance benchmark claim'},
        'checks':{'shapes':True,'all_data_and_outputs_finite':True,'stable_sort_and_unique_Q':True,'original_Q_hashes':True,
                  'lambda_grid_unchanged':True,'train_val_disjoint':True,'boundaries':boundary,
                  'training_node_max_abs_error':reconstruction,'training_node_check_is_not_generalization_evidence':True},
        'no_test_or_Q400_evaluation':True,'protocol_changed_after_validation':False,
        'outside_rows':[{'val_row':int(i),'Q':float(v[i]),'relative_l2':float(errors[i]),
                         'distance_to_train_hull':float(model.q[0]-v[i] if left[i] else v[i]-model.q[-1])} for i in np.flatnonzero(outside)],
        'prediction_sha256':sha(R/'validation-predictions.npz')}
write('validation-result.json',result)
print(json.dumps(result),flush=True)
