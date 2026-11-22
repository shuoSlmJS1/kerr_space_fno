"""Bounded P1 workflow checks, never a formal benchmark or training entry point."""
import argparse
import atexit
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['cpu', 'gpu'], required=True)
p.add_argument('--output-dir', required=True)
a = p.parse_args()
R = Path(a.output_dir)
ROOT = Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
DATA = Path('/workspace/shanjinshuo/datasets/fno_kerr/reviewer_p1_preflight_20261010')


def mark(stage):
    print(json.dumps({'stage': stage, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}), file=sys.stderr, flush=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


atexit.register(mark, 'python_atexit_before_torch')
mark('before_import')
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
import torch
from torch import nn
from src.training.benchmark_track_a import checkpoint_contract
from src.training.benchmark_data import recover_xyz
from src.training.fno1d.input_builder_1d import build_fno1d_input_array
from src.training.fno2d.normalization_2d import normalize_input_field, normalize_output_field
from scripts.evaluate_plan_b_resolution_generalization_2d import compute_raw_metrics
from src.data_generation.orbit_solver_second_order import simulate_one_orbit_second_order
from src.data_generation.orbit_types import KerrParams, InitialState
atexit.register(mark, 'python_atexit_after_torch')
torch.set_num_threads(2)
torch.manual_seed(27)
np.random.seed(27)
mark('after_import')
m = json.loads((DATA/'asset-manifest.json').read_text())
assert sha(DATA/m['subset']['file']) == m['subset']['sha256']
for row in m['source_files']:
    assert sha(ROOT/row['relative_path']) == row['sha256'], 'Source mismatch: '+row['relative_path']
with np.load(DATA/m['subset']['file'], allow_pickle=False) as f:
    assert set(f.files) == {'x_train','y_train','x_val','y_val','lambda_grid'}
    xtr,ytr,xv,yv,grid = [f[k].copy() for k in ['x_train','y_train','x_val','y_val','lambda_grid']]
assert ytr.shape == (16,1200,3) and yv.shape == (4,1200,3)
assert ytr.dtype == yv.dtype == np.float64
assert np.isfinite(ytr).all() and np.isfinite(yv).all()
ckpt = ROOT/'outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t1200/best_model.pt'
payload = torch.load(ckpt,map_location='cpu',weights_only=True)
model,stats = checkpoint_contract(payload)
device = 'cpu' if a.mode=='cpu' else 'cuda:0'
model.eval().requires_grad_(False).to(device)
report = {'status':'verified','scope':'SMOKE_ONLY_NOT_FORMAL_BENCHMARK','mode':a.mode,
          'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,
          'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
          'gpu':torch.cuda.get_device_name(0) if a.mode=='gpu' else None,
          'precision':{'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,
                       'matmul_precision':torch.get_float32_matmul_precision(),
                       'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                       'cudnn_benchmark':torch.backends.cudnn.benchmark,
                       'cudnn_deterministic':torch.backends.cudnn.deterministic},
          'seed':27,'threads':2,'heldout_test_truth_available':False,
          'checkpoint_sha256':sha(ckpt),'manifest_sha256':sha(DATA/'asset-manifest.json')}


def timed(fn, sync=lambda:None):
    # 一次预热、三次测量仅检验计时流程，不作为正式速度结论。
    fn();sync()
    values=[]
    for _ in range(3):
        sync();start=time.perf_counter();value=fn();sync()
        values.append(time.perf_counter()-start)
    return {'warmup':1,'repetitions':3,'seconds':values,'median_seconds':float(np.median(values))},value


if a.mode=='cpu':
    mark('before_spline')
    order=np.argsort(xtr[:,0],kind='stable')
    q=xtr[order,0];y=ytr[order]
    assert np.all(np.diff(q)>0)
    spline=CubicSpline(q,y,axis=0,bc_type='natural',extrapolate=False)
    def safe_predict(query):
        if np.any(query<q[0]) or np.any(query>q[-1]):
            raise ValueError('Q outside training hull; extrapolation is not approved')
        return spline(query)
    predicted=safe_predict(xv[:,0])
    assert predicted.shape==yv.shape and np.isfinite(predicted).all()
    assert np.allclose(safe_predict(q),y,atol=1e-10,rtol=1e-12)
    rejected=False
    try:safe_predict(np.array([q[0]-0.01]))
    except ValueError:rejected=True
    assert rejected
    # 合成固定样例验证 global 与逐 Q 均值不同，且与历史函数一致。
    truth=np.array([[[1.,2.,3.]],[[10.,20.,30.]]])
    guess=truth+np.array([[[1.,0.,0.]],[[1.,0.,0.]]])
    measured=compute_raw_metrics(guess,truth,np.array([1.,2.]))
    g=np.linalg.norm(guess-truth)/(np.linalg.norm(truth)+1e-12)
    per=np.mean([np.linalg.norm(guess[i]-truth[i])/(np.linalg.norm(truth[i])+1e-12) for i in range(2)])
    assert abs(measured['global_relative_l2']-g)<1e-15 and abs(measured['mean_per_q_relative_l2']-per)<1e-15
    report['spline']={'status':'verified','axis':'Q','train_Q_count':16,'validation_Q_count':4,
                      'shape':list(predicted.shape),'finite':True,'knot_reconstruction':True,
                      'out_of_hull_rejected':rejected,'bc_type':'natural',
                      'validation_smoke_metrics':compute_raw_metrics(predicted,yv,xv),
                      'metrics_used_to_select_parameters':False,
                      'synthetic_metric_test':{'global':float(g),'mean_per_Q':float(per)}}
    mark('after_spline')
    cfg=m['source_dataset']['metadata']['task_spec']['fixed_params']
    pars=KerrParams(**{k:cfg[k] for k in ['M','a','E','Lz']})
    init=InitialState(**{k:cfg[k] for k in ['r0','theta0','phi0','sign_r','sign_th']})
    mark('before_solver_timing')
    timing,sol=timed(lambda:simulate_one_orbit_second_order(pars,init,float(xtr[0,0]),1200,.005))
    assert sol['xyz'].shape==(1200,3) and np.isfinite(sol['xyz']).all()
    assert np.array_equal(sol['lambda_grid'],grid)
    delta=float(np.max(np.abs(sol['xyz']-ytr[0])))
    report['solver']={'status':'verified','Q':float(xtr[0,0]),'T':1200,'step_size':.005,
                      'cpu_wall_clock':timing,'scope':'initialization + RK4 + xyz + diagnostics; excludes imports and disk IO',
                      'training_truth_max_abs_difference':delta,'diagnostics':asdict(sol['diagnostics'])}
    assert np.allclose(sol['xyz'],ytr[0],atol=1e-10,rtol=1e-10),'Solver/training truth mismatch'
    mark('after_solver_timing')
else:
    mark('before_mlp_forward_backward')
    mlp=nn.Sequential(nn.Linear(2,16),nn.GELU(),nn.Linear(16,16),nn.GELU(),nn.Linear(16,3)).to(device)
    initial={k:v.detach().clone() for k,v in mlp.state_dict().items()}
    xx=normalize_input_field(build_fno1d_input_array(xtr[:2],grid[:32])[None],stats)[0]
    yy=normalize_output_field(ytr[:2,:32][None],stats)[0]
    pred=mlp(torch.from_numpy(xx).to(device))
    loss=(pred-torch.from_numpy(yy).to(device)).square().mean()
    loss.backward();torch.cuda.synchronize()
    assert pred.shape==(2,32,3) and torch.isfinite(pred).all() and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mlp.parameters())
    assert all(torch.equal(initial[k],v) for k,v in mlp.state_dict().items())
    proposal=nn.Sequential(nn.Linear(2,576),nn.GELU(),*[layer for _ in range(3) for layer in [nn.Linear(576,576),nn.GELU()]],nn.Linear(576,3))
    report['mlp']={'status':'verified','device':str(pred.device),'shape':list(pred.shape),
                   'loss_finite':True,'gradients_finite':True,'optimizer_steps':0,'weights_unchanged':True,
                   'smoke_parameters':sum(p.numel() for p in mlp.parameters()),
                   'proposed_formal_parameters':sum(p.numel() for p in proposal.parameters()),
                   'proposal_count_only_not_trained':True}
    mark('after_mlp_forward_backward')

mark('before_neural_timing')
raw=build_fno1d_input_array(xtr[:1],grid)
normalized=normalize_input_field(raw[None],stats)[0]
resident=torch.from_numpy(normalized).to(device)
sync=torch.cuda.synchronize if a.mode=='gpu' else lambda:None
with torch.inference_mode():
    core,output=timed(lambda:model(resident),sync)
    def end_to_end():
        x=normalize_input_field(build_fno1d_input_array(xtr[:1],grid)[None],stats)[0]
        y=model(torch.from_numpy(x).to(device))
        return recover_xyz(y.cpu().numpy(),stats)
    whole,raw_output=timed(end_to_end,sync)
assert list(output.shape)==[1,1200,3] and torch.isfinite(output).all() and np.isfinite(raw_output).all()
report['neural_timing']={'status':'verified','model':'historical FNO1D BEST','batch':1,'Q':float(xtr[0,0]),
                          'T':1200,'model_only':core,'end_to_end':whole,
                          'same_Q_grid_as_solver':True,'accuracy_evaluation':False,
                          'hardware_speedup_claim':False,'formal_benchmark':False,
                          'forward_count':8,'peak_memory_interpreted':False}
assert sha(ckpt)==report['checkpoint_sha256']
mark('after_neural_timing')
with (R/(a.mode+'-smoke.json')).open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report),flush=True)
mark('result_saved')
mark('script_end')
