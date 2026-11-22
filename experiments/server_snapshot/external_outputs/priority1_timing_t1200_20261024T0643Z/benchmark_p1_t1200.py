"""Frozen compute-cost benchmark; no training and no accuracy protocol changes."""
import argparse
import atexit
import csv
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
COMP=Path('/workspace/shanjinshuo/outputs/fno_kerr/priority1_q400_comparison_t1200')
SPL=Path('/workspace/shanjinshuo/outputs/fno_kerr/priority1_spline_t1200_val_20261011T0341Z')
MLP=Path('/workspace/shanjinshuo/outputs/fno_kerr/priority1_mlp_t1200_seed27_20261024T014339Z')
TRAIN=Path('/workspace/shanjinshuo/datasets/fno_kerr/priority1_spline_t1200_val_20261011T0341Z')
FNO=ROOT/'outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t1200/best_model.pt'
DEADLINE=dt.datetime.fromisoformat('2026-10-24T07:07:00+00:00')


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def mark(stage):print(json.dumps({'stage':stage,'utc':utc()}),file=sys.stderr,flush=True)
atexit.register(mark,'python_atexit_before_torch')
sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
import torch
from src.training.benchmark_track_a import checkpoint_contract
from src.training.benchmark_data import restore_normalization,recover_xyz
from src.training.fno1d.input_builder_1d import build_fno1d_input_array
from src.training.fno2d.normalization_2d import normalize_input_field
from src.data_generation.orbit_solver_second_order import simulate_one_orbit_second_order
from src.data_generation.orbit_types import KerrParams,InitialState
atexit.register(mark,'python_atexit_after_torch')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def rawsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def write(root,name,v):
    with (root/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())


def check_time():
    if dt.datetime.now(dt.timezone.utc)>DEADLINE-dt.timedelta(minutes=2):raise TimeoutError('Safety deadline approaching; preserve partial results')


def sync(device):
    if device=='cuda:0':torch.cuda.synchronize()


def stats(values,n):
    a=np.asarray(values,dtype=np.float64)
    return {'repetitions':len(values),'median_seconds':float(np.median(a)),'mean_seconds':float(a.mean()),
            'standard_deviation_seconds':float(a.std(ddof=1)) if len(a)>1 else None,
            'p95_seconds':float(np.percentile(a,95)),'min_seconds':float(a.min()),'max_seconds':float(a.max()),
            'coefficient_of_variation':float(a.std(ddof=1)/a.mean()) if len(a)>1 else None,
            'trajectories_per_second_from_median':float(n/np.median(a)) if n else None}


def main(out,phase):
    assert os.uname().nodename=='job-u15c7468-d8ms2','Revalidate task time for a changed container'
    check_time();out=Path(out)
    assert out.parent==Path('/workspace/shanjinshuo/outputs/fno_kerr') and not out.is_symlink()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    cp=read(COMP/'frozen-protocol.json');best=read(MLP/'run/BEST.json');mp=MLP/'run/checkpoints'/best['file']
    if phase=='freeze':
        if out.exists():assert {p.name for p in out.iterdir()}=={'benchmark_p1_t1200.py'}
        else:out.mkdir()
        assert read(COMP/'completion.json')['numerical_verdict']=='PASS'
        # 沿用上轮已核验的来源，新增本次计时实现及历史训练时间记录。
        source=dict(cp['sources'])
        for p in [Path(__file__),COMP/'result.json',COMP/'canonical-truth.npz',MLP/'run/completion.json',
                  MLP/'run/normalization.json',FNO.parent/'summary.json',*list((MLP/'run').glob('session_*.json'))]:
            source[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
        for path,item in source.items():assert sha(path)==item['sha256']
        affinity=sorted(os.sched_getaffinity(0));chosen=[];physical=set()
        mapping=subprocess.check_output(['lscpu','-p=CPU,CORE,SOCKET'],text=True)
        for line in mapping.splitlines():
            if line.startswith('#'):continue
            cpu,core,socket=map(int,line.split(','));key=(core,socket)
            if cpu in affinity and key not in physical:chosen.append(cpu);physical.add(key)
            if len(chosen)==2:break
        assert len(chosen)==2
        protocol={'status':'FROZEN_BEFORE_TIMING','utc':utc(),'sources':source,'cpu_affinity':chosen,
          'cpu_threads':2,'torch_interop_threads':1,'blas_threads':2,'cpu_solver_parallelism':'Serial Python loop, one trajectory at a time',
          'single_Q_canonical_index':200,'batch32_indices':list(range(32)),'full_Q400_order':'canonical ascending',
          'warmups':{'single':5,'batch32':5,'Q400':1,'preparation':1},
          'repeats':{'single':30,'batch32':30,'Q400':5,'preparation':3},
          'cases':['RK4_CPU','Spline_CPU','MLP_CPU','FNO_CPU','MLP_GPU','FNO_GPU'],
          'case_order':'As listed; single process, no concurrent model timing; all raw samples retained, no outlier removal',
          'query_groups':{'single':'canonical index 200; not a latency survey over Q','batch32':'canonical indices 0..31','Q400':'all 400; neural chunks 32, final16; Spline vectorized400; RK4 serial400'},
          'scopes':{'model_only':'Prepared normalized input resident on device -> forward -> inverse normalization on device -> raw xyz tensor; excludes host transfers and input construction',
                    'end_to_end':'In-memory physical Q/grid -> input construction/normalization -> transfer/forward -> CPU raw xyz; excludes disk, import, initialization, metrics',
                    'RK4':'Historical simulate_one_orbit_second_order including initialization, RK4, xyz, potentials, constraints, turning counts and safety checks; not bare RK4 arithmetic',
                    'Spline':'Original frozen predict incl boundary checks; never refit inside prediction timing; E2E also constructs/copies query array'},
          'synchronization':'CUDA synchronize immediately before and after each measured region; end-to-end includes D2H per chunk; no disk IO inside region',
          'validation':'Shape, finite, Q/grid identity and fresh compute counter outside timer for every repetition; no accuracy reevaluation',
          'preparation_scope':'Python imports excluded. Fit uses already loaded 1400 train trajectories. Load reads artifact, instantiates model and transfers to target device. NFS cache not flushed; first load is not claimed cold.',
          'spline_persistence':'New uncompressed NPZ of exact fitted float64 coefficients, knots, order and two endpoint/slopes; reconstruct original predict without refitting',
          'training_time_policy':'Only existing trustworthy records, hardware and timing scope identified separately; never retrain',
          'safe_stop_utc':DEADLINE.isoformat(),'task_expiry_source':'User creation at 2026-10-24 09:27 UTC+8 plus six hours; scheduler not independently verified',
          'field_caveat':cp['field_caveat'],'neural_dtype':'float32','spline_solver_dtype':'float64','formal_accuracy_recomputed':False,
          'limitations':['Shared CPU/GPU; affinity is not exclusive reservation','Only five complete-field repetitions; p95 is descriptive, not stable tail inference','CPU/GPU ratios are deployment comparisons, not pure algorithm acceleration','No optimized/vectorized/compiled RK4 implementation evaluated']}
        write(out,'timing-protocol.json',protocol);print(json.dumps({'status':'PROTOCOL_FROZEN','output':str(out),'cpu_affinity':chosen}),flush=True);return

    protocol=read(out/'timing-protocol.json')
    for path,item in protocol['sources'].items():assert sha(path)==item['sha256'],path
    os.sched_setaffinity(0,protocol['cpu_affinity'])
    with np.load(COMP/'canonical-truth.npz',allow_pickle=False) as z:q=z['Q'].copy();grid=z['lambda_grid'].copy()
    assert rawsha(q)==cp['Q_sha256'] and rawsha(grid)==cp['lambda_sha256'] and q.shape==(400,) and grid.shape==(1200,)
    with np.load(TRAIN/'train1400_val300_t1200.npz',allow_pickle=False) as z:tq=z['x_train'].reshape(-1).copy();ty=z['y_train'].copy()
    spline_module=module('frozen_spline',SPL/'spline_in_q_priority1.py');mlp_module=module('frozen_mlp',MLP/'code/p1_mlp_runner.py')
    fixed=cp['fixed_physics_and_initial_conditions']
    params=KerrParams(**{k:fixed[k] for k in ['M','a','E','Lz']})
    init=InitialState(**{k:fixed[k] for k in ['r0','theta0','phi0','sign_r','sign_th']})
    rows=[];timings={};preparation={};resources={};calls={};validation={};start_time=utc()
    cpu_stat=Path('/sys/fs/cgroup/cpu.stat')
    cpu_before=cpu_stat.read_text() if cpu_stat.exists() else None
    rawfile=(out/'timing-raw.csv').open('x',newline='');writer=csv.DictWriter(rawfile,fieldnames=['case','scope','group','repeat','trajectory_count','seconds','utc'])
    writer.writeheader();rawfile.flush()

    def sample(case,scope,group,fn,n,repeats,warmups,device,verify):
        values=[]
        for i in range(warmups):
            check_time();v=fn();sync(device);verify(v)
        for i in range(repeats):
            check_time();sync(device);start=time.perf_counter_ns();v=fn();sync(device);elapsed=(time.perf_counter_ns()-start)/1e9
            verify(v)
            row={'case':case,'scope':scope,'group':group,'repeat':i,'trajectory_count':n,'seconds':elapsed,'utc':utc()}
            writer.writerow(row);rawfile.flush();rows.append(row);values.append(elapsed)
        return stats(values,n),v

    def preparation_measure(case,scope,fn,device='cpu'):
        start=time.perf_counter();value=fn();sync(device);first=time.perf_counter()-start;del value;gc.collect()
        measured,value=sample(case,scope,'preparation',fn,0,3,0,device,lambda v:None)
        return {'first_observed_seconds_not_cold_guaranteed':first,'warm_repeated':measured},value

    mark('preparation')
    preparation['Spline_fit'],spline=preparation_measure('Spline_CPU','fit',lambda:spline_module.SplineInQ(tq,ty))
    state=out/'spline-coefficients.npz'
    with state.open('xb') as f:
        np.savez(f,q=spline.q,c=spline.spline.c,xyz=spline.xyz[[0,-1]],left_slope=spline.left_slope,right_slope=spline.right_slope,order=spline.order,lambda_grid=grid)
        f.flush();os.fsync(f.fileno())
    def load_spline():
        with np.load(state,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        s=spline_module.SplineInQ.__new__(spline_module.SplineInQ)
        for k in ['q','xyz','left_slope','right_slope','order']:setattr(s,k,a[k])
        s.spline=CubicSpline.construct_fast(a['c'],a['q'],extrapolate=False,axis=0)
        return s
    preparation['Spline_load'],loaded=preparation_measure('Spline_CPU','load_fitted_coefficients',load_spline)
    assert np.array_equal(spline.predict(q)[0],loaded.predict(q)[0]);spline=loaded
    preparation['Spline_disk']={'fitted_artifact_bytes':state.stat().st_size,'sha256':sha(state),'training_truth_uncompressed_bytes':ty.nbytes,'training_archive_bytes':(TRAIN/'train1400_val300_t1200.npz').stat().st_size}
    resources['Spline_CPU']={'resident_array_bytes':sum(a.nbytes for a in [spline.q,spline.xyz,spline.left_slope,spline.right_slope,spline.order,spline.spline.c]),'dtype':'float64'}
    training_summary=read(FNO.parent/'summary.json');session=read(next((MLP/'run').glob('session_*.json')));completion=read(MLP/'run/completion.json')
    preparation['historical_training']={'FNO':{'seconds':training_summary['training_wall_seconds'],'gpu':training_summary['runtime']['gpu_name'],
      'source':str(FNO.parent/'summary.json'),'scope':'Recorded training_wall_seconds around historical epoch loop; includes validation and in-loop checkpoint handling; excludes initial data/model setup'},
      'MLP':{'seconds':(dt.datetime.fromisoformat(completion['utc'])-dt.datetime.fromisoformat(session['start_utc'])).total_seconds(),'gpu':session['gpu'],
      'source':[str(MLP/'run/completion.json'),str(next((MLP/'run').glob('session_*.json')))],'scope':'Observed session start to completion after initial data/model setup; includes validation and per-epoch NFS checkpoints'}}
    del ty;gc.collect()

    for case in protocol['cases']:
        mark('start_'+case);device='cuda:0' if case.endswith('GPU') else 'cpu';model=None;calls[case]=0
        if case.startswith(('MLP','FNO')):
            name=case.split('_')[0];checkpoint=mp if name=='MLP' else FNO
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=name=='MLP'
            def load_neural():
                payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if name=='MLP':
                    m=mlp_module.make_model();m.load_state_dict(payload['model_state_dict'],strict=True);st=restore_normalization(payload['normalization'])
                else:m,st=checkpoint_contract(payload)
                return m.eval().requires_grad_(False).to(device),st
            preparation[case+'_load'],pair=preparation_measure(case,'load_checkpoint_and_initialize',load_neural,device)
            model,normalization=pair;del pair
            resources[case]={'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'checkpoint_bytes':checkpoint.stat().st_size,
                'checkpoint_sha256':sha(checkpoint),'dtype':str(next(model.parameters()).dtype),'device':str(next(model.parameters()).device),
                'cpu_threads':torch.get_num_threads(),'matmul_precision':torch.get_float32_matmul_precision(),
                'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                'cudnn_deterministic':torch.backends.cudnn.deterministic}
            mean=torch.tensor(normalization.y_mean,dtype=torch.float32,device=device)
            std=torch.tensor(normalization.y_std,dtype=torch.float32,device=device)
            if device=='cuda:0':torch.cuda.reset_peak_memory_stats()
        groups=['single','Q400'] if case=='RK4_CPU' else ['single','batch32','Q400']
        timings[case]={};validation[case]={}
        for group in groups:
            indices=np.array([200]) if group=='single' else np.arange(32 if group=='batch32' else 400)
            query=q[indices].copy();n=len(query);scopes=['end_to_end'] if case=='RK4_CPU' else ['model_only','end_to_end']
            if model is not None:
                resident=torch.from_numpy(normalize_input_field(build_fno1d_input_array(query[:,None],grid)[None],normalization)[0]).to(device)
            timings[case][group]={}
            for scope in scopes:
                def invoke():
                    if case=='RK4_CPU':
                        outputs=[]
                        for value in query:
                            sol=simulate_one_orbit_second_order(params,init,float(value),1200,.005)
                            outputs.append(sol['xyz']);calls[case]+=1
                            # 求解器生成的网格属于返回契约，验证在计时区域之外完成。
                            invoke.last_grid=sol['lambda_grid']
                        return np.stack(outputs)
                    if case=='Spline_CPU':
                        values=np.asarray(query,dtype=np.float64).copy() if scope=='end_to_end' else query
                        calls[case]+=1;return spline.predict(values)[0]
                    outputs=[]
                    for first in range(0,n,32):
                        if scope=='model_only':
                            y=model(resident[first:first+32]);outputs.append(y*std+mean)
                        else:
                            x=normalize_input_field(build_fno1d_input_array(query[first:first+32,None],grid)[None],normalization)[0]
                            y=model(torch.from_numpy(x).to(device));outputs.append(recover_xyz(y.cpu().numpy(),normalization))
                        calls[case]+=1
                    return torch.cat(outputs,0) if scope=='model_only' else np.concatenate(outputs)
                def verify(v):
                    assert tuple(v.shape)==(n,1200,3)
                    if isinstance(v,torch.Tensor):assert torch.isfinite(v).all().item() and str(v.device)==device
                    else:assert np.isfinite(v).all()
                    assert np.array_equal(query,q[indices]) and rawsha(grid)==cp['lambda_sha256']
                    if case=='RK4_CPU':assert np.array_equal(invoke.last_grid,grid)
                before=calls[case]
                with torch.inference_mode():
                    summary,last=sample(case,scope,group,invoke,n,protocol['repeats'][group],protocol['warmups'][group],device,verify)
                assert calls[case]>before
                timings[case][group][scope]=summary
                validation[case][group+'_'+scope]={'status':'PASS','shape':[n,1200,3],'dtype':str(last.dtype),
                  'Q_indices':indices.tolist(),'Q_identity_sha256':rawsha(query),'lambda_sha256':rawsha(grid),'finite':True,'actual_compute_calls':calls[case]-before}
                if group=='single' and scope=='model_only':core_reference=last.detach().cpu().numpy() if isinstance(last,torch.Tensor) else last.copy()
                if group=='single' and scope=='end_to_end' and case!='RK4_CPU':
                    assert np.allclose(core_reference,last,rtol=1e-6,atol=1e-6),'Raw-output core and end-to-end paths differ'
                del last
            if model is not None:del resident
        if device=='cuda:0':
            resources[case]['allocator_peak_bytes']=torch.cuda.max_memory_allocated();resources[case]['allocator_reserved_bytes']=torch.cuda.memory_reserved()
        if model is not None:
            assert all(p.grad is None for p in model.parameters());del model,mean,std;gc.collect()
        mark('end_'+case)
        write(out,case+'-partial.json',{'timing':timings[case],'validation':validation[case]})
    rawfile.flush();os.fsync(rawfile.fileno());rawfile.close()
    for path,item in protocol['sources'].items():assert sha(path)==item['sha256'],path
    resources['system']={'hostname':os.uname().nodename,'cpu_model':next(l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),
      'affinity':sorted(os.sched_getaffinity(0)),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
      'cpu_stat_before':cpu_before,'cpu_stat_after':cpu_stat.read_text() if cpu_stat.exists() else None,
      'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'GPU':torch.cuda.get_device_name(0),
      'gpu_quota_gib_user_report':22.5,'gpu_memory_note':'PyTorch allocator counters, not HAMI physical free memory',
      'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
      'rss_scope':'Whole sequential benchmark process including resident data and temporary preparation copies; not isolated per-model RSS',
      'environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']},
      'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,'cuda':torch.version.cuda}
    ratios={'CPU_same_hardware':{},'CPU_GPU_cross_hardware_deployment':{}}
    for case in timings:
        if case=='RK4_CPU':continue
        target=ratios['CPU_GPU_cross_hardware_deployment' if case.endswith('GPU') else 'CPU_same_hardware']
        target[case]={g:timings['RK4_CPU'][g]['end_to_end']['median_seconds']/timings[case][g]['end_to_end']['median_seconds'] for g in ['single','Q400']}
    write(out,'resource-summary.json',resources)
    write(out,'timing-results.json',{'status':'COMPLETED','start_utc':start_time,'completed_utc':utc(),'preparation':preparation,'timings':timings,'RK4_over_method_latency_ratio':ratios,
      'validation':validation,'actual_compute_calls':calls,'all_input_hashes_unchanged':True,'all_samples_retained':True,
      'precision_results_unchanged':True,'new_training':False,'limitations':protocol['limitations']})
    mark('all_results_saved');print(json.dumps({'status':'COMPLETED','output':str(out),'sample_rows':len(rows),'ratios':ratios}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--phase',choices=['freeze','run'],required=True)
    args=p.parse_args();main(args.out,args.phase);mark('script_end')
