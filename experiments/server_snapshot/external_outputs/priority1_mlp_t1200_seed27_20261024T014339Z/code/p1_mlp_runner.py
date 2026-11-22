"""Isolated, resumable plain-MLP T1200 training; no held-out data access."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT=Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
sys.path.insert(0,str(ROOT))
from src.training.benchmark_data import Trajectories,fit_normalization,recover_xyz
from src.training.fno1d.input_builder_1d import build_fno1d_input_array
from src.training.fno2d.normalization_2d import normalize_input_field,normalize_output_field
from src.training.optimizer_builder import build_optimizer,build_scheduler

SCHEMA='priority1_plain_mlp_t1200_v1'
STOP=False


def utc():return dt.datetime.now(dt.timezone.utc)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def make_model(width=576):
    layers=[nn.Linear(2,width),nn.GELU()]
    for _ in range(3):layers.extend([nn.Linear(width,width),nn.GELU()])
    layers.append(nn.Linear(width,3))
    return nn.Sequential(*layers)


def capture_rng(device):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None}


def restore_rng(state,device):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not str(device).startswith('cuda'):raise ValueError('GPU checkpoint requires GPU resume')
        torch.cuda.set_rng_state(state['cuda'].cpu(),device)


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def publish(path,writer,replace=False):
    path=Path(path)
    for p in [path,*path.parents]:
        if p.is_symlink():raise ValueError('Symlink destination rejected')
    tmp=path.parent/('.'+path.name+'.'+uuid.uuid4().hex+'.part')
    with tmp.open('xb') as f:
        writer(f);f.flush();os.fsync(f.fileno())
    if replace:
        # 仅替换本实验的状态或指针；历史 checkpoint 文件始终不可变。
        os.replace(tmp,path)
    else:
        os.link(tmp,path);tmp.unlink()
    fsync_dir(path.parent)


def atomic_json(path,value,replace=False):
    b=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    publish(path,lambda f:f.write(b),replace)


def commit_checkpoint(directory,payload):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    name=f"epoch_{payload['epoch']:04d}_{uuid.uuid4().hex}"
    path=directory/(name+'.pt')
    publish(path,lambda f:torch.save(payload,f))
    # 提交记录发布后才是可恢复代次；未提交的 .pt 保留但不自动使用。
    record={'epoch':payload['epoch'],'file':path.name,'sha256':sha(path),'bytes':path.stat().st_size,
            'best_epoch':payload['best_epoch'],'best_validation_mse':payload['best_validation_mse'],
            'metrics':payload.get('metrics'),'committed_utc':utc().isoformat()}
    atomic_json(directory/(name+'.json'),record)
    return record


def load_committed(directory,record):
    directory=Path(directory);path=directory/record['file']
    if path.parent!=directory or path.suffix!='.pt' or path.is_symlink():raise ValueError('Invalid checkpoint path')
    if sha(path)!=record['sha256']:raise ValueError('Checkpoint SHA-256 mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['schema']!=SCHEMA or payload['epoch']!=record['epoch']:raise ValueError('Invalid checkpoint schema')
    return payload


def latest_commit(directory):
    records=[json.loads(p.read_text()) for p in Path(directory).glob('epoch_*.json')]
    if not records:raise ValueError('No committed checkpoint')
    records.sort(key=lambda r:(r['epoch'],r['committed_utc']))
    if len({r['epoch'] for r in records})!=len(records):raise ValueError('Ambiguous duplicate committed epoch')
    record=records[-1]
    return record,load_committed(directory,record)


def load_data(config):
    directory=Path(config['data_directory'])
    manifest_path=directory/'data-manifest.json'
    if sha(manifest_path)!=config['data_manifest_sha256']:raise ValueError('Data manifest changed')
    m=json.loads(manifest_path.read_text());path=directory/m['data_file']
    if sha(path)!=m['sha256']:raise ValueError('Data archive changed')
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!={'x_train','y_train','x_val','y_val','lambda_grid'}:raise ValueError('Unexpected dataset members')
        a={k:f[k].copy() for k in f.files}
    for k,v in a.items():
        if v.dtype!=np.float64 or list(v.shape)!=m['arrays'][k]['shape'] or not np.isfinite(v).all():raise ValueError('Invalid data array')
        if hashlib.sha256(v.tobytes()).hexdigest()!=m['arrays'][k]['raw_array_sha256']:raise ValueError('Array identity mismatch')
    if np.intersect1d(a['x_train'],a['x_val']).size:raise ValueError('Train/validation overlap')
    if a['y_train'].shape!=(1400,1200,3) or a['y_val'].shape!=(300,1200,3):raise ValueError('Unexpected split shape')
    if not np.array_equal(a['lambda_grid'],np.arange(1200,dtype=np.float64)*.005):raise ValueError('Grid mismatch')
    train=Trajectories(a['x_train'],a['lambda_grid'],a['y_train'])
    stats=fit_normalization(train)
    datasets={}
    for split in ['train','val']:
        x=normalize_input_field(build_fno1d_input_array(a['x_'+split],a['lambda_grid'])[None],stats)[0]
        y=normalize_output_field(a['y_'+split][None],stats)[0]
        datasets[split]=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(a['y_'+split]))
    return datasets,stats


def loaders(datasets,batch=32):
    return {k:DataLoader(v,batch_size=batch,shuffle=k=='train',drop_last=False,num_workers=0) for k,v in datasets.items()}


class SafeStop(Exception):pass


def epoch_pass(model,loader,device,stats,optimizer=None,stop_at=None):
    model.train(optimizer is not None)
    count=0;loss_sum=0.;error_sq=0.;truth_sq=0.;relative_sum=0.
    with torch.set_grad_enabled(optimizer is not None):
        for x,y,raw_truth in loader:
            if STOP or (stop_at and utc()>=stop_at-dt.timedelta(seconds=120)):
                raise SafeStop('Stop requested; discard incomplete epoch and retain latest committed state')
            x=x.to(device);y=y.to(device)
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            prediction=model(x);loss=nn.functional.mse_loss(prediction,y)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            if optimizer is not None:loss.backward();optimizer.step()
            n=len(x);loss_sum+=float(loss.detach())*n;count+=n
            raw=recover_xyz(prediction.detach().cpu().numpy(),stats).astype(np.float64)
            truth=raw_truth.numpy();delta=raw-truth
            if not np.isfinite(raw).all():raise RuntimeError('Nonfinite prediction')
            e=np.sum(delta*delta,axis=(1,2));t=np.sum(truth*truth,axis=(1,2))
            error_sq+=float(e.sum());truth_sq+=float(t.sum())
            relative_sum+=float((np.sqrt(e)/(np.sqrt(t)+1e-12)).sum())
    if not count:raise ValueError('Empty loader')
    return {'normalized_mse':loss_sum/count,'global_relative_l2':math.sqrt(error_sq)/(math.sqrt(truth_sq)+1e-12),
            'mean_per_q_relative_l2':relative_sum/count,'trajectory_count':count,
            'metric_scope':'online_pre_update_predictions' if optimizer else 'fixed_epoch_end_validation'}


def optimizer_pair(model,config):
    opt=build_optimizer(model,lr=config['lr'],weight_decay=config['weight_decay'])
    return opt,build_scheduler(opt,scheduler_gamma=config['gamma'])


def payload_for(model,opt,scheduler,epoch,best,best_epoch,best_state,history,stats,config,device,metrics=None):
    return {'schema':SCHEMA,'epoch':epoch,'model_state_dict':model.state_dict(),
            'optimizer_state_dict':opt.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
            'rng':capture_rng(device),'best_validation_mse':best,'best_epoch':best_epoch,
            'best_model_state_dict':best_state,'history':history,'normalization':stats.to_dict(),
            'config':config,'metrics':metrics}


def load_states(payload,model,opt,scheduler,device):
    model.load_state_dict(payload['model_state_dict'],strict=True)
    opt.load_state_dict(payload['optimizer_state_dict']);scheduler.load_state_dict(payload['scheduler_state_dict'])
    restore_rng(payload['rng'],device)


def run(config_path,output,stop_at,resume=False):
    global STOP
    config=json.loads(Path(config_path).read_text());output=Path(output).absolute()
    allowed=Path('/workspace/shanjinshuo/outputs/fno_kerr')
    if not output.is_relative_to(allowed) or output==allowed:raise ValueError('Invalid output directory')
    for p in [output,*output.parents]:
        if p.is_symlink():raise ValueError('Symlink output rejected')
    if output.resolve()!=output:raise ValueError('Noncanonical output path')
    if not resume:output.mkdir(exist_ok=False)
    lock=(output/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop_at=dt.datetime.fromisoformat(stop_at).astimezone(dt.timezone.utc)
    if utc()+dt.timedelta(minutes=5)>=stop_at:raise ValueError('Insufficient safe time')
    for path,h in config['source_sha256'].items():
        if sha(path)!=h:raise ValueError('Frozen source mismatch: '+path)
    if config['architecture']!=[2,576,576,576,576,3] or config['epochs']!=500 or config['batch_size']!=32:raise ValueError('Protocol mismatch')
    torch.set_num_threads(2);seed_all(config['seed']);device='cuda:0'
    model=make_model().to(device)
    if sum(p.numel() for p in model.parameters())!=1000515:raise ValueError('Parameter count mismatch')
    datasets,stats=load_data(config);loader=loaders(datasets,32)
    opt,scheduler=optimizer_pair(model,config)
    start_epoch=0;best=None;best_epoch=0;best_state=None;history=[]
    records=output/'checkpoints';records.mkdir(exist_ok=True)
    if resume:
        record,state=latest_commit(records)
        if state['config']!=config or state['normalization']!=stats.to_dict():raise ValueError('Resume provenance mismatch')
        load_states(state,model,opt,scheduler,device)
        start_epoch=state['epoch'];best=state['best_validation_mse'];best_epoch=state['best_epoch']
        best_state=state['best_model_state_dict'];history=state['history']
        for row in history:
            telemetry=output/f"epoch_{row['epoch']:04d}.json"
            if telemetry.exists():row['epoch_seconds']=json.loads(telemetry.read_text())['epoch_seconds']
        atomic_json(output/'LAST.json',record,True)
        best_records=[json.loads(p.read_text()) for p in records.glob('epoch_*.json')]
        selected=[r for r in best_records if r['epoch']==best_epoch]
        if selected:atomic_json(output/'BEST.json',selected[0],True)
    else:
        atomic_json(output/'config.json',config)
        atomic_json(output/'normalization.json',stats.to_dict())
        record=commit_checkpoint(records,payload_for(model,opt,scheduler,0,best,0,None,history,stats,config,device))
        atomic_json(output/'LAST.json',record,True)
    def sig_handler(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,sig_handler);signal.signal(signal.SIGINT,sig_handler)
    session={'pid':os.getpid(),'ppid':os.getppid(),'sid':os.getsid(0),'hostname':socket.gethostname(),
             'start_utc':utc().isoformat(),'server_local':dt.datetime.now().astimezone().isoformat(),
             'safe_stop_utc':stop_at.isoformat(),'resume':resume,'starting_epoch':start_epoch,
             'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
             'threads':torch.get_num_threads(),'matmul_precision':torch.get_float32_matmul_precision(),
             'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
             'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
             'optimizer_defaults':{k:v for k,v in opt.defaults.items() if isinstance(v,(int,float,bool,str,tuple,type(None)))}}
    atomic_json(output/('session_'+uuid.uuid4().hex+'.json'),session)
    epoch=start_epoch
    state_name='RUNNING'
    torch.cuda.reset_peak_memory_stats()
    def status(name,reason=None):
        recent=[h['epoch_seconds'] for h in history[-10:]]
        mean=sum(recent)/len(recent) if recent else None
        value={'status':name,'epoch':epoch,'remaining_epochs':500-epoch,'pid':os.getpid(),
               'utc':utc().isoformat(),'server_local':dt.datetime.now().astimezone().isoformat(),
               'safe_stop_utc':stop_at.isoformat(),'latest_metrics':history[-1] if history else None,
               'best_epoch':best_epoch,'best_validation_mse':best,'recent_mean_epoch_seconds':mean,
               'estimated_finish_utc':(utc()+dt.timedelta(seconds=mean*(500-epoch))).isoformat() if mean else None,
               'allocator_peak_bytes':torch.cuda.max_memory_allocated(),'allocator_reserved_bytes':torch.cuda.memory_reserved(),
               'memory_note':'PyTorch allocator counters, not physical free memory or a platform quota measurement',
               'reason':reason,'checkpoint_last':record}
        atomic_json(output/'status.json',value,True)
        return value
    status('RUNNING')
    try:
        for next_epoch in range(start_epoch+1,501):
            estimate=max([h['epoch_seconds'] for h in history[-5:]] or [120.])
            if STOP or utc()+dt.timedelta(seconds=estimate*1.5+150)>=stop_at:raise SafeStop('Insufficient time for another complete epoch')
            started=time.perf_counter();sync(device);lr=opt.param_groups[0]['lr']
            train=epoch_pass(model,loader['train'],device,stats,opt,stop_at)
            validation=epoch_pass(model,loader['val'],device,stats,None,stop_at)
            scheduler.step();improved=best is None or validation['normalized_mse']<best
            if improved:
                best=validation['normalized_mse'];best_epoch=next_epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            sync(device)
            row={'epoch':next_epoch,'train':train,'validation':validation,'learning_rate':lr,
                 'compute_seconds':time.perf_counter()-started,'epoch_seconds':time.perf_counter()-started,
                 'utc':utc().isoformat()}
            history.append(row)
            record=commit_checkpoint(records,payload_for(model,opt,scheduler,next_epoch,best,best_epoch,best_state,history,stats,config,device,row))
            atomic_json(output/'LAST.json',record,True)
            if improved:atomic_json(output/'BEST.json',record,True)
            row['epoch_seconds']=time.perf_counter()-started
            atomic_json(output/f'epoch_{next_epoch:04d}.json',row)
            epoch=next_epoch
            value=status('RUNNING')
            print(json.dumps(value),flush=True)
        state_name='COMPLETED';final=status(state_name)
        if not (output/'completion.json').exists():atomic_json(output/'completion.json',final)
    except SafeStop as e:
        # 未完成 epoch 的内存更新不覆盖已提交状态，下次从最后完整 epoch 继续。
        record,committed=latest_commit(records)
        epoch=committed['epoch'];best=committed['best_validation_mse'];best_epoch=committed['best_epoch'];history=committed['history']
        state_name='STOPPED_SAFE';status(state_name,str(e))
    except Exception as e:
        record,committed=latest_commit(records)
        epoch=committed['epoch'];best=committed['best_validation_mse'];best_epoch=committed['best_epoch'];history=committed['history']
        state_name='FAILED';status(state_name,type(e).__name__+': '+str(e));raise
    finally:
        if state_name!='RUNNING':print(json.dumps({'final_status':state_name,'epoch':epoch}),flush=True)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--safe-stop-utc',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.output,a.safe_stop_utc,a.resume)
