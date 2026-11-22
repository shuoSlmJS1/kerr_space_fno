"""Bounded launch-gate tests for the isolated plain MLP runner."""
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset
import p1_mlp_runner as r


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main(config_path,out):
    out=Path(out);out.mkdir(exist_ok=False)
    config=json.loads(Path(config_path).read_text());device='cuda:0'
    torch.set_num_threads(2);r.seed_all(27)
    for path,h in config['source_sha256'].items():assert r.sha(path)==h
    datasets,stats=r.load_data(config)
    # 对照历史逐样本路径，确保批量预处理没有改变输入和标签。
    with np.load(Path(config['data_directory'])/'train1400_val300_t1200.npz') as f:
        historical=r.Trajectories(f['x_train'],f['lambda_grid'],f['y_train'],stats)
        for i in [0,31,1399]:
            x,y=historical[i]
            assert torch.equal(x,datasets['train'].tensors[0][i])
            assert torch.equal(y,datasets['train'].tensors[1][i])
    result={'source_hashes':'PASS','data_and_normalization':'PASS','normalization':stats.to_dict()}
    model=r.make_model().to(device);count=sum(p.numel() for p in model.parameters());assert count==1000515
    opt,sched=r.optimizer_pair(model,config)
    torch.cuda.reset_peak_memory_stats();r.sync(device);started=time.perf_counter()
    x,y,_=next(iter(r.loaders(datasets)['train']));x=x.to(device);y=y.to(device)
    before=model[0].weight.detach().clone();pred=model(x)
    assert list(pred.shape)==[32,1200,3] and torch.isfinite(pred).all()
    loss=torch.nn.functional.mse_loss(pred,y);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step();assert not torch.equal(before,model[0].weight)
    r.sync(device);elapsed=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
    assert peak<22.5*1024**3 and reserved<22.5*1024**3
    result['full_batch']={'status':'PASS','parameters':count,'shape':list(pred.shape),'seconds':elapsed,
                          'normalized_mse':float(loss.detach()),'allocator_peak_bytes':peak,'allocator_reserved_bytes':reserved,
                          'memory_note':'Allocator usage only; user-declared quota is 22.5 GiB'}
    del model,opt,sched,x,y,pred,loss,before
    # 小型断点测试使用同一正式网络；不读取或选择 held-out 数据。
    r.seed_all(4321)
    tiny={}
    for name,n in [('train',5),('val',3)]:
        x=torch.randn(n,8,2);y=torch.randn(n,8,3)
        raw=torch.from_numpy(r.recover_xyz(y.numpy(),stats).astype(np.float64))
        tiny[name]=TensorDataset(x,y,raw)
    loader=r.loaders(tiny,2)
    def initialized():
        r.seed_all(27);m=r.make_model().to(device);o,s=r.optimizer_pair(m,config);return m,o,s
    def advance(m,o,s,epoch,best,best_epoch,best_state,history):
        train=r.epoch_pass(m,loader['train'],device,stats,o)
        val=r.epoch_pass(m,loader['val'],device,stats)
        s.step()
        if best is None or val['normalized_mse']<best:
            best=val['normalized_mse'];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        row={'epoch':epoch,'train':train,'validation':val};history.append(row)
        return r.payload_for(m,o,s,epoch,best,best_epoch,best_state,history,stats,config,device,row)
    m,o,s=initialized();a=advance(m,o,s,1,None,0,None,[])
    ck=out/'resume_checkpoints';record=r.commit_checkpoint(ck,a)
    target=copy.deepcopy(advance(m,o,s,2,a['best_validation_mse'],a['best_epoch'],a['best_model_state_dict'],a['history']))
    del m,o,s,a
    m,o,s=initialized();state=r.load_committed(ck,record);r.load_states(state,m,o,s,device)
    actual=advance(m,o,s,2,state['best_validation_mse'],state['best_epoch'],state['best_model_state_dict'],state['history'])
    assert equal(target,actual),'Interrupted/resumed training differs from continuous training'
    # 未提交文件不参与恢复；重复发布不覆盖既有资产；哈希错误必须拒绝。
    with (ck/'epoch_9999_orphan.pt').open('xb') as f:f.write(b'uncommitted')
    latest,_=r.latest_commit(ck);assert latest==record
    bad=dict(record);bad['sha256']='0'*64
    try:r.load_committed(ck,bad)
    except ValueError:pass
    else:raise AssertionError('Corrupt hash accepted')
    probe=out/'exclusive_probe.json';r.atomic_json(probe,{'generation':1})
    try:r.atomic_json(probe,{'generation':2})
    except FileExistsError:pass
    else:raise AssertionError('Existing asset overwritten')
    assert json.loads(probe.read_text())=={'generation':1}
    r.atomic_json(out/'state_probe.json',{'generation':1});r.atomic_json(out/'state_probe.json',{'generation':2},True)
    assert json.loads((out/'state_probe.json').read_text())=={'generation':2}
    r.STOP=True
    try:r.epoch_pass(m,loader['train'],device,stats,o)
    except r.SafeStop:pass
    else:raise AssertionError('Safe-stop ignored')
    r.STOP=False
    result.update({'resume_equivalence':'PASS_EXACT','resume_compared':['model','optimizer','scheduler','RNG','BEST','metrics'],
                   'atomic_NFS_publication':'PASS','hash_rejection':'PASS','orphan_ignored':'PASS','safe_stop':'PASS',
                   'test_status':'PASS','utc':r.utc().isoformat(),'torch':str(torch.__version__),
                   'gpu':torch.cuda.get_device_name(0),'pid':os.getpid()})
    r.atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)
    print('TEST_COMPUTATION_FINISHED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.config,a.output)
