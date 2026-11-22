import datetime,json,tempfile
from pathlib import Path
import numpy as np
import torch
import stage1_runner as r
b=r.b
results=[]
def passed(name):results.append({'test':name,'status':'PASS'})
b.TRAINING['seed']=41
torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='synthetic-tests-',dir=r.SESSION) as tmp:
 root=Path(tmp)
 def inputs():
  q=np.linspace(1.6,3,48).reshape(-1,1);lam=np.linspace(0,1,7)
  xyz=np.stack([q+lam,q-lam,q*lam],axis=-1)
  return {'train':r.d.Trajectories(q[:40],lam,xyz[:40]),'val':r.d.Trajectories(q[40:],lam,xyz[40:])}
 def model():return torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(8,3))
 data=inputs();stats=r.d.fit_normalization(data['train'])
 original_epoch=b.epoch_mse;original_save=b.save_checkpoint
 full=root/'full';full.mkdir();b.seed_training();m=model();b.fit_epochs(m,inputs(),stats,'cpu',full,{},epochs=6)
 expected=torch.load(full/'epoch_0006.pt',weights_only=True)
 interrupted=root/'interrupted';interrupted.mkdir();b.seed_training();m=model()
 progress=r.Progress({'id':'synthetic'},root/'status.json',r.utc()+datetime.timedelta(hours=1));b.save_checkpoint=r.publish_checkpoint;progress.install()
 b.fit_epochs(m,inputs(),stats,'cpu',interrupted,{},epochs=3)
 resume=torch.load(interrupted/'epoch_0003.pt',weights_only=True)
 b.fit_epochs(m,inputs(),stats,'cpu',interrupted,{},epochs=6,resume=resume)
 actual=torch.load(interrupted/'epoch_0006.pt',weights_only=True)
 assert expected['history']==actual['history']
 for key in ['model_state_dict','best_model_state_dict']:
  assert all(torch.equal(expected[key][n],actual[key][n]) for n in expected[key])
 assert expected['scheduler_state_dict']==actual['scheduler_state_dict']
 assert expected['best_epoch']==actual['best_epoch']
 passed('Original uninterrupted vs adapted interrupted/resumed kernel: exact histories, final weights, BEST weights, scheduler')
 path=interrupted/'epoch_0006.pt';before=r.digest(path)
 try:r.publish_checkpoint(path,{'overwrite':True})
 except FileExistsError:pass
 else:raise AssertionError('Overwrite was permitted')
 assert r.digest(path)==before;passed('Atomic checkpoint publication refuses overwrite and preserves original bytes')
 for seed in [27,89,131]:
  try:r.setup(seed)
  except ValueError:pass
  else:raise AssertionError('Unauthorized training seed')
 passed('Seed 27, 89 and 131 launch gates reject before work')
 stopped=r.Progress({'id':'synthetic'},root/'deadline.json',r.utc()-datetime.timedelta(seconds=1))
 try:stopped.guard()
 except r.SafeStop:pass
 else:raise AssertionError('Deadline was ignored')
 assert json.loads((root/'deadline.json').read_text())['status']=='SAFE_STOP';passed('Expired deadline fails closed')
 b.epoch_mse=original_epoch;b.save_checkpoint=original_save
r.atomic_json(r.HERE/'tests-result.json',{'status':'PASS','scope':'Synthetic CPU tests; not formal scientific results','tests':results,'utc':r.utc().isoformat()})
print(json.dumps(results))