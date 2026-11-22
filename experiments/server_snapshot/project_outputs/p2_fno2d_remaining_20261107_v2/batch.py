import sys,os,time,json,datetime,subprocess,shutil,hashlib,math,fcntl
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1];P=R/'outputs/p2_fno2d_replay_20261107';sys.path[:0]=[str(P),str(R)]
import run as prior
def read(p):return json.loads(p.read_text())
def now():return datetime.datetime.now().astimezone()
def rem():return (prior.deadline-now()).total_seconds()
def put(p,v):
 tmp=p.with_name(p.name+'.partial')
 with tmp.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def cp(k,s):return W/f'{k}_seed{s}'/'checkpoints/best_model.pt'
def evalunit(k,s):
 prior.paths=cp;prior.W=W
 if k=='B':g=prior.plan_b(k,s);assert len(g)==4
 else:g=prior.plan_a(k,s);assert len(g)==3
 assert all(math.isfinite(float(v['mean_per_q_relative_l2'] if k=='B' else v['full']['mean_per_q_relative_l2'])) for v in g.values())
 put(W/f'{k}_{s}_evaluated.json',{'seed':s,'best_sha256':prior.digest(cp(k,s)),'grids':list(g),'evaluation_complete':True})
def train(k,s):
 u=W/f'{k}_seed{s}';u.mkdir(exist_ok=False)
 old=read((R/prior.C['config'][k]['best']).parent.parent/'run_config.json')
 stamp=time.monotonic();times=[]
 def tick(epoch):
  nonlocal stamp
  n=time.monotonic();times.append(n-stamp);stamp=n;rate=sum(times[-10:])/len(times[-10:])
  put(W/'active.json',dict(config=k,seed=s,epoch=epoch,pid=os.getpid(),seconds_per_epoch=rate,local=now().isoformat(),eta=(now()+datetime.timedelta(seconds=(500-epoch)*rate)).isoformat()))
 def guard(epoch):
  if rem()<max(100,3*max(times[-4:] or [10])) or shutil.disk_usage(W).free<3*2**30:raise RuntimeError('Safe stop after committed history/checkpoint')
 def writer(orig):
  def f(path,*args,**kw):
   tmp=path.with_name(path.name+'.partial');orig(tmp,*args,**kw)
   with tmp.open('rb') as stream:os.fsync(stream.fileno())
   os.replace(tmp,path)
   if path.name=='last_model.pt':tick(kw['epoch'] if 'epoch' in kw else args[2])
  return f
 def match(cfg,skip):
  assert cfg['training_seed']==s
  for field in old:
   if field not in skip:assert json.loads(json.dumps(cfg[field]))==old[field],f'Historical config mismatch: {field}'
 if k=='B':
  from scripts import run_plan_b_bidirectional_reverse_2d as b
  tasks=R/'data/tasks';source,validation=b.validate_historical_t1200_source(tasks/'q_1p6-3_n2000_t1200');ds=tasks/'q_1p6-3_n2000_t2399_plan_b_matched_v1'
  b._require_matched_t2399_dataset(ds);assert b.validate_plan_b_ground_truth(source.directory,ds)['structural_valid']
  spec=b._training_spec_from_contract();spec.training_config['training_seed']=s
  b._save_training_checkpoint=writer(b._save_training_checkpoint);orig=b._write_json
  def write(path,obj):
   if path.name=='run_config.json':match(obj,{'training_seed'})
   orig(path,obj)
   if path.name=='train_history.json':guard(len(obj['lr']))
  b._write_json=write
  b.train_matched_t2399_model(dataset_dir=ds,output_dir=u,device='cuda',source_validation=validation,spec=spec)
 else:
  from scripts import train_fno2d_physical_frequency_r3 as c
  task=old['task_name'];name='p2_r3_b1_seed'+str(s)
  def dirs(task_name,model_name):
   assert task_name==task and model_name==name
   d=dict(model_dir=u,run_config_json=u/'run_config.json',summary_json=u/'summary.json')
   for key in ['checkpoints','logs','inference','analysis']:(u/key).mkdir();d[key+'_dir']=u/key
   return d
  c.get_model_output_dirs=dirs;c.save_checkpoint_2d=writer(c.save_checkpoint_2d);orig=c.save_json
  def write(obj,path):
   if path.name=='run_config.json':match(obj,{'training_seed','run_name','model_name','runtime'})
   orig(obj,path)
   if path.name=='train_history.json':guard(len(obj['epochs']))
  c.save_json=write;c.main(['--task-name',task,'--run-name',name,'--training-seed',str(s),'--device','cuda','--epochs','500','--train-lengths',*map(str,old['train_lengths']),'--validation-lengths',*map(str,old['validation_lengths'])])
def queue():
 state=dict(status='RUNNING',completed=[],pending=['B41','B73','C41','C73'],stage2='NOT_AUTHORIZED',safe_stop=prior.deadline.isoformat(),pid=os.getpid(),sid=os.getsid(0));timings={}
 try:
  assert read(P/'queue.json')['status']=='BATCH_COMPLETE' and read(P/'launch.json')['wrapper_sha']==prior.digest(P/'run.py') and read(P/'launch.json')['manifest_sha']==prior.digest(P/'frozen.json')
  for p,h in prior.C['sha'].items():assert prior.digest(R/p)==h,p
  for k in ['B','C']:
   v=read(P/f'{k}_27_verification.json');assert v['passed'] and v['best_sha256']==prior.digest(R/prior.C['config'][k]['best'])
   for seed in [41,73]:
    label=k+str(seed);budget=(timings[k]*1.12+100 if k in timings else {'B':1649,'C':1215}[k]+180)
    if rem()<budget:state.update(status='SAFE_STOP_PENDING',reason='Admission estimate exceeds remaining time',next_unit=label);return
    prior.gpu();assert not (W/f'{k}_seed{seed}').exists();start=time.monotonic()
    for mode in ['train','eval']:
     state.update(active=label,mode=mode);put(W/'queue.json',state)
     with (W/f'{label}_{mode}.log').open('x') as f:subprocess.run([sys.executable,'-B','-u',__file__,mode,k,str(seed)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    timings[k]=time.monotonic()-start;state['completed'].append(label);state['pending'].remove(label)
  state['status']='BATCH_COMPLETE_REQUIRES_ACCEPTANCE'
 except Exception as e:state.update(status='STOPPED_REQUIRES_REVIEW',error=str(e));raise
 finally:put(W/'queue.json',state)
if __name__=='__main__':
 if len(sys.argv)==1:queue()
 else:
  m,k,s=sys.argv[1:];s=int(s);assert k in ['B','C'] and s in [41,73] and m in ['train','eval'];(train if m=='train' else evalunit)(k,s)