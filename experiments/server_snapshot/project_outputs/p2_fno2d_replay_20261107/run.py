import datetime,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R))
C=json.loads((W/'frozen.json').read_text());deadline=datetime.datetime.fromisoformat('2026-11-07T14:33:00+08:00')
def now():return datetime.datetime.now().astimezone()
def remaining():return (deadline-now()).total_seconds()
def save(n,obj):
 p=W/n;tmp=p.with_suffix('.tmp.'+str(os.getpid()))
 with tmp.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def gpu():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
 if s.strip():raise RuntimeError('GPU occupied; refusing to launch')
 if shutil.disk_usage(W).free<4*2**30 or remaining()<180:raise RuntimeError('Disk or deadline guard')
def paths(key,seed):
 cp=R/C['config'][key]['best'] if seed==27 else W/f'A_seed{seed}'/'checkpoints/best_model.pt'
 return cp
def plan_a(key,seed):
 cfg=C['config'][key];cp=paths(key,seed);task=cfg['task'];name=cfg['name'];out=W/f'{key}_{seed}_planA';script='evaluate_formal_length_extrapolation_r3_2d.py' if key=='C' else 'evaluate_formal_length_extrapolation_2d.py'
 cmd=[sys.executable,'-B',str(R/'scripts'/script),'--training-task-name',task,'--model-name',name,'--checkpoint-path',str(cp),'--output-dir',str(out),'--dataset-pair-validation-json',str(R/C['pair']),'--device','cuda']
 for kind,t in zip(['short','medium','long'],[1200,1800,2400]):cmd+=['--'+kind+'-task-name',f'q_1p6007-2p9993_n400_t{t}']
 subprocess.run(cmd,check=True)
 return json.loads(next(out.glob('*summary.json')).read_text())['results']
def plan_b(key,seed):
 import numpy as np
 from scripts import evaluate_plan_b_resolution_generalization_2d as pb
 from scripts.run_analysis_2d import load_checkpoint_2d,load_fno2d_checkpoint_model
 from scripts.run_plan_b_bidirectional_reverse_2d import validate_t2399_checkpoint_contract
 from src.training.benchmark_data import EVAL_TASKS
 cp=load_checkpoint_2d(paths(key,seed),'cuda');ct=(pb.validate_checkpoint_contract if key=='A' else validate_t2399_checkpoint_contract)(cp);model=load_fno2d_checkpoint_model(cp,'cuda');out={};ref=None
 for t,name in EVAL_TASKS.items():
  field=pb.build_canonical_field_from_artifact(R/'data/tasks'/name)
  if ref is None:ref=field.canonical_q.copy()
  assert len(ref)==400 and np.array_equal(ref,field.canonical_q)
  pred=pb.run_frozen_forward(model=model,contract=ct,field=field,device='cuda');out['T'+str(t)]=pb.compute_raw_metrics(pred,field.canonical_truth,ref)
 save(f'{key}_{seed}_planB.json',out);return out
def replay(key,seed):
 checks=[]
 def eq(a,b):checks.append(math.isclose(float(a),float(b),rel_tol=1e-4,abs_tol=1e-8))
 if key in ['A','C']:
  result=plan_a(key,seed)
  if seed==27:
   for grid,v in C['refA'][key].items():
    for region in ['prefix','extrapolation','full']:
     for metric in ['mean_per_q_relative_l2','global_relative_l2','mse']:
      if v[region] is not None:eq(result[grid][region][metric],v[region][metric])
 if key in ['A','B']:
  result=plan_b(key,seed)
  if seed==27:
   for metric,rows in C['refB'].items():
    for grid,val in rows['theta1200' if key=='A' else 'theta2399'].items():eq(result[grid][metric],val)
 save(f'{key}_{seed}_verification.json',{'best_sha256':digest(paths(key,seed)),'checked_metrics':len(checks),'passed':all(checks),'seed':seed,'historical_execution_hash_retrospectively_proven':False})
def train(seed):
 assert seed in [41,73] and json.loads((W/'A_27_verification.json').read_text())['passed']
 from scripts import train_model_2d as t
 cfg=C['config']['A'];old=json.loads((R/cfg['best']).parent.parent.joinpath('run_config.json').read_text());base=W/f'A_seed{seed}';base.mkdir(exist_ok=False)
 def dirs(task_name,model_name):
  assert task_name==cfg['task'] and model_name==cfg['name'];d=dict(model_dir=base,run_config_json=base/'run_config.json',summary_json=base/'summary.json')
  for n in ['checkpoints','logs','inference','analysis']:(base/n).mkdir();d[n+'_dir']=base/n
  return d
 t.get_model_output_dirs=dirs;origjson=t.save_json_common
 def guarded(obj,path,**kw):
  if Path(path).name=='run_config.json':
   for k in old:
    if k not in ['training_seed','dataset_summary']:assert obj[k]==old[k],k
   for k in ['normalization_stats','target_transform_config','param_name','vary_params_order']:assert obj['dataset_summary'][k]==old['dataset_summary'][k],k
   assert obj['training_seed']==seed
  return origjson(obj,path,**kw)
 t.save_json_common=guarded;orig=t.save_checkpoint_2d;times=[];last=time.monotonic()
 def atomic(path,model,optimizer,epoch,best_val_mse,config):
  nonlocal last
  tmp=path.with_name(path.name+'.partial');orig(tmp,model,optimizer,epoch,best_val_mse,config)
  with tmp.open('rb') as f:os.fsync(f.fileno())
  os.replace(tmp,path)
  if path.name=='last_model.pt':
   stamp=time.monotonic();times.append(stamp-last);last=stamp;mean=sum(times[-10:])/len(times[-10:]);save('active.json',{'seed':seed,'epoch':epoch,'pid':os.getpid(),'local':now().isoformat(),'seconds_per_epoch':mean,'eta':(now()+datetime.timedelta(seconds=(500-epoch)*mean)).isoformat()})
   if remaining()<max(120,3*mean) or shutil.disk_usage(W).free<3*2**30:raise RuntimeError('Safe stop after checkpoint')
 t.save_checkpoint_2d=atomic
 sys.argv=[str(R/'scripts/train_model_2d.py'),'--task-name',cfg['task'],'--device','cuda','--training-seed',str(seed),'--modes-param','16','--modes-lambda','32','--width','64','--depth','4','--hidden-dim','128','--normalization','standard','--target-transform','raw','--batch-size','1','--epochs','500','--lr','0.001','--weight-decay','0.0001','--scheduler-gamma','0.995','--print-every','10'];t.main()
def queue():
 state={'completed':[],'safe_stop':deadline.isoformat(),'sid':os.getsid(0),'pid':os.getpid(),'stage2':'RESERVED','interpretation':'DEFERRED'}
 try:
  for path,expected in C['sha'].items():assert digest(R/path)==expected,path
  for mode,key,seed in [('eval','A',27),('eval','B',27),('eval','C',27),('train','A',41),('eval','A',41),('train','A',73),('eval','A',73)]:
   if mode=='train':
    assert json.loads((W/'A_27_verification.json').read_text())['passed'],'Historical reproduction mismatch: training blocked'
    assert remaining()>1800,'Insufficient training window'
   gpu();label=f'{mode}_{key}_{seed}';state.update(status='RUNNING',active=label);save('queue.json',state)
   with (W/(label+'.log')).open('x') as f:subprocess.run([sys.executable,'-B','-u',__file__,mode,key,str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
   state['completed'].append(label)
  state.update(status='BATCH_COMPLETE',active=None)
 except Exception as exc:
  state.update(status='STOPPED_REQUIRES_REVIEW',error=str(exc));raise
 finally:save('queue.json',state)
if __name__=='__main__':
 if len(sys.argv)==1:queue()
 elif sys.argv[1]=='eval':replay(sys.argv[2],int(sys.argv[3]))
 else:train(int(sys.argv[3]))