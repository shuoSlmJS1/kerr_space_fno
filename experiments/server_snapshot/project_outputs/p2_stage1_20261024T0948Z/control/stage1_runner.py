import argparse,datetime,hashlib,json,os,signal,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'protocol.json').read_text())
ROOT=Path(CONFIG['project_root'])
SESSION=Path(CONFIG['session_root'])
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.training import benchmark_track_a as b
from src.training import benchmark_data as d

STOP=False
class SafeStop(Exception):pass

def utc():return datetime.datetime.now(datetime.timezone.utc)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def check_path(path):
 path=Path(path)
 if not path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe absolute path')
 for item in [path,*path.parents]:
  if item.is_symlink():raise ValueError('Symlink rejected: '+str(item))
 return path

def atomic_json(path,data):
 path=check_path(path);temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with temp.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def publish_checkpoint(path,payload):
 path=check_path(path)
 if not path.is_relative_to(SESSION):raise ValueError('Checkpoint outside session')
 temp=path.with_name(path.name+'.partial.'+str(os.getpid()))
 with temp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
 os.link(temp,path);temp.unlink()

_original_pair=d.validate_pair

def migrated_pair(source,target):
 replay=target.metadata.get('paired_replay',{})
 previous=replay.get('source_dataset_path')
 allowed='/home/shanjinshuo/fno_kerr/kerr_project/data/tasks/'+source.directory.name
 current=ROOT/'data/tasks'/source.directory.name
 if previous==allowed and source.directory==current:
  for asset in [source,target]:
   for name in ['dataset.npz','meta.json']:
    p=asset.directory/name;rel=p.relative_to(ROOT).as_posix()
    if digest(p)!=CONFIG['asset_sha256'].get(rel):raise ValueError('Migration hash mismatch: '+rel)
  replay['source_dataset_path']=str(current)
  try:return _original_pair(source,target)
  finally:replay['source_dataset_path']=previous
 return _original_pair(source,target)

def provenance():
 hashes={name:digest(ROOT/name) for name in CONFIG['source_sha256']}
 if hashes!=CONFIG['source_sha256']:raise ValueError('Registered source changed')
 return {'git_head':None,'git_status':'SOURCE_SNAPSHOT_WITHOUT_GIT','historical_reference_git_head':CONFIG['historical_track_a'][0]['source_provenance']['git_head'],'source_sha256':hashes,'execution_adapter_sha256':digest(__file__),'protocol_sha256':digest(HERE/'protocol.json')}

def setup(seed):
 if seed not in CONFIG['launch_seeds'] or seed not in [41,73]:raise ValueError('Seed is not authorized for Stage 1 launch')
 b.TRAINING['seed']=seed
 b.source_provenance=provenance
 b.save_checkpoint=publish_checkpoint
 d.validate_pair=migrated_pair
 torch.set_num_threads(1)
 torch.set_num_interop_threads(1)
 provenance()
 for name,value in CONFIG['asset_sha256'].items():
  if digest(check_path(ROOT/name))!=value:raise ValueError('Registered data changed: '+name)

def signal_stop(*unused):
 global STOP
 STOP=True

class Progress:
 def __init__(self,unit,status,deadline,resume=None):
  self.unit=unit;self.path=status;self.deadline=deadline;self.epoch=0;self.times=[];self.started=time.perf_counter();self.last=None;self.best=float('inf');self.best_epoch=0
  if resume:
   state=torch.load(resume,map_location='cpu',weights_only=True);self.epoch=state['epoch'];self.best=state['best_validation_mse'];self.best_epoch=state['best_epoch'];self.last=str(resume)
 def write(self,status,**extra):
  mean=sum(self.times[-10:])/len(self.times[-10:]) if self.times else None
  body={'status':status,'unit':self.unit,'pid':os.getpid(),'utc':utc().isoformat(),'epoch':self.epoch,'remaining_epochs':500-self.epoch,'safe_stop_utc':self.deadline.isoformat(),'recent_mean_epoch_seconds':mean,'estimated_finish_utc':(utc()+datetime.timedelta(seconds=mean*(500-self.epoch))).isoformat() if mean else None,'eta_basis':'MEASURED recent up to 10 epochs on current device; excludes future evaluation and checkpoint I/O','best_epoch':self.best_epoch,'best_validation_mse':self.best if self.best_epoch else None,'last_committed_checkpoint':self.last,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,'peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,**extra}
  atomic_json(self.path,body)
 def guard(self):
  remaining=(self.deadline-utc()).total_seconds()
  estimate=(sum(self.times[-10:])/len(self.times[-10:])) if self.times else 10
  until_commit=25-(self.epoch%25)
  if STOP or remaining < max(90,estimate*(until_commit+2)*1.5):
   self.write('SAFE_STOP',reason='Signal or insufficient time to next immutable checkpoint')
   raise SafeStop('Deadline guard reached; retain last committed checkpoint')
 def install(self):
  original_epoch=b.epoch_mse;original_save=b.save_checkpoint
  def epoch(model,loader,device,optimizer=None):
   if optimizer is not None:self.guard();self.started=time.perf_counter()
   result=original_epoch(model,loader,device,optimizer)
   if optimizer is None:
    b.synchronize(device);self.epoch+=1;self.times.append(time.perf_counter()-self.started)
    if result<self.best:self.best=result;self.best_epoch=self.epoch
    self.write('RUNNING',validation_normalized_mse=result)
    if self.epoch%25==0:print(json.dumps({'epoch':self.epoch,'unit':self.unit,'recent_epoch_seconds':sum(self.times[-10:])/len(self.times[-10:])}),flush=True)
   return result
  def save(path,payload):
   original_save(path,payload)
   if Path(path).name.startswith('epoch_'):
    self.last=str(path);self.write('RUNNING',checkpoint_sha256=digest(path))
  b.epoch_mse=epoch;b.save_checkpoint=save

def preflight():
 setup(41)
 checked=[]
 for row in CONFIG['historical_track_a']:
  path=ROOT/'outputs/benchmark_track_a_v1'/Path(row['checkpoint_metadata']['path'].replace('\\','/')).parent.parent.name/('train_t'+str(row['train_resolution']))/'best_model.pt'
  if digest(path)!=row['best_checkpoint_sha256']:raise ValueError('Historical BEST hash mismatch')
  payload=torch.load(path,map_location='cpu',weights_only=True)
  b.TRAINING['seed']=27
  model,stats=b.checkpoint_contract(payload)
  if not all(torch.isfinite(v).all().item() for v in model.state_dict().values()):raise ValueError('Historical tensor is nonfinite')
  checked.append({'model':row['model_name'],'resolution':row['train_resolution'],'strict_load':True,'finite':True,'sha256':digest(path)})
 b.TRAINING['seed']=41
 for t in [1200,2399]:
  datasets,prov=d.load_training(t);stats=d.fit_normalization(datasets['train'])
  historical=next(v for v in CONFIG['historical_track_a'] if v['train_resolution']==t)
  if prov['dataset_sha256']!=historical['dataset_provenance']['dataset_sha256']:raise ValueError('Training identity mismatch')
 for t in [1200,2399,3598,4797]:d.load_evaluation(t)
 result={'status':'PASS','historical_best_checks':checked,'registered_data_and_migration_checks':'PASS','source_hashes':'PASS','cuda_preflight':b.inspect_device('cuda:0',0),'utc':utc().isoformat()}
 atomic_json(HERE/'preflight-result.json',result);print(json.dumps(result),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval','preflight']);p.add_argument('--unit');p.add_argument('--resume');args=p.parse_args()
 if args.mode=='preflight':return preflight()
 unit=next(v for v in CONFIG['run_units'] if v['id']==args.unit)
 if unit['family']!='track_a' or not unit['enabled']:raise ValueError('Unit is not verified for execution')
 setup(unit['seed']);deadline=datetime.datetime.fromisoformat(CONFIG['safe_stop_utc'])
 if (deadline-utc()).total_seconds()<180:raise SafeStop('Too close to safety cutoff to start')
 signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
 directory=SESSION/'runs'/unit['id'];status=SESSION/(unit['id']+'-'+args.mode+'-status.json')
 progress=Progress(unit,status,deadline,args.resume)
 progress.write('STARTING')
 try:
  if args.mode=='train':
   progress.install()
   b.train_run(unit['model'],unit['train_resolution'],directory,'cuda:0',0,True,args.resume)
  else:
   original_predict=b.predict_frozen
   def guarded_predict(*a,**kw):
    if STOP or (deadline-utc()).total_seconds()<90:raise SafeStop('Evaluation deadline guard')
    return original_predict(*a,**kw)
   b.predict_frozen=guarded_predict
   b.evaluate_run(directory/'best_model.pt',SESSION/'evaluations'/unit['id'],'cuda:0',0,True)
  progress.write('COMPLETE')
 except SafeStop as e:
  progress.write('SAFE_STOP',reason=str(e));raise SystemExit(75)
 except Exception as e:
  progress.write('FAILED_NO_PROTOCOL_ADAPTATION',error_type=type(e).__name__,reason=str(e));raise

if __name__=='__main__':main()