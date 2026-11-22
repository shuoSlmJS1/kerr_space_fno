import argparse,collections,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path


def utc():return datetime.datetime.now(datetime.timezone.utc)
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def save(path,payload):
 path=Path(path)
 if path.is_symlink():raise ValueError('Symlink output rejected')
 temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with temp.open('x') as f:json.dump(payload,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 values=(p/'stat').read_text().rsplit(')',1)[1].split()
 return {'state':values[0],'ppid':int(values[1]),'start_ticks':int(values[19]),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
def classify(root,unit,active=None,alive=False):
 uid=unit['id'];run=root/'runs'/uid;ev=root/'evaluations'/uid
 row={'id':uid,'seed':unit['seed'],'family':unit['family'],'status':'PENDING','eligibility':'READY' if unit['enabled'] else 'REQUIRES_VERIFICATION','run_directory':str(run),'evaluation_directory':str(ev),'next_action':'TRAIN_FROM_SCRATCH','resume_checkpoint':None}
 checkpoints=sorted(run.glob('epoch_*.pt')) if run.exists() else []
 if checkpoints:
  row['resume_checkpoint']=str(checkpoints[-1]);row['next_action']='VALIDATE_AND_RESUME_CHECKPOINT';row['resume_validation_required']=True
 if (run/'summary.json').exists():
  s=read(run/'summary.json');best=run/'best_model.pt'
  if s['completed_epochs']!=500 or s['training']['seed']!=unit['seed'] or not best.is_file() or sha(best)!=s['best_checkpoint_sha256']:raise ValueError('Completed training identity mismatch: '+uid)
  row.update(next_action='FROZEN_EVALUATION_ONLY',training_complete=True,best_checkpoint=str(best),best_sha256=s['best_checkpoint_sha256'])
  if (ev/'matrix.json').exists():
   m=read(ev/'matrix.json')
   if m['checkpoint_sha256']!=s['best_checkpoint_sha256'] or set(m['metrics'])!={'1200','2399','3598','4797'}:raise ValueError('Evaluation identity mismatch: '+uid)
   row.update(status='COMPLETED',next_action='SKIP_ALREADY_COMPLETED',evaluation_matrix=str(ev/'matrix.json'));return row
 if uid==active and alive:row.update(status='RUNNING',next_action='ALLOW_CURRENT_UNIT_TO_FINISH');return row
 for mode in ['train','eval']:
  p=root/(uid+'-'+mode+'-status.json')
  if p.exists():
   s=read(p)
   if s['status'].startswith('FAILED'):
    row.update(status='FAILED',next_action='REVIEW_FAILURE_BEFORE_RESUME',failure_reason=s.get('reason'));return row
 if (ev.exists() and not (ev/'matrix.json').exists()):row.update(status='FAILED',next_action='REVIEW_PARTIAL_EVALUATION_WITHOUT_OVERWRITE')
 return row

def persist(root,pause,config,request,phase,alive=False,extra=None):
 rows=[classify(root,u,request['active_unit'],alive) for u in config['run_units']]
 counts={key:sum(r['status']==key for r in rows) for key in ['COMPLETED','RUNNING','PENDING','FAILED']}
 manifest={'schema':'p2_pause_resume_v1','utc':utc().isoformat(),'queue_status':phase,'automatic_next_unit':False,'active_unit':request['active_unit'] if alive else None,'counts':counts,'units':rows,'historical_seed27_track_a':'12 verified historical assets reused; not new training','historical_seed27_fno2d':'Three candidates require verification; no seed27 retraining authorized','stage2':{'seeds':[89,131],'status':'RESERVED_NOT_STARTED','automatic_start':False},'session_root':str(root),'safe_stop_utc':config['safe_stop_utc'],'resume_policy':['Never SIGCONT the frozen old supervisor or relaunch stage1_supervisor_v2.py','Verify new container runtime, GPU, NFS mount and source/data hashes first','Use this manifest and refreshed result identities to skip every COMPLETED unit','For training-complete units with missing evaluation, evaluate BEST only; do not retrain','For incomplete units validate checkpoint, optimizer, scheduler and RNG before resume; preserve original directories and outputs','Create a new operational deadline/launch manifest; the old runner cutoff is expired in a later container and must not be silently reused','Only seeds 41 and 73 may launch; FNO2D verification gates remain; Stage 2 requires new user instructions'],**(extra or {})}
 save(pause/'resume-manifest.json',manifest)
 q=read(root/'queue-status.json');q.update(status=phase,automatic_next_unit=False,active=manifest['active_unit'],completed=[r['id'] for r in rows if r['status']=='COMPLETED'],pending=[r['id'] for r in rows if r['status']=='PENDING'],running=[r['id'] for r in rows if r['status']=='RUNNING'],failed=[r['id'] for r in rows if r['status']=='FAILED'],counts=counts,resume_manifest=str(pause/'resume-manifest.json'),updated_utc=utc().isoformat())
 save(root/'queue-status.json',q);return manifest

def inventory(root,pause):
 entries=[]
 for folder in ['runs','evaluations','logs','control']:
  for p in sorted((root/folder).rglob('*')):
   if p.is_symlink():raise ValueError('Unexpected symlink: '+str(p))
   if p.is_file():
    with p.open('rb') as f:os.fsync(f.fileno())
    entries.append({'path':str(p),'bytes':p.stat().st_size})
 for name in ['stage1-statistics.json','supervisor-v2.log','queue-status.json']:
  p=root/name
  if p.exists():
   with p.open('rb') as f:os.fsync(f.fileno())
   entries.append({'path':str(p),'bytes':p.stat().st_size})
 save(pause/'nfs-output-inventory.json',{'utc':utc().isoformat(),'fsync_completed':True,'file_count':len(entries),'files':entries})
 return len(entries)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pause-dir',type=Path,required=True);args=p.parse_args()
 root=args.root;pause=args.pause_dir;config=read(root/'control/protocol.json');request=read(pause/'pause-request.json');deadline=datetime.datetime.fromisoformat(config['safe_stop_utc']);pid=request['active_train_pid']
 if request['active_unit']!='transformer_t1200_seed41' or config['launch_seeds']!=[41,73]:raise ValueError('Unexpected authorized unit or seeds')
 for name,digest in read(root/'control/execution-hashes-v2.json').items():
  if sha(root/'control'/name)!=digest:raise ValueError('Frozen execution file changed: '+name)
 initial=proc(pid);scheduler=proc(request['paused_scheduler_pid'])
 if not scheduler or scheduler['state']!='T':raise ValueError('Original scheduler is not frozen')
 if initial and initial['state'] not in ['Z','X'] and (initial['ppid']!=request['paused_scheduler_pid'] or request['active_unit'] not in initial['command']):raise ValueError('Active process identity mismatch')
 save(pause/'drain-status.json',{'status':'WAITING_CURRENT_TRAINING','pid':os.getpid(),'sid':os.getsid(0),'utc':utc().isoformat(),'training_pid':pid,'safe_stop_utc':config['safe_stop_utc'],'automatic_next_unit':False})
 persist(root,pause,config,request,'PAUSE_AFTER_CURRENT_REQUESTED',bool(initial and initial['state'] not in ['Z','X']))
 while True:
  current=proc(pid)
  if not current or current['state'] in ['Z','X']:break
  if current['start_ticks']!=initial['start_ticks']:raise ValueError('PID reuse detected')
  if utc()>=deadline:
   persist(root,pause,config,request,'PAUSED_DEADLINE_GUARD_ACTIVE',True,{'note':'Existing runner/timeout safety mechanisms own training stop; drain sends no signal to training'})
   return
  time.sleep(20)
 rows=persist(root,pause,config,request,'CURRENT_TRAINING_EXITED',False)['units'];unit=next(x for x in rows if x['id']==request['active_unit'])
 if unit['status']!='COMPLETED' and unit.get('training_complete') and unit['status']!='FAILED' and (deadline-utc()).total_seconds()>180:
  save(pause/'drain-status.json',{'status':'EVALUATING_CURRENT_BEST','pid':os.getpid(),'sid':os.getsid(0),'utc':utc().isoformat(),'automatic_next_unit':False})
  persist(root,pause,config,request,'FINISHING_CURRENT_EVALUATION',True)
  command=[sys.executable,'-B','-u',str(root/'control/stage1_runner.py'),'eval','--unit',request['active_unit']]
  with (root/'logs'/(request['active_unit']+'-eval.log')).open('x') as log:
   child=subprocess.Popen(command,cwd=config['project_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
   code=child.wait()
  print(json.dumps({'event':'CURRENT_EVALUATION_EXITED','returncode':code}),flush=True)
 subprocess.run([sys.executable,'-B',str(root/'control/stage1_statistics_v2.py')],cwd=root/'control',check=True)
 final=persist(root,pause,config,request,'PAUSED_AFTER_CURRENT',False)
 count=inventory(root,pause)
 save(pause/'drain-status.json',{'status':'COMPLETE_PAUSED','pid':os.getpid(),'utc':utc().isoformat(),'counts':final['counts'],'nfs_files_fsynced':count,'automatic_next_unit':False,'stage2':'RESERVED_NOT_STARTED'})
 print(json.dumps({'status':'COMPLETE_PAUSED','counts':final['counts']}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  print(json.dumps({'status':'DRAIN_REQUIRES_REVIEW','error':str(exc),'automatic_next_unit':False}),flush=True)
  raise