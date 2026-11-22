import datetime,fcntl,hashlib,json,os,signal,subprocess,sys
from pathlib import Path
W=Path(__file__).resolve().parent;C=json.loads((W/'window.json').read_text());R=Path(C['session_root']);D=datetime.datetime.fromisoformat(C['safe_stop_utc']);STOP=False

def now():return datetime.datetime.now(datetime.timezone.utc)
def save(s):
 s['updated_utc']=now().isoformat();s['counts']={k:sum(u['status']==k for u in s['units']) for k in ['COMPLETED','RUNNING','PENDING','FAILED','BLOCKED']}
 p=W/'queue-status.json';t=W/('status.tmp.'+str(os.getpid()))
 with t.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def stop(*args):
 global STOP
 STOP=True

def main():
 for name,sha in json.loads((W/'execution-hashes.json').read_text()).items():
  assert hashlib.sha256((W/name).read_bytes()).hexdigest()==sha
 with (R/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  s={'status':'RUNNING','pid':os.getpid(),'sid':os.getsid(0),'started_utc':now().isoformat(),'safe_stop_utc':D.isoformat(),'active':None,'units':C['units'],'stage2':'RESERVED_NOT_STARTED','statistics_analysis_enabled':False};save(s)
  for u in s['units']:
   if u['status']!='PENDING':continue
   assert u['seed'] in [41,73] and u['family']=='track_a'
   if STOP or (D-now()).total_seconds()<300:
    s['status']='SAFE_STOP_PENDING';break
   run=R/'runs'/u['id'];ev=R/'evaluations'/u['id']
   if run.exists() or ev.exists():
    s.update(status='STOPPED_UNEXPECTED_EXISTING_OUTPUT',reason=u['id']);break
   s['active']=u['id'];u['status']='RUNNING';save(s)
   for mode in ['train','eval']:
    s['active_mode']=mode;save(s)
    with (W/(u['id']+'-'+mode+'.log')).open('x') as log:
     p=subprocess.Popen([sys.executable,'-B','-u',str(W/'window_runner.py'),mode,'--unit',u['id']],cwd=C['session_root'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
     s['child_pid']=p.pid;save(s)
     try:code=p.wait(timeout=max(1,(D-now()).total_seconds()))
     except subprocess.TimeoutExpired:
      p.terminate()
      try:code=p.wait(timeout=60)
      except subprocess.TimeoutExpired:p.kill();p.wait();code=75
    if code:
     u.update(status='PENDING' if code==75 else 'FAILED',exit_code=code,phase=mode)
     checkpoints=sorted(run.glob('epoch_*.pt'));u['last_checkpoint']=str(checkpoints[-1]) if checkpoints else None
     s['status']='SAFE_STOP_PENDING' if code==75 else 'STOPPED_REQUIRES_REVIEW';save(s);break
   if code:break
   summary=json.loads((run/'summary.json').read_text());matrix=json.loads((ev/'matrix.json').read_text())
   assert summary['completed_epochs']==500 and summary['training']['seed']==u['seed']
   assert summary['best_checkpoint_sha256']==matrix['checkpoint_sha256']
   u.update(status='COMPLETED',training_wall_seconds=summary['training_wall_seconds'],best_checkpoint_sha256=summary['best_checkpoint_sha256']);s['active']=None;save(s)
  else:s['status']='TRACK_A_COMPLETE_FNO2D_BLOCKED'
  s['active']=None;s['finished_utc']=now().isoformat();save(s)
  print(json.dumps({'status':s['status'],'counts':s['counts']}),flush=True)
if __name__=='__main__':main()
