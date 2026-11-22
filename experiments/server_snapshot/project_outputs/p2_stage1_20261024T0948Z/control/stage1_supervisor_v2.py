import datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'protocol.json').read_text())
SESSION=Path(CONFIG['session_root'])
DEADLINE=datetime.datetime.fromisoformat(CONFIG['safe_stop_utc'])
STOP=False
CHILD=None

def utc():return datetime.datetime.now(datetime.timezone.utc)
def save(data):
 path=SESSION/'queue-status.json';temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with temp.open('x') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def stop(*args):
 global STOP
 STOP=True
 if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
def run(command,log):
 global CHILD
 with log.open('x') as f:
  CHILD=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=CONFIG['project_root'],env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
  try:return CHILD.wait(timeout=max(1,(DEADLINE-utc()).total_seconds()))
  except subprocess.TimeoutExpired:
   CHILD.terminate()
   try:CHILD.wait(timeout=60)
   except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
   return 75
  finally:CHILD=None

def main():
 if CONFIG['launch_seeds']!=[41,73] or CONFIG['stage2_automatic_start'] is not False:raise ValueError('Stage seed guard failed')
 for name in ['preflight-result.json','tests-result.json']:
  if json.loads((HERE/name).read_text())['status']!='PASS':raise ValueError('Preflight or tests not passed')
 for path,sha in json.loads((HERE/'execution-hashes-v2.json').read_text()).items():
  if hashlib.sha256((HERE/path).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen execution source changed')
 with (SESSION/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  prior=json.loads((SESSION/'queue-status.json').read_text())
  if prior['status']!='STOPPED_STATISTICS_ERROR':raise ValueError('Expected stopped statistics incident')
  with (SESSION/'queue-status-before-statistics-fix.json').open('x') as f:json.dump(prior,f,indent=2)
  completed=list(prior['completed'])
  state={'status':'RUNNING','pid':os.getpid(),'sid':os.getsid(0),'started_utc':utc().isoformat(),'safe_stop_utc':DEADLINE.isoformat(),'active':None,'completed':completed,'pending':[u['id'] for u in CONFIG['run_units'] if u['id'] not in completed],'stage2':'RESERVED_NOT_QUEUED','historical_gpu_hours_for_30_new_runs':23.350767055,'eta_scope':'Per-run measured recent-epoch ETA in unit status; total current-device ETA unknown until representative measurements'}
  (SESSION/'logs').mkdir(exist_ok=True)
  for unit in CONFIG['run_units']:
   if not unit['enabled'] or unit['id'] in completed:continue
   if unit['seed'] not in [41,73]:raise ValueError('Disallowed seed')
   remaining=(DEADLINE-utc()).total_seconds()
   estimate=unit['historical_seconds']*2.0+300
   if STOP or remaining<estimate:
    state.update(status='SAFE_STOP_PENDING',reason='Insufficient remaining window for conservative two-times historical run plus evaluation allowance',next_unit=unit['id'],remaining_seconds=remaining,estimated_required_seconds=estimate);save(state);break
   state.update(active=unit['id'],status='RUNNING',updated_utc=utc().isoformat());save(state)
   for mode in ['train','eval']:
    state['active_mode']=mode;save(state)
    code=run([sys.executable,'-B','-u',str(HERE/'stage1_runner.py'),mode,'--unit',unit['id']],SESSION/'logs'/(unit['id']+'-'+mode+'.log'))
    if code:
     state.update(status='SAFE_STOP_PENDING' if code==75 else 'STOPPED_REQUIRES_REVIEW',exit_code=code,reason='Child stopped; outputs retained; no automatic protocol adjustment');save(state);break
   if code:break
   state['completed'].append(unit['id']);state['pending'].remove(unit['id']);state['active']=None;save(state)
   result=subprocess.run([sys.executable,'-B',str(HERE/'stage1_statistics_v2.py')],cwd=HERE,capture_output=True,text=True)
   if result.returncode:state.update(status='STOPPED_STATISTICS_ERROR',reason=result.stderr);save(state);break
  else:state.update(status='PENDING_VERIFICATION' if state['pending'] else 'STAGE1_COMPLETE')
  state['finished_utc']=utc().isoformat();state['active']=None;save(state)
  subprocess.run([sys.executable,'-B',str(HERE/'stage1_statistics_v2.py')],cwd=HERE,check=True)
  print(json.dumps(state),flush=True)
if __name__=='__main__':main()