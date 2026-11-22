"""One short isolated P1 preflight subprocess with a fixed task deadline."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['cpu','gpu'],required=True)
p.add_argument('--output-dir',required=True)
a=p.parse_args()
r=Path(a.output_dir)
assert r==Path('/workspace/shanjinshuo/outputs/fno_kerr/reviewer_p1_preflight_20261010')
assert socket.gethostname()=='job-l480b340-gxchp','Different container; revalidate deadline'
now=datetime.datetime.now(datetime.timezone.utc)
assert now+datetime.timedelta(seconds=120)<datetime.datetime.fromisoformat('2026-10-11T04:08:00+00:00')
if a.mode=='gpu':
    occ=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
    assert not occ.stdout.strip(),'GPU occupied; do not run'
    with (r/'gpu-preflight.json').open('x') as f:json.dump({'gpu':gpu.stdout,'processes':occ.stdout,'utc':now.isoformat()},f)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
         MPLCONFIGDIR=str(r/'matplotlib'),XDG_CACHE_HOME=str(r/'cache'))
cmd=['/workspace/shanjinshuo/environments/fno_kerr/bin/python','-B',str(r/'p1_preflight_smoke.py'),
     '--mode',a.mode,'--output-dir',str(r)]
with (r/(a.mode+'-command.json')).open('x') as f:json.dump({'argv':cmd,'timeout_seconds':120,'utc':now.isoformat()},f)
start=time.perf_counter()
with (r/(a.mode+'-stdout.log')).open('x') as out,(r/(a.mode+'-stderr.log')).open('x') as err:
    try:
        process=subprocess.run(cmd,cwd=r,env=env,stdout=out,stderr=err,timeout=120)
        code=process.returncode
    except subprocess.TimeoutExpired:
        code=124
stderr=(r/(a.mode+'-stderr.log')).read_text()
errors=[line for line in stderr.splitlines() if 'cuCtxSetCurrent111 failed res=4' in line]
pos=stderr.rfind('python_atexit_before_torch')
summary={'status':'verified' if code==0 else 'failed','exit_code':code,'wall_seconds':time.perf_counter()-start,
         'hami_error_count':len(errors),'hami_only_after_atexit':bool(errors) and pos>=0 and all(stderr.find(e)>pos for e in errors)}
with (r/(a.mode+'-process.json')).open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary),flush=True)
raise SystemExit(code)
