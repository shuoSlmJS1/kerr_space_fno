"""Run one bounded FNO condition and preserve complete process evidence."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--run-dir', required=True)
p.add_argument('--name', required=True)
p.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
p.add_argument('--batch', type=int, choices=[1, 32], required=True)
p.add_argument('--disable-cudnn-tf32', action='store_true')
a = p.parse_args()
r = Path(a.run_dir)
assert r.parent == Path('/workspace/shanjinshuo/logs') and not r.is_symlink()
assert a.name.replace('_', '').isalnum()
now = datetime.datetime.now(datetime.timezone.utc)
# 沿用用户报告的同一任务到期时间，并预留十五分钟；新容器不能沿用。
assert subprocess.check_output(['hostname'], text=True).strip() == 'job-l480b340-gxchp'
assert now < datetime.datetime.fromisoformat('2026-10-11T04:08:00+00:00')
if a.device.startswith('cuda'):
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=True)
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=True)
    assert not processes.stdout.strip(), 'GPU compute occupancy detected'
    with (r / (a.name + '.preflight.json')).open('x') as f:
        json.dump({'utc': now.isoformat(), 'gpu_stdout': gpu.stdout, 'gpu_stderr': gpu.stderr,
                   'process_stdout': processes.stdout, 'process_stderr': processes.stderr,
                   'memory_note': 'HAMI allocation remains platform-defined; physical query is not a quota.'}, f, indent=2)
env = os.environ.copy()
env.update({'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
            'MPLCONFIGDIR': str(r / 'matplotlib'), 'XDG_CACHE_HOME': str(r / 'cache')})
cmd = ['/workspace/shanjinshuo/environments/fno_kerr/bin/python', '-B', str(r / 'diagnose_fno_numerics.py'),
       '--run-dir', str(r), '--name', a.name, '--device', a.device, '--batch', str(a.batch)]
if a.disable_cudnn_tf32:
    cmd.append('--disable-cudnn-tf32')
with (r / (a.name + '.command.json')).open('x') as f:
    json.dump({'argv': cmd, 'timeout_seconds': 60, 'utc': now.isoformat()}, f, indent=2)
started = time.monotonic()
timeout = False
with (r / (a.name + '.stdout.log')).open('x') as out, (r / (a.name + '.stderr.log')).open('x') as err:
    try:
        proc = subprocess.run(cmd, env=env, cwd=r, stdout=out, stderr=err, timeout=60)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        # subprocess.run 只终止本次诊断子进程，不影响其他任务。
        code, timeout = 124, True
stderr = (r / (a.name + '.stderr.log')).read_text()
errors = [line for line in stderr.splitlines() if 'cuCtxSetCurrent111 failed res=4' in line]
last_atexit = stderr.rfind('python_atexit_before_torch')
positions = [stderr.find(line) for line in errors]
summary = {'name': a.name, 'exit_code': code, 'timeout': timeout,
           'wall_seconds': time.monotonic() - started, 'hami_error_count': len(errors),
           'hami_only_after_python_atexit': bool(errors) and last_atexit >= 0 and all(i > last_atexit for i in positions),
           'stderr_sha256': hashlib.sha256(stderr.encode()).hexdigest()}
result = r / (a.name + '.json')
if result.exists():
    data = json.loads(result.read_text())
    summary['metrics'] = data['comparison_index0_to_history']
    summary['precision'] = data['precision']
with (r / (a.name + '.process.json')).open('x') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary), flush=True)
raise SystemExit(code)
