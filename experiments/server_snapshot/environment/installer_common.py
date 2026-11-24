"""Approved underwater installation; persistent artifacts and bounded execution."""
import argparse
import concurrent.futures
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path('/workspace/shanjinshuo')
PREFIX = ROOT / 'environments/underwater'
CACHE = ROOT / 'logs/phase1-conda-dry-run-1791607582912289384/metadata-cache'
CANDIDATE = ROOT / 'environment-locks/underwater-phase1-candidate-1791607696712183995.json'
RUN = ROOT / 'environment-locks/underwater-phase2-20261010T0546Z'
WHEELS = RUN / 'wheelhouse'
GIB = 1024 ** 3

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(name, obj):
    # 只替换本次安装的状态文件；历史事件另行追加。
    path = RUN / name
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def event(stage, **fields):
    row = dict(utc=utc(), stage=stage, pid=os.getpid(), **fields)
    print(json.dumps(row), flush=True)
    with (RUN / 'events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    save('status.json', row)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def remaining():
    return ARGS.deadline - time.time()

def require_time(seconds):
    if remaining() < seconds:
        raise RuntimeError('Insufficient time before the installation safety deadline')

def command(label, argv, timeout=1800, env=None):
    require_time(90)
    event(label, command=argv, remaining_seconds=int(remaining()))
    with (RUN / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=utc(), label=label, argv=argv)) + '\n')
    with (RUN / (label + '.log')).open('a') as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env or ENV,
                           timeout=min(timeout, max(1, remaining() - 45)))
    if p.returncode:
        raise RuntimeError(label + ' failed with exit code ' + str(p.returncode))

def fetch_json(url):
    require_time(120)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=20) as f:
                return json.load(f)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def select_artifact(pair, tags):
    from pip._vendor.packaging.utils import parse_wheel_filename
    name, version = pair
    if name in ('torch', 'torchvision', 'torchaudio'):
        url = 'https://download.pytorch.org/whl/cu126/' + name + '/'
        with urllib.request.urlopen(url, timeout=30) as f:
            text = f.read(2000000).decode()
        links = [html.unescape(s) for s in re.findall('href="([^"]+)"', text)
                 if name + '-' + version in s and 'cp311-cp311-manylinux_2_28_x86_64.whl' in s]
        if len(links) != 1:
            raise RuntimeError('Ambiguous official CUDA wheel: ' + name)
        indexed = links[0]
        checksum = urllib.parse.parse_qs(urllib.parse.urlsplit(indexed).fragment).get('sha256', [''])[0]
        if not re.fullmatch('[0-9a-f]{64}', checksum):
            raise RuntimeError('Missing official SHA256: ' + name)
        source = indexed.split('#')[0].replace('https://download-r2.pytorch.org/', 'https://download.pytorch.org/')
        # 大文件先只读 1 字节，从 Content-Range 获取完整长度。
        with urllib.request.urlopen(urllib.request.Request(source, headers={'Range': 'bytes=0-0'}), timeout=30) as f:
            f.read(1)
            cr = f.headers.get('Content-Range', '')
            if f.status != 206 or not cr.startswith('bytes 0-0/'):
                raise RuntimeError('Unexpected size probe response: ' + name)
            size = int(cr.split('/')[-1])
        return dict(name=name, version=version + '+cu126', filename=urllib.parse.unquote(source.rsplit('/', 1)[-1]),
                    url=source, sha256=checksum, size=size, official_index_link=indexed)
    data = fetch_json('https://pypi.org/pypi/' + name + '/' + version + '/json')
    ranked = []
    for w in data['urls']:
        if w['packagetype'] != 'bdist_wheel' or w.get('yanked'):
            continue
        _, _, _, wheel_tags = parse_wheel_filename(w['filename'])
        ranks = [tags[str(t)] for t in wheel_tags if str(t) in tags]
        if ranks:
            ranked.append((min(ranks), w))
    if not ranked:
        raise RuntimeError('No compatible binary wheel: ' + name)
    w = min(ranked, key=lambda item: item[0])[1]
    if urllib.parse.urlsplit(w['url']).hostname != 'files.pythonhosted.org':
        raise RuntimeError('Unexpected wheel host')
    return dict(name=name, version=version, filename=w['filename'], url=w['url'],
                sha256=w['digests']['sha256'], size=w['size'])

def download(a):
    dst = WHEELS / a['filename']
    partial = dst.with_name(dst.name + '.part')
    if dst.exists():
        if dst.stat().st_size == a['size'] and digest(dst) == a['sha256']:
            event('wheel_reused', package=a['name'], bytes=a['size'])
            return
        raise RuntimeError('Existing wheel failed integrity verification: ' + dst.name)
    if partial.is_symlink() or dst.is_symlink():
        raise RuntimeError('Symlink refused in wheel cache')
    start = time.monotonic()
    initial = partial.stat().st_size if partial.exists() else 0
    for attempt in range(3):
        require_time(3600)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > a['size']:
            raise RuntimeError('Oversized partial wheel')
        if offset == a['size']:
            break
        event('downloading', package=a['name'], received_bytes=offset, total_bytes=a['size'], attempt=attempt + 1)
        try:
            req = urllib.request.Request(a['url'], headers={'Range': 'bytes=' + str(offset) + '-'})
            with urllib.request.urlopen(req, timeout=30) as response:
                cr = response.headers.get('Content-Range', '')
                if response.status == 206:
                    if not cr.startswith('bytes ' + str(offset) + '-') or not cr.endswith('/' + str(a['size'])):
                        raise RuntimeError('Unexpected Content-Range')
                    mode = 'ab' if offset else 'wb'
                elif response.status == 200:
                    offset = 0
                    mode = 'wb'
                else:
                    raise RuntimeError('Unexpected download response')
                last = time.monotonic()
                with partial.open(mode) as f:
                    while True:
                        require_time(2700)
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                        if offset > a['size']:
                            raise RuntimeError('Download exceeds declared size')
                        if time.monotonic() - last > 30:
                            f.flush()
                            event('downloading', package=a['name'], received_bytes=offset,
                                  total_bytes=a['size'], MiB_per_second=round((offset-initial)/1048576/(time.monotonic()-start), 3))
                            last = time.monotonic()
                    f.flush()
                    os.fsync(f.fileno())
            if offset == a['size']:
                break
        except (OSError, urllib.error.URLError) as e:
            event('download_retry', package=a['name'], error=type(e).__name__)
            if attempt == 2:
                raise
            time.sleep(3)
    if partial.stat().st_size != a['size'] or digest(partial) != a['sha256']:
        raise RuntimeError('SHA256 or size mismatch: ' + a['name'])
    partial.rename(dst)
    event('wheel_verified', package=a['name'], bytes=a['size'], seconds=round(time.monotonic()-start, 2))

def base_fingerprint():
    p = subprocess.run(['/opt/conda/bin/python', '-B', '-c',
        "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"],
        env=ENV, capture_output=True, text=True, check=True, timeout=20)
    return hashlib.sha256(p.stdout.encode()).hexdigest()

def main():
    global ENV
    require_time(10800)
    for p in [ROOT, ROOT/'environments', RUN, CACHE]:
        if p.is_symlink() or not p.is_dir():
            raise RuntimeError('Unexpected deployment directory: ' + str(p))
    ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        CONDA_PKGS_DIRS=str(CACHE), CONDA_ENVS_PATH=str(ROOT/'environments'),
        CONDA_REMOTE_CONNECT_TIMEOUT_SECS='10', CONDA_REMOTE_READ_TIMEOUT_SECS='30',
        CONDA_REMOTE_MAX_RETRIES='2', CONDA_REPORT_ERRORS='false',
        PIP_CONFIG_FILE='/dev/null', PIP_DISABLE_PIP_VERSION_CHECK='1',
        PIP_NO_CACHE_DIR='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        YOLO_CONFIG_DIR=str(RUN/'ultralytics-config'), LC_ALL='C', LANG='C')
    ENV.pop('PYTHONPATH', None)
    baseline = base_fingerprint()
    save('base-before.json', {'package_fingerprint': baseline})
    if PREFIX.is_symlink():
        raise RuntimeError('Prefix symlink refused')
    ready = RUN/'conda-ready.json'
    if not PREFIX.exists():
        if ready.exists():
            raise RuntimeError('Recorded prefix is missing; inspect before recovery')
        command('conda-create', ['/opt/conda/bin/conda','create','--yes','--copy',
            '--override-channels','-c','https://repo.anaconda.com/pkgs/main','--solver','libmamba',
            '--prefix',str(PREFIX),'python=3.11.15','pip'], timeout=1200)
        save('conda-ready.json', {'prefix':str(PREFIX),'utc':utc()})
    elif not ready.exists():
        raise RuntimeError('Unfinished or unknown prefix; inspect conda log before recovery')
    python = str(PREFIX/'bin/python')
    result = subprocess.run([python,'-B','-c',
        "import sys,json; from pip._vendor.packaging.tags import sys_tags; assert sys.version_info[:3]==(3,11,15); print(json.dumps([str(t) for t in sys_tags()]))"],
        capture_output=True,text=True,env=ENV,check=True,timeout=30)
    tags = {t:i for i,t in enumerate(json.loads(result.stdout))}
    pins = json.loads(CANDIDATE.read_text())['pins']
    WHEELS.mkdir(exist_ok=True)
    manifest = RUN/'wheel-manifest.json'
    if manifest.exists():
        artifacts = json.loads(manifest.read_text())
    else:
        event('resolving_wheel_manifest', count=len(pins))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            artifacts = list(pool.map(lambda pair: select_artifact(pair,tags), pins.items()))
        save('wheel-manifest.json', artifacts)
    total = sum(a['size'] for a in artifacts)
    if total > 5 * GIB:
        raise RuntimeError('Wheel plan exceeds 5 GiB; cache budget is 6 GiB including Conda cache')
    save('download-budget.json', {'wheel_bytes':total, 'cache_budget_GiB':6, 'environment_budget_GiB':12})
    # 大包优先，启动后可尽快根据实际吞吐量估算剩余时间。
    for a in sorted(artifacts, key=lambda item:item['size'], reverse=True):
        download(a)
    cache_size = int(subprocess.check_output(['du','-s','-B1',str(CACHE)],text=True).split()[0])
    if total + cache_size > 6*GIB:
        raise RuntimeError('Cache budget exceeded')
    requirements = ''.join(a['name']+'=='+a['version']+' --hash=sha256:'+a['sha256']+'\n' for a in artifacts)
    (RUN/'requirements-hashed.txt').write_text(requirements)
    require_time(2700)
    command('pip-install', [python,'-B','-m','pip','install','--no-index','--find-links',str(WHEELS),
        '--only-binary=:all:','--require-hashes','--report',str(RUN/'pip-install-report.json'),
        '-r',str(RUN/'requirements-hashed.txt')],timeout=2100)
    command('pip-check',[python,'-B','-m','pip','check'],timeout=60)
    command('smoke',[python,'-B',str(RUN/'smoke_underwater.py')],timeout=180)
    command('python-version',[python,'--version'],timeout=30)
    command('pip-freeze',[python,'-B','-m','pip','freeze','--all'],timeout=60)
    command('conda-list',['/opt/conda/bin/conda','list','--prefix',str(PREFIX),'--json'],timeout=60)
    command('conda-explicit',['/opt/conda/bin/conda','list','--prefix',str(PREFIX),'--explicit'],timeout=60)
    size = int(subprocess.check_output(['du','-s','-B1',str(PREFIX)],text=True).split()[0])
    after = base_fingerprint()
    if after != baseline:
        raise RuntimeError('Base package fingerprint changed; review required')
    if size > 12*GIB:
        raise RuntimeError('Underwater exceeds planned 12 GiB; budget review required')
    summary = {'status':'completed','utc':utc(),'prefix':str(PREFIX),'environment_allocated_bytes':size,
        'elapsed_seconds':round(time.time()-STARTED,2),'base_package_fingerprint_unchanged':True,
        'cache_bytes':total+cache_size,'cross_container_reuse':'not_tested',
        'model_inference':'not_tested; no research assets migrated', 'formal_training':False,
        'activation':'source /opt/conda/etc/profile.d/conda.sh && conda activate '+str(PREFIX),
        'reuse_check':'On a new matching-image task run bin/python -m pip check and the saved smoke_underwater.py, compare versions and wheel manifest.'}
    save('completion.json',summary)
    event('completed',**{k:v for k,v in summary.items() if k not in ['utc','status']})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline',type=float,required=True)
    ARGS = parser.parse_args()
    STARTED = time.time()
    def stopped(signum, frame):
        raise RuntimeError('Installation stopped by deadline or termination signal')
    signal.signal(signal.SIGTERM,stopped)
    try:
        import fcntl
        with (RUN/'installer.lock').open('a+') as guard:
            fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            main()
    except BaseException as exc:
        event('stopped',error_type=type(exc).__name__,reason=str(exc),elapsed_seconds=round(time.time()-STARTED,2))
        raise
