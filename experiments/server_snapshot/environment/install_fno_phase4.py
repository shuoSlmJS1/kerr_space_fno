"""Bounded CPU-container deployment using the verified Phase 2 utility module."""
import argparse
import concurrent.futures
import hashlib
import html
import http.client
import json
import os
from pathlib import Path
import re
import signal
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import installer_common as common

ROOT = Path('/workspace/shanjinshuo')
RUN = Path(__file__).resolve().parent
PREFIX = ROOT / 'environments/fno_kerr'
CACHE = RUN / 'conda-cache'
WHEELS = RUN / 'wheelhouse'
GIB = 1024 ** 3
PINS = {
    'torch': '2.5.1+cu121', 'numpy': '2.0.1', 'scipy': '1.15.3',
    'matplotlib': '3.10.8', 'PyYAML': '6.0.3', 'plotly': '6.9.0',
    'narwhals': '2.24.0', 'triton': '3.1.0', 'sympy': '1.13.1',
    'filelock': '3.20.3', 'typing-extensions': '4.15.0', 'networkx': '3.4.2',
    'Jinja2': '3.1.6', 'MarkupSafe': '3.0.2', 'fsspec': '2026.2.0',
    'mpmath': '1.3.0', 'contourpy': '1.3.1', 'cycler': '0.12.1',
    'fonttools': '4.62.1', 'kiwisolver': '1.4.9', 'packaging': '26.0',
    'Pillow': '12.1.1', 'pyparsing': '3.2.5', 'python-dateutil': '2.9.0.post0',
    'six': '1.17.0', 'nvidia-cuda-nvrtc-cu12': '12.1.105',
    'nvidia-cuda-runtime-cu12': '12.1.105', 'nvidia-cuda-cupti-cu12': '12.1.105',
    'nvidia-cudnn-cu12': '9.1.0.70', 'nvidia-cublas-cu12': '12.1.3.1',
    'nvidia-cufft-cu12': '11.0.2.54', 'nvidia-curand-cu12': '10.3.2.106',
    'nvidia-cusolver-cu12': '11.4.5.107', 'nvidia-cusparse-cu12': '12.1.0.106',
    'nvidia-nccl-cu12': '2.21.5', 'nvidia-nvtx-cu12': '12.1.105',
    'nvidia-nvjitlink-cu12': '12.1.105'
}


def allocated(paths):
    result = subprocess.check_output(['du', '-s', '-B1', *map(str, paths)], text=True, timeout=90)
    return sum(int(line.split()[0]) for line in result.splitlines())


def footprint():
    return allocated([ROOT / 'environments', ROOT / 'environment-locks', ROOT / 'logs'])


def fingerprint(python):
    out = subprocess.check_output([str(python), '-B', '-c',
        "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"],
        env=ENV, text=True, timeout=45)
    return hashlib.sha256(out.encode()).hexdigest()


def select(pair, tags):
    name, version = pair
    if name != 'torch':
        artifact = common.select_artifact(pair, tags)
        info = common.fetch_json('https://pypi.org/pypi/' + name + '/' + version + '/json')['info']
        artifact['requires_dist'] = info.get('requires_dist') or []
        artifact['requires_python'] = info.get('requires_python') or ''
        return artifact
    index = 'https://download.pytorch.org/whl/cu121/torch/'
    with urllib.request.urlopen(index, timeout=30) as response:
        document = response.read(4 * 1024 * 1024).decode()
    expected = 'torch-2.5.1+cu121-cp310-cp310-linux_x86_64.whl'
    links = [html.unescape(x) for x in re.findall('href="([^"]+)"', document)]
    links = [x for x in links if urllib.parse.unquote(x.split('#')[0].rsplit('/', 1)[-1]) == expected]
    if len(links) != 1:
        raise RuntimeError('Ambiguous official torch wheel')
    indexed = urllib.parse.urljoin(index, links[0])
    checksum = urllib.parse.parse_qs(urllib.parse.urlsplit(indexed).fragment).get('sha256', [''])[0]
    if not re.fullmatch('[0-9a-f]{64}', checksum):
        raise RuntimeError('Official wheel SHA256 missing')
    url = indexed.split('#')[0].replace('https://download-r2.pytorch.org/', 'https://download.pytorch.org/')
    if urllib.parse.urlsplit(url).hostname != 'download.pytorch.org':
        raise RuntimeError('Unexpected torch wheel host')
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=30) as response:
        cr = response.headers.get('Content-Range', '')
        if response.status != 206 or not cr.startswith('bytes 0-0/'):
            raise RuntimeError('Official wheel size probe failed')
        response.read(1)
        size = int(cr.split('/')[-1])
    return dict(name=name, version=version, filename=expected, url=url,
                sha256=checksum, size=size, official_index_link=indexed)


def download(artifact):
    # 只恢复本轮缓存；磁盘和 NFS 写入错误不归类为网络重试。
    target = WHEELS / artifact['filename']
    partial = target.with_name(target.name + '.part')
    for path in (target, partial):
        if path.is_symlink():
            raise RuntimeError('Wheel symlink refused')
    if target.exists():
        if target.stat().st_size != artifact['size'] or common.digest(target) != artifact['sha256']:
            raise RuntimeError('Existing wheel integrity failed')
        common.event('wheel_reused', package=artifact['name'], bytes=artifact['size'])
        return
    started = time.monotonic()
    transferred = 0
    max_attempts = 5
    network_errors = (TimeoutError, ConnectionError, urllib.error.URLError,
                      http.client.IncompleteRead, ssl.SSLEOFError)
    for attempt in range(max_attempts):
        common.require_time(3600)
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > artifact['size']:
            raise RuntimeError('Oversized partial wheel')
        if offset == artifact['size']:
            break
        common.event('downloading', package=artifact['name'], received_bytes=offset,
                     total_bytes=artifact['size'], attempt=attempt + 1)
        retry_reason = None
        try:
            response = urllib.request.urlopen(urllib.request.Request(artifact['url'],
                headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'}), timeout=90)
        except urllib.error.HTTPError as exc:
            # 只重试临时服务错误；认证、权限和无效 Range 不反复请求。
            exc.close()
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            retry_reason = type(exc).__name__ + ': ' + str(exc)
        except network_errors as exc:
            retry_reason = type(exc).__name__ + ': ' + str(exc)
        else:
            with response:
                cr = response.headers.get('Content-Range', '')
                encoding = response.headers.get('Content-Encoding', 'identity')
                if encoding != 'identity':
                    raise RuntimeError('Encoded response refused for byte-range recovery')
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', cr)
                    if not match:
                        raise RuntimeError('Malformed Content-Range; partial file preserved')
                    begin, end, total = map(int, match.groups())
                    if begin != offset or total != artifact['size'] or not begin <= end < total:
                        raise RuntimeError('Unexpected Content-Range; partial file preserved')
                    expected_body = end - begin + 1
                    mode = 'ab'
                elif response.status == 200:
                    # 完整响应只能覆盖本文件的未完成片段，绝不能追加到旧偏移。
                    if cr:
                        raise RuntimeError('Unexpected Content-Range on full response')
                    expected_body = artifact['size']
                    mode = 'wb'
                else:
                    raise RuntimeError('Unexpected download response status')
                length = response.headers.get('Content-Length')
                if length is not None and (not length.isdigit() or int(length) != expected_body):
                    raise RuntimeError('Unexpected Content-Length; partial file preserved')
                if response.status == 200 and offset:
                    common.event('range_fallback_restart_partial', package=artifact['name'],
                                 previous_partial_bytes=offset)
                if response.status == 200:
                    offset = 0
                received = 0
                last = time.monotonic()
                with partial.open(mode) as stream:
                    try:
                        while True:
                            common.require_time(2700)
                            # 网络异常仅在读取范围内捕获；文件写入异常必须立即停止。
                            try:
                                chunk = response.read(min(1024 * 1024, expected_body - received + 1))
                            except network_errors as exc:
                                retry_reason = type(exc).__name__ + ': ' + str(exc)
                                break
                            if not chunk:
                                break
                            if received + len(chunk) > expected_body or offset + len(chunk) > artifact['size']:
                                raise RuntimeError('Download exceeds declared range or size')
                            stream.write(chunk)
                            offset += len(chunk)
                            received += len(chunk)
                            transferred += len(chunk)
                            if time.monotonic() - last >= 30:
                                stream.flush()
                                common.event('downloading', package=artifact['name'], received_bytes=offset,
                                    total_bytes=artifact['size'], MiB_per_second=round(
                                        transferred / 1048576 / (time.monotonic() - started), 3))
                                last = time.monotonic()
                    finally:
                        stream.flush()
                        os.fsync(stream.fileno())
                if not retry_reason and (received != expected_body or offset != artifact['size']):
                    retry_reason = 'Incomplete response; verified offset will be resumed'
        if retry_reason is None and offset == artifact['size']:
            break
        if attempt + 1 == max_attempts:
            raise RuntimeError('Download retry limit reached; partial preserved: ' + str(retry_reason))
        delay = min(5 * (2 ** attempt), 40)
        common.require_time(3600 + delay)
        common.event('download_retry', package=artifact['name'], attempt=attempt + 1,
                     max_attempts=max_attempts, reason=retry_reason, backoff_seconds=delay,
                     retained_bytes=partial.stat().st_size if partial.exists() else 0)
        time.sleep(delay)
    if partial.stat().st_size != artifact['size'] or common.digest(partial) != artifact['sha256']:
        raise RuntimeError('Downloaded wheel size or SHA256 mismatch')
    partial.rename(target)
    common.event('wheel_verified', package=artifact['name'], bytes=artifact['size'],
                 seconds=round(time.monotonic() - started, 2))


def validate_dependencies(artifacts, python):
    # 使用已下载 wheel 的真实元数据进行离线解析，不依赖 PyPI 汇总信息。
    requirement_text = ''.join(a['name'] + '==' + a['version'] + ' --hash=sha256:' + a['sha256'] + '\n' for a in artifacts)
    (RUN / 'requirements-hashed.txt').write_text(requirement_text)
    argv = [python, '-B', '-m', 'pip', 'install', '--no-index', '--find-links', str(WHEELS),
            '--only-binary=:all:', '--require-hashes', '--no-compile', '-r', str(RUN / 'requirements-hashed.txt')]
    common.command('pip-resolve', argv + ['--dry-run', '--report', str(RUN / 'pip-resolve.json')], timeout=180)
    return argv


def validate_metadata(artifacts, python):
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.specifiers import SpecifierSet
    from pip._vendor.packaging.utils import canonicalize_name
    installed = json.loads(subprocess.check_output([python, '-B', '-c',
        "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"],
        env=ENV, text=True, timeout=30))
    available = {canonicalize_name(k): v for k, v in installed.items()}
    available.update({canonicalize_name(a['name']): a['version'] for a in artifacts})
    markers = dict(default_environment(), python_version='3.10', python_full_version='3.10.20', extra='')
    errors = []
    for artifact in artifacts:
        required_python = artifact.get('requires_python', '')
        if required_python and not SpecifierSet(required_python).contains('3.10.20'):
            errors.append(artifact['name'] + ' Python constraint ' + required_python)
        for text in artifact.get('requires_dist', []):
            req = Requirement(text)
            if req.marker and not req.marker.evaluate(markers):
                continue
            version = available.get(canonicalize_name(req.name))
            if version is None or not req.specifier.contains(version, prereleases=True):
                errors.append(artifact['name'] + ' requires ' + text + '; selected=' + str(version))
    common.save('metadata-resolution.json', {'status': 'failed' if errors else 'passed', 'errors': errors,
        'note': 'PyPI metadata precheck. Torch CUDA pins follow the audited official wheel. Offline wheel resolution remains mandatory.'})
    if errors:
        raise RuntimeError('Dependency metadata conflict: ' + '; '.join(errors))


def main():
    global ENV
    common.require_time(10800)
    for path in (ROOT, ROOT / 'environments', RUN):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError('Unexpected deployment path')
    for path in (CACHE, WHEELS, RUN / 'tmp'):
        if path.is_symlink():
            raise RuntimeError('Cache symlink refused')
        path.mkdir(exist_ok=True)
    ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        CONDA_PKGS_DIRS=str(CACHE), CONDA_ENVS_PATH=str(ROOT / 'environments'),
        CONDA_REMOTE_CONNECT_TIMEOUT_SECS='10', CONDA_REMOTE_READ_TIMEOUT_SECS='30',
        CONDA_REMOTE_MAX_RETRIES='2', CONDA_REPORT_ERRORS='false',
        PIP_CONFIG_FILE='/dev/null', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1',
        TMPDIR=str(RUN / 'tmp'), MPLCONFIGDIR=str(RUN / 'matplotlib-config'),
        MPLBACKEND='Agg', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        LC_ALL='C', LANG='C', CUDA_VISIBLE_DEVICES='')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL'):
        ENV.pop(key, None)
    common.ENV = ENV
    protected = {'base': Path('/opt/conda/bin/python'), 'underwater': ROOT / 'environments/underwater/bin/python'}
    before = {name: fingerprint(path) for name, path in protected.items()}
    previous = RUN / 'protected-before.json'
    if previous.exists() and json.loads(previous.read_text()) != before:
        raise RuntimeError('Protected environment changed since original launch')
    common.save('protected-before.json', before)
    common.save('target-pins.json', PINS)
    baseline = footprint()
    if not PREFIX.exists() and baseline + 19 * GIB > 30 * GIB:
        raise RuntimeError('Combined 30 GiB budget lacks 19 GiB conservative headroom')
    common.save('budget-before.json', {'managed_bytes': baseline, 'total_budget_bytes': 30 * GIB,
        'scope': 'Both environments, environment-locks and logs including caches', 'planned_additional_max_GiB': 19})
    ready = RUN / 'conda-ready.json'
    if PREFIX.is_symlink():
        raise RuntimeError('Prefix symlink refused')
    if not PREFIX.exists():
        if ready.exists():
            raise RuntimeError('Recorded prefix is missing')
        common.command('conda-create', ['/opt/conda/bin/conda', 'create', '--yes', '--copy',
            '--override-channels', '-c', 'https://repo.anaconda.com/pkgs/main', '--solver', 'libmamba',
            '--prefix', str(PREFIX), 'python=3.10.20', 'pip'], timeout=1200)
        common.save('conda-ready.json', {'prefix': str(PREFIX), 'utc': common.utc()})
    elif not ready.exists():
        raise RuntimeError('Unknown or interrupted prefix; manual inspection required')
    python = str(PREFIX / 'bin/python')
    tags = subprocess.check_output([python, '-B', '-c',
        "import sys,json; from pip._vendor.packaging.tags import sys_tags; assert sys.version_info[:3]==(3,10,20); print(json.dumps([str(t) for t in sys_tags()]))"],
        env=ENV, text=True, timeout=30)
    ranks = {tag: i for i, tag in enumerate(json.loads(tags))}
    manifest_path = RUN / 'wheel-manifest.json'
    if manifest_path.exists():
        artifacts = json.loads(manifest_path.read_text())
    else:
        common.event('resolving_wheel_manifest', count=len(PINS))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            artifacts = list(pool.map(lambda item: select(item, ranks), PINS.items()))
        common.save('wheel-manifest.json', artifacts)
    validate_metadata(artifacts, python)
    total = sum(a['size'] for a in artifacts)
    if total > 4.5 * GIB or allocated([CACHE]) > 1.5 * GIB:
        raise RuntimeError('Cache planning cap exceeded')
    common.save('download-budget.json', {'wheel_bytes': total, 'cache_cap_GiB': 6, 'prefix_cap_GiB': 12})
    for artifact in sorted(artifacts, key=lambda item: item['size'], reverse=True):
        download(artifact)
    raw = 0
    for artifact in artifacts:
        with zipfile.ZipFile(WHEELS / artifact['filename']) as wheel:
            raw += sum(((item.file_size + 4095) // 4096) * 4096 for item in wheel.infolist())
    predicted_prefix = allocated([PREFIX]) + int(raw * 1.15)
    if predicted_prefix > 12 * GIB or footprint() + int(raw * 1.15) + 2 * GIB > 30 * GIB:
        raise RuntimeError('Unpacked wheel estimate exceeds environment or combined budget')
    argv = validate_dependencies(artifacts, python)
    common.require_time(2700)
    common.command('pip-install', argv + ['--report', str(RUN / 'pip-install-report.json')], timeout=2100)
    common.command('pip-check', [python, '-B', '-m', 'pip', 'check'], timeout=60)
    common.command('cpu-smoke', [python, '-B', str(RUN / 'smoke_fno_cpu.py')], timeout=120)
    common.command('python-version', [python, '--version'], timeout=30)
    common.command('pip-freeze', [python, '-B', '-m', 'pip', 'freeze', '--all'], timeout=60)
    common.command('conda-list', ['/opt/conda/bin/conda', 'list', '--prefix', str(PREFIX), '--json'], timeout=60)
    common.command('conda-explicit', ['/opt/conda/bin/conda', 'list', '--prefix', str(PREFIX), '--explicit'], timeout=60)
    after = {name: fingerprint(path) for name, path in protected.items()}
    common.save('protected-after.json', after)
    if before != after:
        raise RuntimeError('Protected environment package fingerprint changed')
    size, combined, cache_bytes = allocated([PREFIX]), footprint(), allocated([CACHE, WHEELS])
    if size > 12 * GIB or combined > 30 * GIB or cache_bytes > 6 * GIB:
        raise RuntimeError('Final storage budget exceeded')
    result = dict(status='completed_cpu_only', utc=common.utc(), prefix=str(PREFIX),
        elapsed_seconds=round(time.time() - STARTED, 2), environment_allocated_bytes=size,
        managed_total_bytes=combined, cache_bytes=cache_bytes, protected_package_fingerprints_unchanged=True,
        cpu_smoke=json.loads((RUN / 'cpu-smoke.json').read_text()), cuda_tensor_test='not_run_cpu_container',
        gpu_validation='pending', cross_container_reuse='not_tested', checkpoint_loading='not_tested',
        activation='source /opt/conda/etc/profile.d/conda.sh && conda activate ' + str(PREFIX),
        formal_training=False)
    common.save('completion.json', result)
    common.event('completed_cpu_only', **{k: v for k, v in result.items() if k not in ('status', 'utc')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    ARGS = parser.parse_args()
    STARTED = time.time()
    common.RUN, common.CACHE, common.WHEELS, common.PREFIX, common.ARGS = RUN, CACHE, WHEELS, PREFIX, ARGS
    def terminate(signum, frame):
        raise RuntimeError('Safety deadline or termination signal received')
    signal.signal(signal.SIGTERM, terminate)
    try:
        import fcntl
        with (RUN / 'installer.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main()
    except BaseException as exc:
        common.event('stopped', error_type=type(exc).__name__, reason=str(exc),
                     elapsed_seconds=round(time.time() - STARTED, 2))
        raise
