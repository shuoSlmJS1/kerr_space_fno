"""Prepare a bounded resume; launch only with explicit --confirm-start."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

RUN = Path('/workspace/shanjinshuo/environment-locks/fno-phase4-20261010T0755Z')
PREFIX = Path('/workspace/shanjinshuo/environments/fno_kerr')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--confirm-start', action='store_true')
    args = parser.parse_args()
    deadline = datetime.datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None or deadline.utcoffset() != datetime.timedelta(0):
        raise RuntimeError('Provide an explicit UTC deadline ending in Z or +00:00')
    seconds = int(deadline.timestamp() - time.time())
    if seconds < 10800:
        raise RuntimeError('Less than the unchanged three-hour installer startup safety window remains')
    if RUN.is_symlink() or PREFIX.is_symlink() or not PREFIX.is_dir():
        raise RuntimeError('Unexpected run directory or prefix')
    ready = json.loads((RUN / 'conda-ready.json').read_text())
    if ready['prefix'] != str(PREFIX):
        raise RuntimeError('Conda recovery marker mismatch')
    state = json.loads((RUN / 'status.json').read_text())
    if state.get('stage') != 'stopped' or (RUN / 'completion.json').exists():
        raise RuntimeError('Resume requires a stopped, incomplete installation')
    # 先检查旧安装锁；安装器启动后仍会自行持锁，避免同时操作 prefix。
    with (RUN / 'installer.lock').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=30s', str(seconds) + 's',
            '/opt/conda/bin/python', '-u', '-B', str(RUN / 'install_fno_phase4.py'),
            '--deadline', str(deadline.timestamp())]
    record = {'mode': 'launch' if args.confirm_start else 'dry_run',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'deadline_utc': deadline.isoformat(), 'command': argv,
              'previous_state': state, 'log': str(RUN / 'installer.log')}
    if args.confirm_start:
        # 原始 pid.json 与日志不覆盖；每次恢复另存启动记录并追加总日志。
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        record_path = RUN / ('resume-' + stamp + '.json')
        with record_path.open('x') as saved, (RUN / 'installer.log').open('ab') as log:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1'))
            record.update(supervisor_pid=process.pid, session_id=process.pid)
            json.dump(record, saved, indent=2)
            saved.flush()
            os.fsync(saved.fileno())
        record['record_path'] = str(record_path)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
