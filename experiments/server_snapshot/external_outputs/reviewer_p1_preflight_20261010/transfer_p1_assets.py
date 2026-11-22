"""Dry-run or transfer the two approved P1 subset files through existing SSH transport."""
import argparse
import json
import subprocess
import transfer_yolo_d1 as t

p = argparse.ArgumentParser()
p.add_argument('--execute', action='store_true')
a = p.parse_args()
local = t.PROJECT / 'reports/reviewer_p1_preflight_20261010'
dest = '/workspace/shanjinshuo/datasets/fno_kerr/reviewer_p1_preflight_20261010'
log = '/workspace/shanjinshuo/outputs/fno_kerr/reviewer_p1_preflight_20261010'
t.SERVER = t.SERVER.replace("ROOT = Path('/workspace/shanjinshuo/projects/underwater')", "ROOT = Path('" + dest + "')").replace("STAGE = Path('/workspace/shanjinshuo/logs/yolo-d1-20261010T1325Z/transfer-staging')", "STAGE = Path('" + log + "/transfer-staging')")
rows = [{'relative_path': n, 'destination': dest + '/' + n, 'bytes': (local/n).stat().st_size,
         'sha256': t.digest(local/n)} for n in ['asset-manifest.json', 'train16_val4_t1200.npz']]
states = t.inspect(rows)
results = []
if a.execute:
    for row, state in zip(rows, states):
        if state['state'] == 'already_verified':
            results.append(state)
            continue
        assert t.digest(local / row['relative_path'], state['offset']) == state['partial_sha256']
        request = {'mode': 'receive', 'file': row, 'offset': state['offset'], 'partial_sha256': state['partial_sha256']}
        with (local / row['relative_path']).open('rb') as source:
            source.seek(state['offset'])
            done = subprocess.run(t.argv(request), stdin=source, capture_output=True, timeout=60)
        if done.returncode:
            raise RuntimeError(done.stderr.decode(errors='replace'))
        results.append(json.loads(done.stdout))
    assert all(s['state'] == 'already_verified' for s in t.inspect(rows))
report = {'mode': 'execute' if a.execute else 'dry_run', 'files': rows, 'states': states, 'results': results}
with (local / ('transfer-result.json' if a.execute else 'transfer-dry-run.json')).open('x') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report))
