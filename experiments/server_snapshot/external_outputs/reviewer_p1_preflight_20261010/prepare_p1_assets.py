"""Extract an auditable train/validation-only smoke subset using WSL standard Python."""
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

BASE = Path('/mnt/d/AAA_MyNote/Server_Migration/amax_2026-10-01/fno_kerr/kerr_project')
OUT = Path('/mnt/d/Codex_Research_Workspace/projects/school_gpu_control/reports/reviewer_p1_preflight_20261010')
OUT.mkdir(exist_ok=False)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_array(z, name):
    raw = z.read(name + '.npy')
    f = io.BytesIO(raw)
    assert f.read(6) == b'\x93NUMPY'
    version = tuple(f.read(2))
    size = struct.unpack('<H' if version == (1, 0) else '<I', f.read(2 if version == (1, 0) else 4))[0]
    header = ast.literal_eval(f.read(size).decode('latin1'))
    assert header['descr'] == '<f8' and not header['fortran_order']
    body = f.read()
    assert len(body) == math.prod(header['shape']) * 8
    return header['shape'], body


def encode(shape, body):
    header = repr({'descr': '<f8', 'fortran_order': False, 'shape': tuple(shape)}).encode('ascii')
    header += b' ' * ((64 - ((10 + len(header) + 1) % 64)) % 64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + body


tasks = ['q_1p6-3_n2000_t1200', 'q_1p6-3_n2000_t2399_plan_b_matched_v1',
         'q_1p6007-2p9993_n400_t1200', 'q_1p6007-2p9993_n400_t1800', 'q_1p6007-2p9993_n400_t2400']
audit, q_cache = {}, {}
for task in tasks:
    d = BASE / 'data/tasks' / task
    p = d / 'dataset.npz'
    meta = json.loads((d / 'meta.json').read_text())
    rows = {}
    with zipfile.ZipFile(p) as z:
        for split in ['train', 'val', 'test']:
            shape, body = read_array(z, 'x_' + split)
            q = list(struct.unpack('<' + 'd' * shape[0], body))
            assert shape == (len(q), 1)
            q_cache[(task, split)] = q
            rows[split] = {'shape': shape, 'dtype': 'float64', 'sha256_raw_q': hashlib.sha256(body).hexdigest(),
                           'min_Q': min(q), 'max_Q': max(q), 'unique': len(set(q)) == len(q)}
        grid_shape, grid = read_array(z, 'lambda_grid')
    audit[task] = {'source': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p),
                   'metadata_source': str(d / 'meta.json'), 'metadata_sha256': sha(d / 'meta.json'),
                   'split_Q': rows, 'lambda_shape': grid_shape,
                   'lambda_sha256': hashlib.sha256(grid).hexdigest(),
                   'lambda_endpoints': [struct.unpack('<d', grid[:8])[0], struct.unpack('<d', grid[-8:])[0]],
                   'metadata': meta, 'truth_read': task == tasks[0]}
assert audit[tasks[0]]['sha256'] == '3e0358831e6b96c38f15c8e3178f6d37f38b4bddbe1ab01a63cfaba043f211ce'
expected = None
# 用科研源码内的正式契约提取三组哈希，避免维护第二份手写协议。
contract = (BASE / 'src/data_generation/plan_b_matched_training.py').read_text()
tree = ast.parse(contract)
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'HistoricalT1200TrainingContract':
        expected = ast.literal_eval(next(k.value for k in node.keywords if k.arg == 'split_q_sha256'))
for split, value in expected.items():
    assert audit[tasks[0]]['split_Q'][split]['sha256_raw_q'] == value
    assert q_cache[(tasks[0], split)] == q_cache[(tasks[1], split)]
train = q_cache[(tasks[0], 'train')]
val = q_cache[(tasks[0], 'val')]
test = q_cache[(tasks[0], 'test')]
assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
order = sorted(range(len(train)), key=lambda i: train[i])
selected = {'train': [order[round(i * (len(order) - 1) / 15)] for i in range(16)],
            'val': [i for i, q in enumerate(val) if min(train) <= q <= max(train)][:4]}
subset = OUT / 'train16_val4_t1200.npz'
with zipfile.ZipFile(BASE / 'data/tasks' / tasks[0] / 'dataset.npz') as z, zipfile.ZipFile(subset, 'x', compression=zipfile.ZIP_DEFLATED) as dest:
    for split in ['train', 'val']:
        for prefix in ['x_', 'y_']:
            shape, body = read_array(z, prefix + split)
            stride = math.prod(shape[1:]) * 8
            sliced = b''.join(body[i*stride:(i+1)*stride] for i in selected[split])
            dest.writestr(prefix + split + '.npy', encode((len(selected[split]), *shape[1:]), sliced))
    shape, body = read_array(z, 'lambda_grid')
    dest.writestr('lambda_grid.npy', encode(shape, body))
source_files = ['src/training/benchmark_data.py', 'src/training/benchmark_track_a.py',
                'src/data_generation/plan_b_matched_training.py', 'src/data_generation/orbit_solver_second_order.py',
                'src/data_generation/orbit_math.py', 'src/data_generation/orbit_types.py',
                'scripts/evaluate_plan_b_resolution_generalization_2d.py', 'scripts/evaluate_formal_length_extrapolation_2d.py',
                'src/training/fno2d/normalization_2d.py', 'src/models/fno1d/fno1d.py', 'src/models/fno1d/layers1d.py',
                'outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912/train_t1200/best_model.pt']
q400 = sorted(sum([q_cache[(tasks[2], s)] for s in ['train', 'val', 'test']], []))
manifest = {'status': 'VERIFIED_LOCAL_SUBSET', 'source_dataset': audit[tasks[0]], 'all_candidate_assets': audit,
            'source_files': [{'relative_path': rel, 'bytes': (BASE/rel).stat().st_size, 'sha256': sha(BASE/rel)} for rel in source_files],
            'subset': {'file': subset.name, 'bytes': subset.stat().st_size, 'sha256': sha(subset),
                       'selection': '16 uniformly spaced ranks of ascending training Q; first 4 original validation rows inside the training hull; selected before observing errors.',
                       'source_rows': selected, 'Q': {s: [q_cache[(tasks[0], s)][i] for i in selected[s]] for s in selected},
                       'purpose': 'SMOKE_ONLY_NOT_FORMAL_BENCHMARK', 'test_truth_read': False},
            'split_audit': {'train_val_test_disjoint': True, 'T1200_T2399_split_Q_exact_equal': True,
                            'Q400_intersection_with_n2000_count': len(set(q400) & set(train+val+test)),
                            'Q400_min_distance_to_n2000': min(abs(x-y) for x in q400 for y in train+val+test),
                            'Q400_inside_train_hull': min(q400) >= min(train) and max(q400) <= max(train),
                            'val_outside_train_hull': sum(q < min(train) or q > max(train) for q in val),
                            'test_outside_train_hull': sum(q < min(train) or q > max(train) for q in test),
                            'Plan_A_Q400_T1200_T1800_T2400_split_Q_equal': all(q_cache[(tasks[2],s)] == q_cache[(t,s)] for t in tasks[3:] for s in ['train','val','test'])},
            'no_heldout_error_computed': True}
with (OUT / 'asset-manifest.json').open('x') as f:
    json.dump(manifest, f, indent=2)
print(json.dumps({'subset': manifest['subset'], 'split_audit': manifest['split_audit']}))
