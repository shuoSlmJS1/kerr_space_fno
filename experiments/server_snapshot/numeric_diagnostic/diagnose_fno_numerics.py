"""Bounded single-condition historical FNO prediction diagnosis."""
import argparse
import atexit
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
parser.add_argument('--batch', type=int, choices=[1, 32], required=True)
parser.add_argument('--disable-cudnn-tf32', action='store_true')
args = parser.parse_args()
RUN = Path(args.run_dir)
ROOT = Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
WAVE = ROOT / 'outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912'
CKPT = WAVE / 'train_t1200/best_model.pt'
REF = WAVE / 'eval_t1200/prediction_t1200.npz'
OLD = Path('/workspace/shanjinshuo/logs/fno-recovery-20261011T0134Z')


def mark(stage):
    print(json.dumps({'stage': stage, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}), file=sys.stderr, flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


atexit.register(mark, 'python_atexit_before_torch')
mark('before_import')
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.training.benchmark_track_a import checkpoint_contract, predict_frozen
from src.training.benchmark_data import Trajectories
from src.training.fno1d.input_builder_1d import build_fno1d_input_array
from src.training.fno2d.normalization_2d import normalize_input_field
atexit.register(mark, 'python_atexit_after_torch')
mark('after_import')
torch.set_num_threads(2)
torch.manual_seed(20261010)
if args.disable_cudnn_tf32:
    # 仅更改当前独立诊断进程，不写入环境或系统设置。
    torch.backends.cudnn.allow_tf32 = False
assert sha(CKPT) == '11ccbebf1d0fd68c57d0895685adeb7d1386da723242ee8271b18286ebbb7ff1'
assert sha(REF) == '4482c19f58f57b5102e33942f6a50fb04ce7a31729727b041457922edef3c0c3'
payload = torch.load(CKPT, map_location='cpu', weights_only=True)
matrix = json.loads((WAVE / 'eval_t1200/matrix.json').read_text())
assert matrix['checkpoint_sha256'] == sha(CKPT)
assert matrix['normalization'] == payload['normalization']
assert payload['training']['target_transform'] == 'raw'
for rel, expected in payload['source_provenance']['source_sha256'].items():
    assert sha(ROOT / rel) == expected, 'Source hash mismatch: ' + rel
model, stats = checkpoint_contract(payload)
with np.load(REF, allow_pickle=False) as f:
    q_all, grid, ref_all = f['Q'].copy(), f['lambda_grid'].copy(), f['prediction_xyz'].copy()
assert q_all.shape == (400, 1) and grid.shape == (1200,)
assert ref_all.shape == (400, 1200, 3) and ref_all.dtype == np.float32
assert np.all(np.diff(q_all[:, 0]) > 0) and q_all[0, 0] == 1.6007
q, reference = q_all[:args.batch], ref_all[:args.batch]
# 真值未迁移；占位数组只满足原始 Dataset 的接口，从不用于模型输入或指标。
dataset = Trajectories(q, grid, np.zeros((args.batch, 1200, 3), dtype=np.float64), stats)
x_exact = np.stack([dataset[i][0].numpy() for i in range(args.batch)])
x_vectorized = normalize_input_field(build_fno1d_input_array(q, grid)[None], stats)[0]
assert np.array_equal(x_exact, x_vectorized)
with np.load(OLD / 'paired-prediction.npz', allow_pickle=False) as previous:
    assert np.array_equal(x_exact[:1], previous['normalized_input'])
    assert np.array_equal(reference[:1], previous['historical_prediction_xyz'])
    assert np.array_equal(q[:1], previous['Q']) and np.array_equal(grid, previous['lambda_grid'])
mark('input_and_provenance_verified')
model.eval().requires_grad_(False).to(args.device)
flags = {'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
         'float32_matmul_precision': torch.get_float32_matmul_precision(),
         'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
         'cudnn_enabled': torch.backends.cudnn.enabled,
         'cudnn_benchmark': torch.backends.cudnn.benchmark,
         'cudnn_deterministic': torch.backends.cudnn.deterministic,
         'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
         'default_dtype': str(torch.get_default_dtype()),
         'autocast_enabled': torch.is_autocast_enabled(),
         'environment_overrides': {k: os.environ.get(k) for k in ['NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG']}}
mark('before_forward')
start = time.perf_counter()
with torch.inference_mode():
    actual, compute = predict_frozen(model, dataset, stats, args.device)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
elapsed = time.perf_counter() - start
mark('after_forward_synchronize')
assert actual.shape == reference.shape and np.isfinite(actual).all()
assert all(p.grad is None for p in model.parameters())


def metrics(a, b):
    # 指标统一用 float64 归约，逐点容差以第二项为参考。
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = a - b
    close = np.abs(d) <= 1e-5 + 1e-4 * np.abs(b)
    return {'max_abs_diff': float(np.max(np.abs(d))),
            'relative_l2': float(np.linalg.norm(d) / (np.linalg.norm(b) + 1e-12)),
            'MSE': float(np.mean(d ** 2)), 'allclose': bool(close.all()),
            'outside_tolerance': int((~close).sum()), 'scalars': int(close.size),
            'atol': 1e-5, 'rtol': 1e-4}


with (RUN / (args.name + '.npz')).open('xb') as f:
    np.savez_compressed(f, prediction_xyz=actual, historical_prediction_xyz=reference,
                        normalized_input=x_exact, Q=q, lambda_grid=grid)
assert sha(CKPT) == matrix['checkpoint_sha256']
result = {'status': 'COMPUTATION_PASS', 'name': args.name, 'device': args.device,
          'batch': args.batch, 'forward_calls': 1, 'eval': True, 'inference_mode': True,
          'canonical_indices': list(range(args.batch)), 'Q': q[:, 0].tolist(),
          'input_shape': list(x_exact.shape), 'output_shape': list(actual.shape),
          'finite': True, 'first_input_matches_previous_bitwise': True,
          'dataset_and_vectorized_inputs_bitwise_equal': True,
          'normalized_input_sha256': hashlib.sha256(x_exact.tobytes()).hexdigest(),
          'canonical_zero_input_sha256': hashlib.sha256(x_exact[:1].tobytes()).hexdigest(),
          'reference_dtype': str(ref_all.dtype), 'input_dtype': str(x_exact.dtype),
          'output_dtype': str(actual.dtype), 'Q_dtype': str(q_all.dtype),
          'lambda_dtype': str(grid.dtype), 'normalization': stats.to_dict(),
          'target_transform': 'raw', 'ground_truth_available': False,
          'ground_truth_used_for_prediction_or_comparison': False,
          'historical_first_batch_exact_Q_and_grid': args.batch == 32,
          'path': 'Original Trajectories and predict_frozen with DataLoader batch_size=32, shuffle=False, num_workers=0; one dataset subset only; outer inference_mode.',
          'comparison_index0_to_history': metrics(actual[:1], reference[:1]),
          'comparison_all_selected_to_history': metrics(actual, reference),
          'precision': flags, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
          'cudnn_version': torch.backends.cudnn.version(),
          'torch_build': torch.__config__.show(),
          'gpu': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
          'wall_seconds': elapsed, 'checkpoint_sha256': sha(CKPT), 'reference_sha256': sha(REF),
          'source_hashes_verified': len(payload['source_provenance']['source_sha256']),
          'training': False, 'memory_measurement': 'Not interpreted under HAMI'}
with (RUN / (args.name + '.json')).open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result), flush=True)
mark('result_saved')
mark('script_end')
