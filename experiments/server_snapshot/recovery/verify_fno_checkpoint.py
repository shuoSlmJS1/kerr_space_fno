"""One historical Track A checkpoint GPU inference with paired reference comparison."""
import atexit
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

RUN = Path(os.environ['FNO_RUN_DIR'])
ROOT = Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
WAVE = ROOT / 'outputs/benchmark_track_a_v1/phase_i_wave1_fno1d_20260912'
CKPT = WAVE / 'train_t1200/best_model.pt'
REFERENCE = WAVE / 'eval_t1200/prediction_t1200.npz'
ATOL, RTOL = 1e-5, 1e-4


def mark(stage):
    print(json.dumps({'stage': stage, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}), file=sys.stderr, flush=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


atexit.register(mark, 'python_atexit_before_torch')
sys.path.insert(0, str(ROOT))
mark('before_import')
import numpy as np
import torch
from src.training.benchmark_track_a import checkpoint_contract
from src.training.benchmark_data import recover_xyz
from src.training.fno1d.input_builder_1d import build_fno1d_input_array
from src.training.fno2d.normalization_2d import normalize_input_field
from scripts.evaluate_plan_b_resolution_generalization_2d import compute_raw_metrics
atexit.register(mark, 'python_atexit_after_torch')
mark('after_import')
torch.set_num_threads(2)
torch.manual_seed(20261010)
assert sha(CKPT) == '11ccbebf1d0fd68c57d0895685adeb7d1386da723242ee8271b18286ebbb7ff1'
assert sha(REFERENCE) == '4482c19f58f57b5102e33942f6a50fb04ce7a31729727b041457922edef3c0c3'
mark('before_checkpoint_load')
payload = torch.load(CKPT, map_location='cpu', weights_only=True)
training = json.loads((WAVE / 'train_t1200/summary.json').read_text())
matrix = json.loads((WAVE / 'eval_t1200/matrix.json').read_text())
assert matrix['checkpoint_sha256'] == training['best_checkpoint_sha256'] == sha(CKPT)
assert payload['epoch'] == training['best_epoch'] == 499
assert payload['completed_epochs'] == training['completed_epochs'] == 500
assert payload['model_name'] == 'fno1d' and payload['train_resolution'] == 1200
assert matrix['normalization'] == payload['normalization']
assert matrix['source_provenance']['source_sha256'] == payload['source_provenance']['source_sha256']
for rel, expected in payload['source_provenance']['source_sha256'].items():
    assert sha(ROOT / rel) == expected, 'Historical source mismatch: ' + rel
# 原始契约同时检查架构、容量、训练协议、归一化及 strict state_dict 加载。
model, stats = checkpoint_contract(payload)
loaded = model.load_state_dict(payload['model_state_dict'], strict=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
dimensions = {k: list(v.shape) for k, v in model.state_dict().items()}
assert all(dimensions[k] == list(v.shape) for k, v in payload['model_state_dict'].items())
mark('checkpoint_loaded_strict')
with np.load(REFERENCE, allow_pickle=False) as source:
    q = source['Q'][:1].copy()
    grid = source['lambda_grid'].copy()
    expected = source['prediction_xyz'][:1].copy()
assert q.shape == (1, 1) and grid.shape == (1200,) and expected.shape == (1, 1200, 3)
assert np.allclose(grid, np.arange(1200, dtype=np.float64) * .005, atol=1e-12, rtol=0)
raw_input = build_fno1d_input_array(q, grid)
normalized = normalize_input_field(raw_input[None], stats)[0]
model.eval().requires_grad_(False).cuda()
mark('before_gpu_forward')
torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode():
    predicted_normalized = model(torch.from_numpy(normalized).cuda())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    assert torch.isfinite(predicted_normalized).all().item()
    actual = recover_xyz(predicted_normalized.cpu().numpy(), stats)
mark('after_gpu_forward_synchronize')
assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
delta = actual.astype(np.float64) - expected.astype(np.float64)
passed = bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))
# 使用同一历史指标实现，但参考项是旧预测而非真值，不能冒充科学精度复现。
metrics = compute_raw_metrics(actual, expected, q)
with (RUN / 'paired-prediction.npz').open('xb') as f:
    np.savez_compressed(f, Q=q, lambda_grid=grid, normalized_input=normalized,
                        prediction_xyz=actual, historical_prediction_xyz=expected)
assert sha(CKPT) == training['best_checkpoint_sha256']
result = {'status': 'PASS' if passed else 'FAIL_REFERENCE_TOLERANCE',
          'checkpoint_load': 'PASS_STRICT', 'missing_keys': loaded.missing_keys,
          'unexpected_keys': loaded.unexpected_keys, 'state_dict_shapes': dimensions,
          'state_dict_entry_count': len(dimensions), 'model_config': payload['model_config'],
          'capacity': payload['capacity'], 'normalization': payload['normalization'],
          'target_transform': payload['training']['target_transform'],
          'checkpoint_sha256': sha(CKPT), 'reference_sha256': sha(REFERENCE),
          'training_git_head': payload['source_provenance']['git_head'],
          'historical_source_hashes_verified': len(payload['source_provenance']['source_sha256']),
          'training_dataset_provenance': payload['dataset_provenance'],
          'selected_reference_index': 0, 'Q': q.tolist(), 'lambda_grid_shape': list(grid.shape),
          'lambda_grid_endpoints': [float(grid[0]), float(grid[-1])],
          'input_shape': list(normalized.shape), 'output_shape': list(actual.shape),
          'all_outputs_finite': True, 'max_abs_difference': float(np.max(np.abs(delta))),
          'atol': ATOL, 'rtol': RTOL, 'allclose': passed,
          'historical_prediction_difference_metrics': metrics,
          'comparison_scope': 'Same Q/lambda and checkpoint train-only normalization, one native T1200 trajectory. Reference is saved old prediction, not ground truth.',
          'batch_difference': 'Historical evaluation used batch 32; this bounded FNO diagnostic uses batch 1. FNO has no cross-sample mixing, but numeric kernel differences remain possible.',
          'full_historical_metrics_reproduced': False, 'formal_evaluation': False,
          'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
          'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
          'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
          'gpu_forward_seconds_including_input_copy': seconds, 'gpu_forward_count': 1,
          'checkpoint_unchanged': True, 'training': False, 'peak_gpu_memory': None}
with (RUN / 'checkpoint-result.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result), flush=True)
assert passed, 'Historical reference comparison exceeded the preset tolerance'
assert all(p.grad is None for p in model.parameters())
mark('result_saved')
mark('script_end')
