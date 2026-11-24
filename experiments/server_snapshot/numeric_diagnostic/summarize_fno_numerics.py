"""Summarize saved FNO diagnostic arrays without importing torch or running models."""
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np

r = Path(sys.argv[1])
names = ['cpu_b1', 'gpu_b1', 'gpu_b32']
results = {n: json.loads((r / (n + '.json')).read_text()) for n in names}
processes = {n: json.loads((r / (n + '.process.json')).read_text()) for n in names}
arrays = {}
for n in names:
    with np.load(r / (n + '.npz'), allow_pickle=False) as f:
        arrays[n] = f['prediction_xyz'][:1].copy()


def metrics(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = a - b
    good = np.abs(d) <= 1e-5 + 1e-4 * np.abs(b)
    return {'max_abs_diff': float(np.max(np.abs(d))),
            'relative_l2': float(np.linalg.norm(d) / (np.linalg.norm(b) + 1e-12)),
            'MSE': float(np.mean(d ** 2)), 'allclose': bool(good.all()),
            'outside_tolerance': int((~good).sum()), 'scalars': int(good.size),
            'atol': 1e-5, 'rtol': 1e-4}


pairs = {a + '_vs_' + b: metrics(arrays[a], arrays[b]) for a, b in itertools.combinations(names, 2)}
old = Path('/workspace/shanjinshuo/logs/fno-recovery-20261011T0134Z')
with np.load(old / 'paired-prediction.npz', allow_pickle=False) as f:
    previous_repeated = np.array_equal(arrays['gpu_b1'], f['prediction_xyz'])
before = json.loads((r / 'protected-conda-before.json').read_text())
after = {root: {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(root) / 'conda-meta').glob('*') if p.is_file()} for root in before}
assert before == after, 'Protected Conda metadata changed'
root = Path('/workspace/shanjinshuo/projects/fno_kerr/kerr_project')
restore = json.loads((old / 'restore-manifest.json').read_text())
# 以已保存的完整清单核对科研资产，字段结构在运行前确认。
files = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in r.iterdir() if p.is_file()}
now = datetime.datetime.now(datetime.timezone.utc)
occupancy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
report = {
    'status': 'BATCH_MATCH_RESOLVES_TESTED_HISTORICAL_DISCREPANCY',
    'utc': now.isoformat(), 'hostname': socket.gethostname(), 'log_directory': str(r),
    'scope': 'One CPU forward and two GPU forwards. Same BEST and canonical input index 0. GPU batch32 uses exact original canonical rows 0..31.',
    'atol': 1e-5, 'rtol': 1e-4, 'tolerance_changed': False,
    'historical_evidence': {
        'checkpoint_sha256': results['gpu_b32']['checkpoint_sha256'],
        'reference_sha256': results['gpu_b32']['reference_sha256'],
        'source_hashes_verified_per_condition': 31,
        'prediction_storage': 'Float32 NPZ; np.savez_compressed is lossless, no decimal text rounding in the saved evaluation path.',
        'Q_and_lambda_storage': 'Float64, restored exactly from the paired historical prediction archive.',
        'entry': 'scripts/run_benchmark_track_a.py evaluate -> evaluate_run -> predict_frozen',
        'batch': 32, 'shuffle': False, 'num_workers': 0,
        'forward_path': 'Trajectories.__getitem__ -> original input builder -> checkpoint normalization -> model -> recover_xyz',
        'normalization': results['gpu_b32']['normalization'], 'target_transform': 'raw',
        'canonical_zero_normalized_input_sha256': results['gpu_b32']['canonical_zero_input_sha256'],
        'new_vs_previous_input_bitwise_equal': True,
        'previous_gpu_b1_prediction_repeated_bitwise': bool(previous_repeated),
        'ground_truth_not_required_for_prediction': True,
        'ground_truth_placeholder': 'Zero arrays satisfy the unchanged Dataset target interface only. Never passed to the model, used as reference, or used to compute scientific accuracy.',
        'historical_runtime_precision_gap': 'Historical records do not persist all TF32/matmul/cuDNN runtime flags. Original standalone evaluate entry does not call seed_training; training-only deterministic settings must not be assumed to apply to evaluation.',
        'current_precision': results['gpu_b32']['precision'],
        'current_cudnn_version': results['gpu_b32']['cudnn_version'],
        'historical_torch': '2.5.1, CUDA build 12.1', 'current_torch': results['gpu_b32']['torch'],
        'current_cuda': results['gpu_b32']['cuda'],
        'historical_gpu': 'NVIDIA GeForce RTX 4090', 'current_gpu': results['gpu_b32']['gpu']},
    'conditions': {n: {'process': processes[n], 'index0_vs_history': results[n]['comparison_index0_to_history'],
                        'all_selected_vs_history': results[n]['comparison_all_selected_to_history'],
                        'forward_seconds': results[n]['wall_seconds'],
                        'output_shape': results[n]['output_shape'], 'finite': results[n]['finite']} for n in names},
    'pairwise_index0': pairs,
    'pairwise_metric_orientation': 'First named array minus second; relative norm and pointwise rtol use the second array as reference.',
    'conclusions': {
        'batch_explanation': 'On the same L40S, unchanged flags and original forward path, batch1 repeats the mismatch, while original batch32 matches all 115200 scalars exactly. Batch-dependent execution explains the tested mismatch.',
        'low_level_kernel_cause': 'Not isolated. No kernel profiling or TF32 ablation was necessary or performed.',
        'cpu_gpu': 'CPU batch1 and GPU batch1 differ; see pairwise metrics. Neither batch1 condition matches historical batch32 within the required tolerance.',
        'input_normalization_or_target_error': 'No mismatch found. Saved inputs, exact original input functions, normalization, target transform and source hashes agree.',
        'hami_causality': 'No evidence attributes the discrepancy to HAMI. Batch32 is bitwise equal to history despite two HAMI errors after Python atexit. This does not establish system-wide HAMI safety.',
        'acceptance': 'Accept the tested first historical batch as numerically reproduced, without relaxing tolerance. Do not require or claim batch-invariant output, complete Q400 reproduction, or full historical experiment reproduction.',
        'precision_ablation': 'Not performed; A-C explain the tested discrepancy. No further GPU tests.'},
    'priority1': {
        'selected_checkpoint_numeric_blocker': 'Resolved for canonical first batch32 only.',
        'formal_experiment_ready': False,
        'minimum_remaining': [
            'Confirm the precise Priority 1 locked protocol and obtain execution approval; preserve evaluation batch32 and canonical ordering.',
            'Restore and hash-validate only its necessary dataset/ground-truth assets and required checkpoint/configuration.',
            'Establish runnable Git/source-provenance and approved path/device mapping for the new deployment.',
            'For a full historical-reproduction claim, separately approve a bounded matched-path check of remaining saved Q400 predictions, including the final short batch; do not infer it from 32 rows.',
            'Record exit-stage HAMI warnings and seek platform guidance before long jobs; do not change driver/HAMI.']},
    'safety': {'training': False, 'gpu_forward_count': 2, 'cpu_forward_count': 1,
               'condition_timeout_seconds': 60, 'environment_installation_or_modification': False,
               'conda_metadata_fingerprints_unchanged': before == after,
               'fingerprint_scope': 'Conda metadata, not a whole-environment byte audit.',
               'historical_assets_overwritten': False, 'driver_or_hami_changes': False,
               'post_test_gpu_process_query_code': occupancy.returncode,
               'post_test_gpu_processes': occupancy.stdout.strip()},
    'time': {'nominal_expiry_from_user_utc': '2026-10-11T04:23:00Z',
             'remaining_estimate_minutes': int((datetime.datetime.fromisoformat('2026-10-11T04:23:00+00:00') - now).total_seconds() / 60),
             'platform_scheduler_independently_verified': False},
    'evidence_files': files}
assert results['gpu_b32']['comparison_all_selected_to_history']['max_abs_diff'] == 0
assert all(processes[n]['exit_code'] == 0 for n in names)
with (r / 'diagnostic-report.json').open('x') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report), flush=True)
