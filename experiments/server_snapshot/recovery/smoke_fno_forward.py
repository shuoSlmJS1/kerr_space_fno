"""One bounded CPU/GPU FNO1D comparison; no checkpoint or training."""
import atexit
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

RUN = Path(os.environ['FNO_RUN_DIR'])
RECORD = Path('/workspace/shanjinshuo/environment-locks/fno-phase4-20261010T0755Z')


def mark(stage):
    print(json.dumps({'stage': stage, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}), file=sys.stderr, flush=True)


# 校验先前快照，不修改原源码或环境。
verified = []
for row in json.loads((RECORD / 'source-manifest.json').read_text()):
    if row['path'].startswith('model-source/'):
        p = RECORD / row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
        verified.append(row)
with (RUN / 'source-hashes.json').open('x') as f:
    json.dump(verified, f, indent=2)
sys.path.insert(0, str(RECORD / 'model-source'))
atexit.register(mark, 'python_atexit_before_torch')
mark('before_import')
import torch
from src.models.fno1d.fno1d import fno1d
atexit.register(mark, 'python_atexit_after_torch')
mark('after_import')
torch.set_num_threads(2)
torch.manual_seed(20261010)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
mark('before_model_init')
model = fno1d(in_dim=2, out_dim=3, width=8, modes=4, depth=1).eval()
x = torch.randn(1, 32, 2)
gpu_model = copy.deepcopy(model).cuda().eval()
assert all(torch.equal(v, gpu_model.state_dict()[k].cpu()) for k, v in model.state_dict().items())
mark('after_model_init')
with torch.inference_mode():
    cpu = model(x)
    mark('after_cpu_reference')
    mark('before_gpu_forward')
    gpu = gpu_model(x.cuda())
    mark('after_gpu_forward')
    torch.cuda.synchronize()
    mark('after_synchronize')
    actual = gpu.cpu()
assert list(cpu.shape) == list(actual.shape) == [1, 32, 3]
assert torch.isfinite(cpu).all().item() and torch.isfinite(actual).all().item()
error = actual - cpu
result = {'numeric_status': 'PASS' if torch.allclose(actual, cpu, atol=1e-5, rtol=1e-4) else 'FAIL',
          'cpu_gpu_max_abs_error': error.abs().max().item(),
          'relative_l2_error': (error.norm() / cpu.norm().clamp_min(1e-12)).item(),
          'atol': 1e-5, 'rtol': 1e-4, 'finite': True, 'output_shape': list(actual.shape),
          'seed': 20261010, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0), 'state_dict_equal_before_forward': True,
          'cpu_forwards': 1, 'gpu_forwards': 1, 'training': False, 'checkpoint_load': False}
with (RUN / 'result.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result), flush=True)
assert result['numeric_status'] == 'PASS', 'Preset numerical tolerance failed'
assert all(p.grad is None for p in model.parameters())
assert all(p.grad is None for p in gpu_model.parameters())
mark('result_saved')
mark('script_end')
