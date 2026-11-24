"""CPU-only package and model import validation. No checkpoint loads or training."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN / 'model-source'))
import torch
import numpy as np
import scipy
import matplotlib
import yaml
import plotly
import triton

expected = {'torch': '2.5.1+cu121', 'numpy': '2.0.1', 'scipy': '1.15.3',
            'matplotlib': '3.10.8', 'PyYAML': '6.0.3', 'triton': '3.1.0',
            'sympy': '1.13.1', 'plotly': '6.9.0', 'narwhals': '2.24.0'}
assert sys.version_info[:3] == (3, 10, 20)
versions = {name: importlib.metadata.version(name) for name in expected}
assert versions == expected, versions
assert torch.version.cuda == '12.1', torch.version.cuda
torch.set_num_threads(2)
with torch.inference_mode():
    value = float(torch.tensor([1., 2., 3., 4.], device='cpu').square().sum())
    assert value == 30.0
    array = np.arange(8, dtype=np.float32)
    assert np.array_equal(torch.from_numpy(array).numpy(), array)
    spectrum = torch.fft.rfft(torch.arange(16, dtype=torch.float32))
    assert torch.isfinite(spectrum).all().item()
modules = [
    'src.models.fno1d.fno1d', 'src.models.fno2d.fno2d',
    'src.models.fno2d.fno2d_physical_frequency', 'src.models.benchmark_track_a',
    'src.models.resnet1d', 'src.models.timesnet1d', 'src.models.timesnet_lambda_isolated1d'
]
for name in modules:
    importlib.import_module(name)
result = {'status': 'passed_cpu_only', 'python': sys.version.split()[0], 'versions': versions,
          'torch_cuda_build': torch.version.cuda, 'cuda_available_observed': torch.cuda.is_available(),
          'cuda_tensor_test': 'not_run_cpu_container', 'cpu_tensor_result': value,
          'numpy_torch_roundtrip': 'passed', 'cpu_fft': 'passed', 'model_imports': modules,
          'model_forward': 'not_run', 'checkpoint_load': 'not_run', 'training': False,
          'gpu_validation': 'pending', 'cross_container_reuse': 'not_tested'}
print(json.dumps(result, indent=2))
# 结果仅写入本轮安装记录，不写入模型源码或科研输出目录。
with (RUN / 'cpu-smoke.json').open('w', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2)
