# Local Development Environment

This file records the local environment used by Codex for lightweight development
and validation.

It is not intended to reproduce every Linux-specific package or build identifier
from the research server.

## Local Environment

- Conda environment: `fno_codex_local`
- Operating system: Windows
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- GPU memory: approximately 8 GB
- Python: 3.10.19
- PyTorch: 2.5.1
- PyTorch CUDA runtime: 12.1
- NumPy: 2.2.5
- SciPy: 1.15.3
- Matplotlib: 3.10.8
- PyYAML: 6.0.3
- Plotly: 6.9.0
- Narwhals: 2.24.0
- torchvision: 0.20.1
- torchaudio: 2.5.1

## Research Server Reference

- Conda environment: `fno_srv`
- Operating system: Linux
- GPUs: 4 NVIDIA GeForce RTX 4090 GPUs
- Python: 3.10.20
- PyTorch: 2.5.1
- PyTorch CUDA runtime: 12.1
- NumPy: 2.0.1
- SciPy: 1.15.3
- Matplotlib: 3.10.8
- PyYAML: 6.0.3
- Plotly: 6.9.0
- Narwhals: 2.24.0
- torchvision: 0.20.1
- torchaudio: 2.5.1

## Accepted Cross-Platform Differences

The following differences are currently accepted:

- Local Python 3.10.19 versus server Python 3.10.20
- Local NumPy 2.2.5 versus server NumPy 2.0.1

These differences exist because the Windows Conda solver could not safely align
the exact server versions without changing or conflicting with other packages.

Codex must not attempt to force these versions to match without explicit user
approval.

## Environment Boundaries

Codex must use `fno_codex_local` for local Python execution.

Codex must not modify:

- the Conda `base` environment
- the `fno_wave` environment
- other local Conda environments
- the Linux server environment
- global Python installations

Any package installation, removal, upgrade, or downgrade requires explicit user
approval.

## Validation Scope

The local environment is suitable for:

- syntax and import checks
- small synthetic-data tests
- tensor-shape checks
- lightweight data-loader checks
- short CPU tests
- short single-GPU smoke tests
- very small training runs

Formal training, complete inference, large dataset generation, multi-GPU
execution, and final experiment validation must be performed on the Linux
research server.
