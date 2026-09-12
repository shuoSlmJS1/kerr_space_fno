"""Track A direct trajectory models and nominal capacity accounting."""

from copy import deepcopy

import torch
from torch import nn

from src.models.fno1d.fno1d import fno1d
from src.models.resnet1d import DilatedResNet1D
from src.models.timesnet1d import TimesNetReconstruction1D


MODEL_CONFIGS = {
    "bilstm": {"hidden_size": 184, "num_layers": 2, "dropout": 0.0},
    "resnet": {"width": 84, "kernel_size": 7, "dilations": [2**i for i in range(11)]},
    "timesnet": {"d_model": 80, "d_ff": 96, "num_blocks": 2,
                 "top_k": 2, "kernel_sizes": [1, 3, 5], "dropout": 0.0},
    "transformer": {"d_model": 192, "nhead": 6, "num_layers": 2,
                    "dim_feedforward": 1024, "dropout": 0.1},
    "fno1d": {"modes": 32, "width": 64, "depth": 4},
    "deeponet": {"width": 384, "hidden_layers": 4, "latent_dim": 128},
}


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """统计名义实标量坐标；不扣除函数冗余，也不重复统计共享参数。"""
    parameters = [p for p in model.parameters() if p.requires_grad]
    return {
        "real_scalar_parameter_count": sum(p.numel() * (2 if p.is_complex() else 1) for p in parameters),
        "tensor_numel": sum(p.numel() for p in parameters),
        "trainable_parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
    }


class BiLSTM(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.recurrent = nn.LSTM(2, hidden_size, num_layers=num_layers,
                                 batch_first=True, bidirectional=True, dropout=dropout)
        self.output = nn.Linear(2 * hidden_size, 3)

    def forward(self, x):
        return self.output(self.recurrent(x)[0])


class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(2, d_model)
        # 每层独立初始化；实际 lambda 已在输入中，无固定长度位置参数。
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                       dropout=dropout, activation="relu",
                                       batch_first=True, norm_first=False)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.input(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def _mlp(width, hidden_layers, out_dim):
    layers = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers.extend([nn.Linear(width, width), nn.Tanh()])
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, width: int, hidden_layers: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.branch = _mlp(width, hidden_layers, 3 * latent_dim)
        self.trunk = _mlp(width, hidden_layers, latent_dim)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 2 or x.shape[1] == 0:
            raise ValueError("DeepONet requires [B,T,2] Q/lambda input.")
        q = x[:, :1, :1]
        if not torch.equal(x[:, :, :1], q.expand(-1, x.shape[1], -1)):
            raise ValueError("DeepONet requires constant broadcast Q within each trajectory.")
        branch = self.branch(q[:, 0]).reshape(-1, 3, self.latent_dim)
        trunk = self.trunk(x[:, :, 1:2])
        return torch.einsum("bcp,btp->btc", branch, trunk) + self.bias


def model_config(name: str) -> dict:
    return deepcopy(MODEL_CONFIGS[name])


def build_model(name: str, config: dict | None = None) -> nn.Module:
    expected = model_config(name)
    if config is not None and config != expected:
        raise ValueError("Checkpoint architecture does not match the Track A implementation.")
    if name == "bilstm":
        model = BiLSTM(**expected)
    elif name == "transformer":
        model = TrajectoryTransformer(**expected)
    elif name == "deeponet":
        model = DeepONet(**expected)
    else:
        factory = {"resnet": DilatedResNet1D, "timesnet": TimesNetReconstruction1D,
                   "fno1d": fno1d}[name]
        model = factory(in_dim=2, out_dim=3, **expected)
    if not 900_000 <= parameter_counts(model)["real_scalar_parameter_count"] <= 1_300_000:
        raise ValueError("Protocol-impacting issue: model capacity is outside the approved band.")
    return model
