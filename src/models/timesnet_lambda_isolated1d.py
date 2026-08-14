from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional

from src.models.timesnet1d import (
    DEFAULT_INCEPTION_KERNELS,
    TimesBlock,
    TimesNetReconstruction1D,
    select_dominant_frequencies,
)


LAMBDA_CHANNEL_INDEX = 4


class TimesNetLambdaIsolatedPeriodSelection1D(TimesNetReconstruction1D):
    """仅在 period selection 中隔离 lambda 直接贡献的 TimesNet 消融模型。"""

    def __init__(
        self,
        in_dim: int = 5,
        out_dim: int = 3,
        d_model: int = 80,
        d_ff: int = 96,
        num_blocks: int = 2,
        top_k: int = 2,
        kernel_sizes: Sequence[int] = DEFAULT_INCEPTION_KERNELS,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_blocks=num_blocks,
            top_k=top_k,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        if self.in_dim != 5:
            raise ValueError("This ablation requires the fixed five-channel input contract.")

    def architecture_metadata(self) -> dict[str, object]:
        """返回明确记录 lambda-isolated period-selection 的实验元数据。"""
        metadata = super().architecture_metadata()
        metadata.update(
            {
                "family": "timesnet_lambda_isolated_period_selection1d",
                "period_selection_treatment": "lambda_isolated_shared_parameter_auxiliary_stream",
                "first_block_lambda_component": "input_lambda_times_input_projection_weight_column",
                "later_block_selection_stream": "shared_parameter_counterfactual_stream",
                "prediction_latent": "full_input_projection_and_full_latent_stream",
                "selection_stream_trainable_parameters": 0,
            }
        )
        return metadata

    def lambda_projection_component(self, x: torch.Tensor) -> torch.Tensor:
        """返回输入投影中不含 bias 的精确 lambda 分量。"""
        self._validate_input(x)
        lambda_values = x[..., LAMBDA_CHANNEL_INDEX : LAMBDA_CHANNEL_INDEX + 1]
        lambda_weight = self.input_projection.weight[:, LAMBDA_CHANNEL_INDEX]
        return lambda_values * lambda_weight.view(1, 1, -1)

    @staticmethod
    def _apply_period_branches(
        block: TimesBlock,
        x: torch.Tensor,
        periods: torch.Tensor,
        period_weight: torch.Tensor,
    ) -> torch.Tensor:
        """将由选择流得到的 periods 应用于给定的完整或辅助 latent 流。"""
        sequence_length = int(x.shape[1])
        branch_outputs: list[torch.Tensor] = []
        for period in periods.tolist():
            padded_length = ((sequence_length + period - 1) // period) * period
            padded = x if padded_length == sequence_length else functional.pad(
                x,
                (0, 0, 0, padded_length - sequence_length),
            )
            height = padded_length // period
            branch = padded.reshape(x.shape[0], height, period, block.d_model)
            branch = block.inception_in(branch.permute(0, 3, 1, 2).contiguous())
            branch = block.activation(branch)
            branch = block.inception_out(branch)
            branch = branch.permute(0, 2, 3, 1).reshape(
                x.shape[0],
                padded_length,
                block.d_model,
            )
            branch_outputs.append(branch[:, :sequence_length, :])
        stacked = torch.stack(branch_outputs, dim=-1)
        return (stacked * period_weight[:, None, None, :]).sum(dim=-1) + x

    def _run_latent_streams(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, list[dict[str, list[int]]]]:
        """并行运行完整预测流与仅用于 period selection 的共享参数辅助流。"""
        self._validate_input(x)
        full_hidden = self.input_projection(x)
        selection_hidden = full_hidden - self.lambda_projection_component(x)
        diagnostics: list[dict[str, list[int]]] = []
        for block, normalization in zip(self.blocks, self.feature_normalizations):
            frequencies, periods, period_weight = select_dominant_frequencies(
                selection_hidden,
                self.top_k,
            )
            if collect_diagnostics:
                diagnostics.append(
                    {
                        "selected_frequency_indices": [int(value) for value in frequencies],
                        "selected_periods": [int(value) for value in periods],
                    }
                )
            # 完整 latent 流始终用于卷积 period branches 与最终预测。
            full_hidden = normalization(
                self._apply_period_branches(block, full_hidden, periods, period_weight)
            )
            # 辅助流共享同一参数，但不进入输出投影或残差预测路径。
            selection_hidden = normalization(
                self._apply_period_branches(
                    block,
                    selection_hidden,
                    periods,
                    period_weight,
                )
            )
        return full_hidden, diagnostics

    @torch.no_grad()
    def inspect_frequency_diagnostics(self, x: torch.Tensor) -> list[dict[str, list[int]]]:
        """返回每个 TimesBlock 的 lambda-isolated selection 频率与周期。"""
        _, diagnostics = self._run_latent_streams(x, collect_diagnostics=True)
        return diagnostics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将完整五通道输入映射为完整轨道，同时只隔离 period-selection 信号。"""
        full_hidden, _ = self._run_latent_streams(x, collect_diagnostics=False)
        return self.output_projection(full_hidden)


def build_timesnet_lambda_isolated1d_model(
    in_dim: int = 5,
    out_dim: int = 3,
    d_model: int = 80,
    d_ff: int = 96,
    num_blocks: int = 2,
    top_k: int = 2,
) -> TimesNetLambdaIsolatedPeriodSelection1D:
    """构造不改变 canonical 宽度与层数的 lambda-isolated 消融模型。"""
    return TimesNetLambdaIsolatedPeriodSelection1D(
        in_dim=in_dim,
        out_dim=out_dim,
        d_model=d_model,
        d_ff=d_ff,
        num_blocks=num_blocks,
        top_k=top_k,
    )
