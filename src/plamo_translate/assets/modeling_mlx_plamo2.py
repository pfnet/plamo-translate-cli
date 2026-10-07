"""Checkpoint-specific PLaMo 2 inference corrections for mlx-lm 0.31.2.

Weights remain compatible with the upstream PLaMo 2 parameter names. This
module is loaded through config.json's model_file by mlx_lm.utils.load_model.
"""

from dataclasses import dataclass
import types

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.plamo2 import Model as BaseModel
from mlx_lm.models.plamo2 import ModelArgs as BaseModelArgs
from mlx_lm.models.ssm import ssm_update


@dataclass
class ModelArgs(BaseModelArgs):
    rope_theta: float = 10000.0
    rope_local_theta: float = 10000.0


def _plamo_ssm(self, x, B, C, dt, cache, mask):
    batch, length, _ = x.shape
    # The PFN implementation does not clip softplus(dt + bias). Its A and
    # recurrent state use FP32 even when the model activations use BF16.
    y, state = ssm_update(
        x.reshape(batch, length, self.num_heads, self.hidden_size_per_head),
        self.A_log.astype(mx.float32),
        B.reshape(batch, length, 1, self.d_state),
        C.reshape(batch, length, 1, self.d_state),
        self.D,
        dt,
        self.dt_bias,
        cache[1] if cache is not None else None,
        time_step_limit=(0.0, float("inf")),
        mask=mask,
        lengths=cache.lengths if cache is not None else None,
    )
    if cache is not None:
        cache[1] = state
    return y.reshape(batch, length, self.intermediate_size)


class Model(BaseModel):
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        full_attention = config.full_attention_idx or []
        for index, layer in enumerate(self.layers):
            if layer.is_mamba:
                object.__setattr__(layer.mixer, "_ssm", types.MethodType(_plamo_ssm, layer.mixer))
            else:
                base = config.rope_theta if index in full_attention else config.rope_local_theta
                layer.mixer.rope = nn.RoPE(layer.mixer.qk_dim, traditional=False, base=base)
