"""PLaMo 2 inference fixes and single-token Metal optimizations.

Keep checkpoint names and the upstream prefill/SSM implementation intact.
The original checkpoint's RoPE configuration must also be applied to quantized
checkpoints: mlx-lm 0.31.2's PLaMo 2 attention otherwise uses base 10,000.
"""

import types

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.plamo2 import Mamba
from mlx_lm.models.ssm import ssm_update


_decode_conv = None


def _conv_kernel():
    global _decode_conv
    if _decode_conv is None:
        _decode_conv = mx.fast.metal_kernel(
            name="plamo2_decode_conv",
            input_names=["x", "state", "weight"],
            output_names=["out", "next_state"],
            source="""
                uint c = thread_position_in_grid.x;
                if (c >= channels) return;
                float acc = 0.0f;
                for (uint k = 0; k < width - 1; ++k) {
                    acc += float(state[k * channels + c]) * float(weight[c * width + k]);
                }
                acc += float(x[c]) * float(weight[c * width + width - 1]);
                // Preserve the convolution output's rounding before SiLU.
                float v = float(T(acc));
                // MLX SiLU rounds sigmoid to the activation dtype before multiply.
                T sigmoid = T(1.0f / (1.0f + metal::exp(-v)));
                out[c] = T(v * float(sigmoid));
                for (uint k = 0; k < width - 2; ++k) {
                    next_state[k * channels + c] = state[(k + 1) * channels + c];
                }
                next_state[(width - 2) * channels + c] = x[c];
            """,
        )
    return _decode_conv


def _fast_conv(self, conv_input, cache, mask):
    # The general upstream path handles prefill, batches, padding and CPU.
    if (
        conv_input.shape[:2] != (1, 1)
        or cache is None
        or cache[0] is None
        or cache.lengths is not None
        or mask is not None
        or mx.default_device() != mx.gpu
        or self.conv_kernel_size < 2
    ):
        return Mamba._conv(self, conv_input, cache, mask)
    channels = conv_input.shape[-1]
    out, state = _conv_kernel()(
        inputs=[conv_input, cache[0], self.conv1d.weight],
        template=[("T", conv_input.dtype), ("channels", channels), ("width", self.conv_kernel_size)],
        grid=(channels, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[conv_input.shape, cache[0].shape],
        output_dtypes=[conv_input.dtype, cache[0].dtype],
    )
    cache[0] = state
    return out


def _plamo_ssm(self, x, B, C, dt, cache, mask):
    batch, length, _ = x.shape
    # PLaMo's reference uses unbounded softplus(dt + bias), unlike the generic
    # MLX SSM default which clamps dt to [0.001, 100]. A is computed in FP32.
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


def configure_model(model, config: dict, *, optimize: bool = False):
    """Configure an already-loaded model without modifying global MLX classes."""
    if config.get("model_type") != "plamo2":
        raise ValueError("The translation backend requires a PLaMo 2 checkpoint")
    full_attention = config.get("full_attention_idx") or []
    for index, layer in enumerate(model.layers):
        if layer.is_mamba:
            object.__setattr__(layer.mixer, "_ssm", types.MethodType(_plamo_ssm, layer.mixer))
            # Bind only this instance; checkpoint parameter names stay unchanged.
            conv = _fast_conv if optimize else Mamba._conv
            object.__setattr__(layer.mixer, "_conv", types.MethodType(conv, layer.mixer))
        else:
            base = (
                config.get("rope_theta", 10000)
                if index in full_attention
                else config.get("rope_local_theta", config.get("rope_theta", 10000))
            )
            layer.mixer.rope = nn.RoPE(layer.mixer.qk_dim, traditional=False, base=base)
    return model
