"""PLaMo 3 translation model for MLX.

Checkpoint parameter names follow the reference implementation, so original
weights load directly. KV caches hold rotated keys: RoPE only depends on the
query/key distance, so each key needs rotating once, when it enters the cache.
"""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

POST_MIXER_OFFSET = 1.0 / 5
POST_MLP_OFFSET = 1.0 / (5**1.5)


@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    window_size: int = 2048
    sliding_window_pattern: int = 8
    rope_theta: float = 150000
    rope_local_theta: float = 10000
    rope_scaling_factor: float = 1.0
    initial_context_length: int | None = None
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> "ModelArgs":
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in config.items() if key in names})

    def is_full_attention(self, layer: int) -> bool:
        return (layer + 1) % self.sliding_window_pattern == 0


def rope_frequencies(args: ModelArgs, full_attention: bool) -> tuple[np.ndarray, float]:
    """Inverse frequencies and output scale; YaRN applies to full attention only."""
    dim = args.head_dim
    base = args.rope_theta if full_attention else args.rope_local_theta
    freq = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    if not full_attention or args.rope_scaling_factor == 1:
        return 1.0 / freq, 1.0
    half = dim // 2
    factor = args.rope_scaling_factor
    original = args.initial_context_length
    low = half * math.log(original / (32.0 * 2 * math.pi)) / math.log(base)
    high = half * math.log(original / (1.0 * 2 * math.pi)) / math.log(base)
    mask = 1 - np.clip((np.arange(half, dtype=np.float64) - low) / (high - low), 0, 1)
    inverse = (1.0 - mask) / (factor * freq) + mask / freq
    return inverse, 0.1 * math.log(factor) + 1.0


def rope_table(inverse: np.ndarray, scale: float, start: int, length: int, dtype: mx.Dtype) -> mx.array:
    """Rotation terms shaped [length, dim]: cos for the first half, sin for the second."""
    # FP64 keeps high positions exact; the reference instead crops to local coordinates.
    angles = np.arange(start, start + length, dtype=np.float64)[:, None] * inverse[None]
    table = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1) * scale
    return mx.array(table.astype(np.float32)).astype(dtype)


def rotate(x: mx.array, table: mx.array) -> mx.array:
    # x: [batch, heads, length, dim], table: [length, dim]
    half = table.shape[-1] // 2
    cos, sin = table[:, :half], table[:, half:]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def rms_norm(x: mx.array, weight: mx.array, eps: float, offset: float) -> mx.array:
    # The reference rounds the RMS-normalized activations back to the activation dtype.
    wide = x.astype(mx.float32)
    wide = wide * mx.rsqrt(mx.mean(wide * wide, axis=-1, keepdims=True) + eps)
    return (offset + weight) * wide.astype(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float, offset: float = 1.0):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps
        self.offset = offset

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps, self.offset)


class KVCache:
    """Rotated keys and values shaped [heads, capacity, dim]; sliding layers wrap."""

    def __init__(self, heads: int, dim: int, window: int | None, dtype: mx.Dtype, capacity: int = 4096):
        self.window = window
        self.offset = 0
        self.capacity = window or capacity
        self.keys = mx.zeros((heads, self.capacity, dim), dtype=dtype)
        self.values = mx.zeros((heads, self.capacity, dim), dtype=dtype)

    def reset(self) -> None:
        self.offset = 0

    def reserve(self, length: int) -> None:
        if self.window or length <= self.capacity:
            return
        capacity = max(length, self.capacity * 2)
        heads, _, dim = self.keys.shape
        keys = mx.zeros((heads, capacity, dim), dtype=self.keys.dtype)
        values = mx.zeros((heads, capacity, dim), dtype=self.values.dtype)
        keys[:, : self.capacity] = self.keys
        values[:, : self.capacity] = self.values
        self.keys, self.values, self.capacity = keys, values, capacity

    def _history(self) -> tuple[mx.array, mx.array]:
        """Up to window - 1 stored entries that the next query can still attend to, oldest first."""
        stored = min(self.offset, self.capacity)
        keep = min(stored, self.window - 1)
        end = self.offset % self.capacity
        if self.offset <= self.capacity:
            indices = slice(stored - keep, stored)
            return self.keys[:, indices], self.values[:, indices]
        order = (mx.arange(end - keep, end) + self.capacity) % self.capacity
        return self.keys[:, order], self.values[:, order]

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Store [heads, length, dim] entries; return every entry visible to them, oldest first."""
        count = keys.shape[1]
        if self.window is None:
            self.reserve(self.offset + count)
            self.keys[:, self.offset : self.offset + count] = keys
            self.values[:, self.offset : self.offset + count] = values
            self.offset += count
            return self.keys[:, : self.offset], self.values[:, : self.offset]
        old_keys, old_values = self._history()
        merged_keys = mx.concatenate([old_keys, keys], axis=1)
        merged_values = mx.concatenate([old_values, values], axis=1)
        kept = min(count, self.capacity)
        position = self.offset + count - kept
        start = position % self.capacity
        first = min(kept, self.capacity - start)
        self.keys[:, start : start + first] = keys[:, count - kept : count - kept + first]
        self.values[:, start : start + first] = values[:, count - kept : count - kept + first]
        if first < kept:
            self.keys[:, : kept - first] = keys[:, count - kept + first :]
            self.values[:, : kept - first] = values[:, count - kept + first :]
        self.offset += count
        return merged_keys, merged_values


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.q_heads = args.num_attention_heads
        self.kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.full_attention = args.is_full_attention(layer)
        self.window = None if self.full_attention else args.window_size
        q_dim = self.q_heads * self.head_dim
        kv_dim = self.kv_heads * self.head_dim
        self.split = (q_dim, q_dim + kv_dim)
        self.qkv_proj = nn.Linear(args.hidden_size, q_dim + 2 * kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, args.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self._inverse, self._scale = rope_frequencies(args, self.full_attention)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        batch, length, _ = x.shape
        q, k, v = mx.split(self.qkv_proj(x), self.split, axis=-1)
        q = self.q_norm(q.reshape(batch, length, self.q_heads, self.head_dim).transpose(0, 2, 1, 3))
        k = self.k_norm(k.reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3))
        v = v.reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        table = rope_table(self._inverse, self._scale, cache.offset, length, x.dtype)
        keys, values = cache.update(rotate(k, table)[0], v[0])
        attend = None
        if length > 1:
            visible = mx.arange(keys.shape[1])[None]
            queries = mx.arange(keys.shape[1] - length, keys.shape[1])[:, None]
            attend = visible <= queries
            if self.window:
                attend = attend & (visible > queries - self.window)
        out = mx.fast.scaled_dot_product_attention(
            rotate(q, table), keys[None], values[None], scale=self.head_dim**-0.5, mask=attend
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_up_proj = nn.Linear(args.hidden_size, args.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer: int):
        super().__init__()
        self.mixer = Attention(args, layer)
        self.mlp = MLP(args)
        eps = args.rms_norm_eps
        self.pre_mixer_norm = RMSNorm(args.hidden_size, eps)
        self.post_mixer_norm = RMSNorm(args.hidden_size, eps, POST_MIXER_OFFSET)
        self.pre_mlp_norm = RMSNorm(args.hidden_size, eps)
        self.post_mlp_norm = RMSNorm(args.hidden_size, eps, POST_MLP_OFFSET)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        x = x + self.post_mixer_norm(self.mixer(self.pre_mixer_norm(x), cache))
        return x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))


class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = [DecoderLayer(args, layer) for layer in range(args.num_hidden_layers)]


class Backbone(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = Decoder(args)
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = Backbone(args)
        self.lm_head = nn.Linear(args.hidden_size, ((args.vocab_size + 15) // 16) * 16, bias=False)

    def make_cache(self, dtype: mx.Dtype | None = None) -> list[KVCache]:
        dtype = dtype or self.model.embed_tokens.weight.dtype
        return [
            KVCache(layer.mixer.kv_heads, layer.mixer.head_dim, layer.mixer.window, dtype)
            for layer in self.model.layers.layers
        ]

    def __call__(self, ids: mx.array, cache: list[KVCache], last_only: bool = True) -> mx.array:
        x = self.model.embed_tokens(ids)
        for layer, layer_cache in zip(self.model.layers.layers, cache):
            x = layer(x, layer_cache)
        x = self.model.norm(x)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(x)[..., : self.args.vocab_size]

    def prefill(self, ids: list[int], cache: list[KVCache], step: int = 512) -> None:
        """Run the prompt through the cache; decode then starts from the token after `ids`."""
        for start in range(0, len(ids), step):
            x = self.model.embed_tokens(mx.array(ids[start : start + step])[None])
            for layer, layer_cache in zip(self.model.layers.layers, cache):
                x = layer(x, layer_cache)
            mx.eval(x, [(c.keys, c.values) for c in cache])


def load_config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


def load_model(path: str | Path, dtype: mx.Dtype = mx.float16, lazy: bool = False) -> tuple[Model, dict]:
    path = Path(path)
    config = load_config(path)
    if config.get("model_type") != "plamo3":
        raise ValueError(f"Expected a PLaMo 3 checkpoint, got {config.get('model_type')}")
    if config.get("quantization"):
        raise ValueError(
            "PLaMo 3 needs original weights such as pfnet/plamo-3-translate-2b; they are quantized on load"
        )
    model = Model(ModelArgs.from_dict(config))
    weights = {}
    for shard in sorted(path.glob("model*.safetensors")):
        weights.update(mx.load(str(shard)))
    # Cast the two largest tensors on the CPU, which gives the same values and has no Metal time limit.
    on_cpu = {"model.embed_tokens.weight", "lm_head.weight"}
    model.load_weights(
        [(name, array.astype(dtype, stream=mx.cpu if name in on_cpu else None)) for name, array in weights.items()]
    )
    if not lazy:
        mx.eval(model.parameters())
    return model, config
