"""Single-token PLaMo 3 decode built from fused Metal kernels.

Every token costs exactly seven dispatches per layer. Matrix-vector products use
one SIMD-group per four rows. Normalization and residuals share a dispatch, and
SwiGLU is computed by the thread that finishes its row pair. KV entries are written
in place, which is safe because every dispatch of a token depends on the previous one.

Greedy decode needs no CPU work between tokens: the sampled token is a GPU array that
the next embedding dispatch reads, so the following step can be encoded before it is known.
"""

import mlx.core as mx
import mlx.nn as nn

from plamo_translate.servers.mlx.plamo3 import (
    POST_MIXER_OFFSET,
    POST_MLP_OFFSET,
    KVCache,
    Model,
    rope_frequencies,
    rope_table,
)

_ROWS_PER_THREADGROUP = 8
_THREADS_PER_THREADGROUP = 64
_BLOCK = 512
_ATTENTION_GROUPS = 32
_NORM_GROUPS = 16

_MATVEC_PRELUDE = """
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup * 4;
"""

_LOAD_X = """
const device T* xl = x + k + lane * 16;
float sum = 0;
float xt[16];
for (uint i = 0; i < 4; ++i) {
    float x0 = xl[4 * i], x1 = xl[4 * i + 1], x2 = xl[4 * i + 2], x3 = xl[4 * i + 3];
    sum += x0 + x1 + x2 + x3;
    xt[4 * i] = x0;
    xt[4 * i + 1] = x1 / 16.0f;
    xt[4 * i + 2] = x2 / 256.0f;
    xt[4 * i + 3] = x3 / 4096.0f;
}
"""

# Masks instead of shifts, with x pre-scaled by 16^-i, leave one AND and one convert per weight.
# This exact expression form (two uint loads, one sum) measured fastest of the forms tried on M1 Max.
_DOT_Q4 = """
inline float qdot4(const device uint* w, thread const float* xt, float sum, float scale, float bias) {
    uint a = w[0], c = w[1];
    float accum = xt[0] * float(a & 15u) + xt[1] * float(a & 0xf0u) + xt[2] * float(a & 0xf00u) + xt[3] * float(a & 0xf000u)
        + xt[4] * float((a >> 16) & 15u) + xt[5] * float((a >> 16) & 0xf0u) + xt[6] * float((a >> 16) & 0xf00u) + xt[7] * float(a >> 28 << 12)
        + xt[8] * float(c & 15u) + xt[9] * float(c & 0xf0u) + xt[10] * float(c & 0xf00u) + xt[11] * float(c & 0xf000u)
        + xt[12] * float((c >> 16) & 15u) + xt[13] * float((c >> 16) & 0xf0u) + xt[14] * float((c >> 16) & 0xf00u) + xt[15] * float(c >> 28 << 12);
    return scale * accum + bias * sum;
}
"""

_DOT_FP16 = """
inline float dot16(const device half* w, thread const float* xt) {
    float accum = 0;
    for (uint i = 0; i < 16; ++i) accum += xt[i] * float(w[i]);
    return accum;
}
"""

_LOAD_X_DENSE = """
const device T* xl = x + k + lane * 16;
float xt[16];
for (uint i = 0; i < 16; ++i) xt[i] = xl[i];
"""


def _accumulate(rows: list[tuple[str, str]], quantized: bool) -> str:
    """Add this block's contribution to each (accumulator, first row of its half) pair."""
    lines = []
    for index, (accumulator, offset) in enumerate(rows):
        row = f"(row + {index % 4} + {offset})"
        if quantized:
            lines.append(
                f"{accumulator} += qdot4(w + {row} * (cols / 8) + k / 8 + lane * 2, xt, sum, "
                f"scales[{row} * (cols / 64) + k / 64 + lane / 4], biases[{row} * (cols / 64) + k / 64 + lane / 4]);"
            )
        else:
            lines.append(f"{accumulator} += dot16(w + {row} * cols + k + lane * 16, xt);")
    return "\n    ".join(lines)


def _matvec_source(quantized: bool) -> str:
    rows = [(f"r{i}", "0") for i in range(4)]
    return f"""{_MATVEC_PRELUDE}
float r0 = 0, r1 = 0, r2 = 0, r3 = 0;
for (uint k = 0; k < cols; k += {_BLOCK}) {{
    {_LOAD_X if quantized else _LOAD_X_DENSE}
    {_accumulate(rows, quantized)}
}}
r0 = simd_sum(r0); r1 = simd_sum(r1); r2 = simd_sum(r2); r3 = simd_sum(r3);
if (lane == 0) {{
    y[row] = Y(r0); y[row + 1] = Y(r1); y[row + 2] = Y(r2); y[row + 3] = Y(r3);
}}
"""


def _gate_source(quantized: bool) -> str:
    rows = [(f"g{i}", "0") for i in range(4)] + [(f"u{i}", "half_rows") for i in range(4)]
    return f"""{_MATVEC_PRELUDE}
float g0 = 0, g1 = 0, g2 = 0, g3 = 0, u0 = 0, u1 = 0, u2 = 0, u3 = 0;
for (uint k = 0; k < cols; k += {_BLOCK}) {{
    {_LOAD_X if quantized else _LOAD_X_DENSE}
    {_accumulate(rows, quantized)}
}}
g0 = simd_sum(g0); g1 = simd_sum(g1); g2 = simd_sum(g2); g3 = simd_sum(g3);
u0 = simd_sum(u0); u1 = simd_sum(u1); u2 = simd_sum(u2); u3 = simd_sum(u3);
float g = lane == 0 ? g0 : lane == 1 ? g1 : lane == 2 ? g2 : g3;
float u = lane == 0 ? u0 : lane == 1 ? u1 : lane == 2 ? u2 : u3;
if (lane < 4) y[row + lane] = T(g / (1.0f + exp(-g)) * u);
"""


# Reductions go through simd_sum, then across the threadgroup's SIMD-groups in one more simd_sum.
_REDUCE = """
sum = simd_sum(sum);
if (lane == 0) partial[sg] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
s = rsqrt(simd_sum(lane < groups ? partial[lane] : 0.0f) / float(n) + eps);
threadgroup_barrier(mem_flags::mem_threadgroup);
"""

_NORM_SOURCE = f"""
threadgroup float partial[groups];
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint at = sg * 32 + lane;
uint steps = n / (groups * 32);
float sum = 0, s;
for (uint i = 0; i < steps; ++i) {{ float v = a[at + i * groups * 32]; sum += v * v; }}
{_REDUCE}
sum = 0;
for (uint i = 0; i < steps; ++i) {{
    uint j = at + i * groups * 32;
    T t = T(float(b[j]) + float(T(float(T(float(a[j]) * s)) * wa[j])));
    r[j] = t;
    sum += float(t) * float(t);
}}
{_REDUCE}
for (uint i = 0; i < steps; ++i) {{
    uint j = at + i * groups * 32;
    normed[j] = T(float(T(float(r[j]) * s)) * wb[j]);
}}
"""

_EMBED_SOURCE = f"""
threadgroup float partial[groups];
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint at = sg * 32 + lane;
uint steps = n / (groups * 32);
const device T* e = table + uint(token[0]) * n;
float sum = 0, s;
for (uint i = 0; i < steps; ++i) {{ float v = e[at + i * groups * 32]; sum += v * v; }}
{_REDUCE}
for (uint i = 0; i < steps; ++i) {{
    uint j = at + i * groups * 32;
    T v = e[j];
    r[j] = v;
    normed[j] = T(float(T(float(v) * s)) * wb[j]);
}}
"""

# One threadgroup per query head: logit and value accumulation share one pass over the keys,
# so every core stays busy and nothing needs per-key storage. The query's own key and value
# come straight from qkv, leaving no overlap between the KV slots being read and written.
_ATTENTION_SOURCE = """
threadgroup float qh[128], ks[128], partial[groups * 128], denoms[groups];
uint head = threadgroup_position_in_grid.x;
uint group = head / 6;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint slot = params[1], stored = params[2], cap = params[3];
const device T* table = rope + params[0] * 128;
const device T* own_key = qkv + q_dim + group * 128;
const device T* own_value = qkv + q_dim + kv_dim + group * 128 + lane;
if (sg < 2) {
    const device T* src = sg == 0 ? qkv + head * 128 : own_key;
    const device float* nw = sg == 0 ? q_weight : k_weight;
    float a0 = src[lane], a1 = src[lane + 32], a2 = src[lane + 64], a3 = src[lane + 96];
    float s = rsqrt(simd_sum(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) / 128.0f + eps);
    a0 = float(T(float(T(a0 * s)) * nw[lane]));
    a1 = float(T(float(T(a1 * s)) * nw[lane + 32]));
    a2 = float(T(float(T(a2 * s)) * nw[lane + 64]));
    a3 = float(T(float(T(a3 * s)) * nw[lane + 96]));
    float c0 = table[lane], c1 = table[lane + 32], s0 = table[lane + 64], s1 = table[lane + 96];
    T r0 = T(a0 * c0 - a2 * s0), r1 = T(a1 * c1 - a3 * s1), r2 = T(a2 * c0 + a0 * s0), r3 = T(a3 * c1 + a1 * s1);
    if (sg == 0) {
        qh[lane] = r0; qh[lane + 32] = r1; qh[lane + 64] = r2; qh[lane + 96] = r3;
    } else {
        ks[lane] = r0; ks[lane + 32] = r1; ks[lane + 64] = r2; ks[lane + 96] = r3;
        if (head % 6 == 0) {
            device T* kw = const_cast<device T*>(keys) + (group * cap + slot) * 128 + lane;
            kw[0] = r0; kw[32] = r1; kw[64] = r2; kw[96] = r3;
            device T* vw = const_cast<device T*>(values) + (group * cap + slot) * 128 + lane;
            vw[0] = own_value[0]; vw[32] = own_value[32]; vw[64] = own_value[64]; vw[96] = own_value[96];
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float q0 = qh[lane], q1 = qh[lane + 32], q2 = qh[lane + 64], q3 = qh[lane + 96];
// Softmax relative to the query's own key: exp cannot overflow and the denominator stays positive.
float bias = simd_sum(q0 * ks[lane] + q1 * ks[lane + 32] + q2 * ks[lane + 64] + q3 * ks[lane + 96]) * attention_scale;
bool first = sg == 0;
float a0 = first ? float(own_value[0]) : 0.0f, a1 = first ? float(own_value[32]) : 0.0f;
float a2 = first ? float(own_value[64]) : 0.0f, a3 = first ? float(own_value[96]) : 0.0f;
float den = first ? 1.0f : 0.0f;
for (uint j = sg; j < stored; j += groups) {
    if (j == slot) continue;
    const device T* kr = keys + (group * cap + j) * 128 + lane;
    float p = exp(min(simd_sum(q0 * kr[0] + q1 * kr[32] + q2 * kr[64] + q3 * kr[96]) * attention_scale - bias, 70.0f));
    const device T* vr = values + (group * cap + j) * 128 + lane;
    den += p;
    a0 += p * vr[0]; a1 += p * vr[32]; a2 += p * vr[64]; a3 += p * vr[96];
}
uint at = sg * 128 + lane;
partial[at] = a0; partial[at + 32] = a1; partial[at + 64] = a2; partial[at + 96] = a3;
if (lane == 0) denoms[sg] = den;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg < 4) {
    float total = simd_sum(lane < groups ? denoms[lane] : 0.0f);
    float acc = 0;
    for (uint i = 0; i < groups; ++i) acc += partial[i * 128 + sg * 32 + lane];
    out[head * 128 + sg * 32 + lane] = T(acc / total);
}
"""


def _is_quantized(linear) -> bool:
    return "scales" in linear


def _projections(model: Model) -> list[nn.Module]:
    projections = [model.lm_head]
    for layer in model.model.layers.layers:
        projections += [layer.mixer.qkv_proj, layer.mixer.o_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj]
    return projections


class FastDecoder:
    @staticmethod
    def supports(model: Model) -> bool:
        """The kernels decode dense FP16 or 4-bit, group-size 64 weights on a Metal GPU."""
        args = model.args
        q_dim = args.head_dim * args.num_attention_heads
        return (
            mx.metal.is_available()
            and mx.default_device() == mx.gpu
            and model.model.embed_tokens.weight.dtype == mx.float16
            and args.head_dim == 128
            and args.num_attention_heads == 6 * args.num_key_value_heads
            and all(cols % _BLOCK == 0 for cols in (args.hidden_size, args.intermediate_size, q_dim))
            and all(
                rows % _ROWS_PER_THREADGROUP == 0 for rows in (args.intermediate_size, model.lm_head.weight.shape[0])
            )
            and all(
                (p.bits, p.group_size) == (4, 64) if _is_quantized(p) else p.weight.dtype == mx.float16
                for p in _projections(model)
            )
        )

    def __init__(self, model: Model):
        self.model = model
        self.args = args = model.args
        self.dtype = mx.float16
        self.n = args.hidden_size
        layers = model.model.layers.layers
        self.layers = layers
        self.embedding = model.model.embed_tokens.weight
        self.tables = []
        for index, layer in enumerate(layers):
            following = layers[index + 1].pre_mixer_norm if index + 1 < len(layers) else model.model.norm
            self.tables.append(
                dict(
                    post_mixer=(layer.post_mixer_norm.weight + POST_MIXER_OFFSET).astype(mx.float32),
                    pre_mlp=(layer.pre_mlp_norm.weight + 1).astype(mx.float32),
                    post_mlp=(layer.post_mlp_norm.weight + POST_MLP_OFFSET).astype(mx.float32),
                    following=(following.weight + 1).astype(mx.float32),
                    q=(layer.mixer.q_norm.weight + 1).astype(mx.float32),
                    k=(layer.mixer.k_norm.weight + 1).astype(mx.float32),
                )
            )
        self.first_norm = (layers[0].pre_mixer_norm.weight + 1).astype(mx.float32)
        scale = args.head_dim**-0.5
        mx.eval([list(table.values()) for table in self.tables], self.first_norm)

        self.rope_capacity = 0
        self._rope = {}
        self._ensure_rope(8192)

        constants = (
            f"constant float eps = {float(args.rms_norm_eps)!r}f;\nconstant float attention_scale = {scale!r}f;\n"
        )
        matvec = dict(input_names=["w", "scales", "biases", "x"], output_names=["y"])
        dense = dict(input_names=["w", "x"], output_names=["y"])
        self._qmv = mx.fast.metal_kernel(name="plamo3_qmv", header=_DOT_Q4, source=_matvec_source(True), **matvec)
        self._qgate = mx.fast.metal_kernel(name="plamo3_qgate", header=_DOT_Q4, source=_gate_source(True), **matvec)
        self._dmv = mx.fast.metal_kernel(name="plamo3_dmv", header=_DOT_FP16, source=_matvec_source(False), **dense)
        self._dgate = mx.fast.metal_kernel(name="plamo3_dgate", header=_DOT_FP16, source=_gate_source(False), **dense)
        self._norm = mx.fast.metal_kernel(
            name="plamo3_norm",
            input_names=["a", "b", "wa", "wb"],
            output_names=["r", "normed"],
            header=constants,
            source=_NORM_SOURCE,
        )
        self._embed = mx.fast.metal_kernel(
            name="plamo3_embed",
            input_names=["table", "token", "wb"],
            output_names=["r", "normed"],
            header=constants,
            source=_EMBED_SOURCE,
        )
        self._attention = mx.fast.metal_kernel(
            name="plamo3_attention",
            input_names=["qkv", "keys", "values", "rope", "params", "q_weight", "k_weight"],
            output_names=["out"],
            header=constants,
            source=_ATTENTION_SOURCE,
        )

    def _ensure_rope(self, length: int) -> None:
        if length <= self.rope_capacity:
            return
        self.rope_capacity = max(length, self.rope_capacity * 2)
        for full in (False, True):
            inverse, scale = rope_frequencies(self.args, full)
            self._rope[full] = rope_table(inverse, scale, 0, self.rope_capacity, self.dtype)
        mx.eval(list(self._rope.values()))

    def _mv(self, linear, x: mx.array, gate: bool = False, output: mx.Dtype | None = None) -> mx.array:
        rows, cols = linear["weight"].shape if not _is_quantized(linear) else (linear["scales"].shape[0], x.shape[0])
        output = output or self.dtype
        template = [("T", self.dtype), ("Y", output), ("cols", cols)]
        if _is_quantized(linear):
            inputs = [linear["weight"], linear["scales"], linear["biases"], x]
            kernel = self._qgate if gate else self._qmv
        else:
            inputs = [linear["weight"], x]
            kernel = self._dgate if gate else self._dmv
        out_rows = rows // 2 if gate else rows
        if gate:
            template.append(("half_rows", out_rows))
        groups = out_rows // _ROWS_PER_THREADGROUP
        return kernel(
            inputs=inputs,
            template=template,
            grid=(groups * _THREADS_PER_THREADGROUP, 1, 1),
            threadgroup=(_THREADS_PER_THREADGROUP, 1, 1),
            output_shapes=[(out_rows,)],
            output_dtypes=[output],
        )[0]

    def _pair(self, a: mx.array, b: mx.array, wa: mx.array, wb: mx.array) -> list[mx.array]:
        return self._norm(
            inputs=[a, b, wa, wb],
            template=[("T", self.dtype), ("n", self.n), ("groups", _NORM_GROUPS)],
            grid=(_NORM_GROUPS * 32, 1, 1),
            threadgroup=(_NORM_GROUPS * 32, 1, 1),
            output_shapes=[(self.n,), (self.n,)],
            output_dtypes=[self.dtype, self.dtype],
        )

    def _step_attention(self, layer, table, qkv, cache: KVCache, params: mx.array) -> mx.array:
        mixer = layer.mixer
        return self._attention(
            inputs=[qkv, cache.keys, cache.values, self._rope[mixer.full_attention], params, table["q"], table["k"]],
            template=[
                ("T", self.dtype),
                ("groups", _ATTENTION_GROUPS),
                ("q_dim", mixer.q_heads * mixer.head_dim),
                ("kv_dim", mixer.kv_heads * mixer.head_dim),
            ],
            grid=(_ATTENTION_GROUPS * 32 * mixer.q_heads, 1, 1),
            threadgroup=(_ATTENTION_GROUPS * 32, 1, 1),
            output_shapes=[(mixer.q_heads * mixer.head_dim,)],
            output_dtypes=[self.dtype],
        )[0]

    def logits(self, token: mx.array, caches: list[KVCache]) -> mx.array:
        """FP32 scores following a one-element uint32 `token`, appended to `caches`."""
        position = caches[0].offset
        self._ensure_rope(position + 1)
        if any(not cache.window and position >= cache.capacity for cache in caches):
            # Pending dispatches write the old storage in place; finish them before copying.
            mx.eval(token)
            for cache in caches:
                cache.reserve(position + 1)
            mx.eval([(cache.keys, cache.values) for cache in caches])
        window = self.args.window_size
        sliding_params = mx.array([position, position % window, min(position, window), window], dtype=mx.uint32)
        x, normed = self._embed(
            inputs=[self.embedding, token, self.first_norm],
            template=[("T", self.dtype), ("n", self.n), ("groups", _NORM_GROUPS)],
            grid=(_NORM_GROUPS * 32, 1, 1),
            threadgroup=(_NORM_GROUPS * 32, 1, 1),
            output_shapes=[(self.n,), (self.n,)],
            output_dtypes=[self.dtype, self.dtype],
        )
        for layer, table, cache in zip(self.layers, self.tables, caches):
            if cache.window:
                params = sliding_params
            else:
                params = mx.array([position, position, position, cache.capacity], dtype=mx.uint32)
            qkv = self._mv(layer.mixer.qkv_proj, normed)
            attended = self._step_attention(layer, table, qkv, cache, params)
            cache.offset += 1
            projected = self._mv(layer.mixer.o_proj, attended)
            x, normed = self._pair(projected, x, table["post_mixer"], table["pre_mlp"])
            hidden = self._mv(layer.mlp.gate_up_proj, normed, gate=True)
            projected = self._mv(layer.mlp.down_proj, hidden)
            x, normed = self._pair(projected, x, table["post_mlp"], table["following"])
        return self._mv(self.model.lm_head, normed, output=mx.float32)[: self.args.vocab_size]

    def next_token(self, token: mx.array, caches: list[KVCache]) -> mx.array:
        return mx.argmax(self.logits(token, caches), keepdims=True).astype(mx.uint32)
