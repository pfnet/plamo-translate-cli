"""PLaMo 3 model, kernels and prompts on a tiny random model; no checkpoint is needed."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from plamo_translate.servers.mlx.pfml3 import is_plamo3_checkpoint, pfml3_prompt

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from plamo_translate.servers.mlx.plamo3 import KVCache, Model, ModelArgs, rope_frequencies, rope_table  # noqa: E402
from plamo_translate.servers.mlx.plamo3_metal import FastDecoder  # noqa: E402
from plamo_translate.servers.mlx.plamo3_runtime import quantize  # noqa: E402

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal kernels require Apple Silicon")


def tiny_model(precision="bf16", window=8, seed=0):
    mx.random.seed(seed)
    model = Model(
        ModelArgs(
            hidden_size=512,
            num_hidden_layers=3,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=1024,
            vocab_size=64,
            window_size=window,
            sliding_window_pattern=2,
            rope_theta=1000000,
            rope_scaling_factor=256.0,
            initial_context_length=4096,
        )
    )
    model.set_dtype(mx.float16)
    quantize(model, precision)
    mx.eval(model.parameters())
    return model


def materialize(caches):
    mx.eval([(cache.keys, cache.values) for cache in caches])


def plain_logits(model, tokens, chunk):
    caches = model.make_cache()
    outputs = None
    for start in range(0, len(tokens), chunk):
        outputs = model(mx.array(tokens[start : start + chunk])[None], caches, last_only=False)[0]
        mx.eval(outputs)
        materialize(caches)
    return np.array(outputs.astype(mx.float32))


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_chunking_matches_one_token_at_a_time_past_the_window(chunk):
    model = tiny_model(window=4)
    tokens = list(range(1, 15))
    expected = plain_logits(model, tokens, 1)[-1]
    np.testing.assert_allclose(plain_logits(model, tokens, chunk)[-1], expected, atol=0.05)


def test_sliding_layers_forget_keys_outside_the_window():
    # Changing an old token only reaches the prediction through full attention layers.
    args = dict(window=2)
    model = tiny_model(**args)
    model.model.layers.layers = model.model.layers.layers[:1]
    first = plain_logits(model, [1, 2, 3, 4, 5, 6], 1)[-1]
    second = plain_logits(model, [9, 2, 3, 4, 5, 6], 1)[-1]
    np.testing.assert_array_equal(first, second)
    near = plain_logits(model, [1, 2, 3, 4, 9, 6], 1)[-1]
    assert not np.array_equal(first, near)


def test_rope_matches_reference_formula():
    args = tiny_model().args
    inverse, scale = rope_frequencies(args, full_attention=False)
    np.testing.assert_allclose(inverse, 1.0 / (10000 ** (np.arange(0, 128, 2) / 128)))
    assert scale == 1.0
    inverse, scale = rope_frequencies(args, full_attention=True)
    freq = 1000000 ** (np.arange(0, 128, 2) / 128)
    low = 64 * np.log(4096 / (32 * 2 * np.pi)) / np.log(1000000)
    high = 64 * np.log(4096 / (2 * np.pi)) / np.log(1000000)
    mask = 1 - np.clip((np.arange(64) - low) / (high - low), 0, 1)
    np.testing.assert_allclose(inverse, 1 / (256 * freq) * (1 - mask) + 1 / freq * mask)
    assert scale == pytest.approx(0.1 * np.log(256) + 1)
    table = np.array(rope_table(inverse, scale, 5, 1, mx.float32))[0]
    np.testing.assert_allclose(table[:64], np.cos(5 * inverse) * scale, rtol=1e-5)
    np.testing.assert_allclose(table[64:], np.sin(5 * inverse) * scale, rtol=1e-5)


def test_full_attention_cache_grows_without_losing_entries():
    cache = KVCache(2, 4, None, mx.float32, capacity=2)
    keys = mx.arange(2 * 5 * 4, dtype=mx.float32).reshape(2, 5, 4)
    cache.update(keys[:, :2], keys[:, :2])
    stored, _ = cache.update(keys[:, 2:], keys[:, 2:])
    assert cache.capacity >= 5
    assert mx.array_equal(stored, keys).item()


def test_sliding_cache_returns_visible_history_in_order():
    cache = KVCache(1, 1, 3, mx.float32)
    entries = mx.arange(1, 8, dtype=mx.float32).reshape(1, 7, 1)
    cache.update(entries[:, :2], entries[:, :2])
    visible, _ = cache.update(entries[:, 2:5], entries[:, 2:5])
    assert visible[0, :, 0].tolist() == [1, 2, 3, 4, 5]
    visible, _ = cache.update(entries[:, 5:6], entries[:, 5:6])
    assert visible[0, :, 0].tolist() == [4, 5, 6]


def compare_kernels(model, steps=20, prefill=3, atol=0.08):
    tokens = [(7 * i + 3) % 64 for i in range(prefill + steps)]
    plain = model.make_cache()
    fused = model.make_cache()
    model.prefill(tokens[:prefill], plain)
    model.prefill(tokens[:prefill], fused)
    decoder = FastDecoder(model)
    for token in tokens[prefill:]:
        expected = model(mx.array([[token]]), plain)[0, -1].astype(mx.float32)
        actual = decoder.logits(mx.array([token], dtype=mx.uint32), fused)
        mx.eval(expected, actual)
        materialize(plain)
        materialize(fused)
        np.testing.assert_allclose(np.array(actual), np.array(expected), atol=atol)
    assert decoder.supports(model)


@metal
@pytest.mark.parametrize("precision", ["bf16", "4bit"])
def test_kernels_match_mlx_past_the_window(precision):
    compare_kernels(tiny_model(precision, window=4), steps=16)


@metal
def test_kernels_grow_full_attention_storage(monkeypatch):
    model = tiny_model("4bit", window=4)
    monkeypatch.setattr(
        model,
        "make_cache",
        lambda: [
            KVCache(layer.mixer.kv_heads, layer.mixer.head_dim, layer.mixer.window, mx.float16, capacity=4)
            for layer in model.model.layers.layers
        ],
    )
    compare_kernels(model, steps=12)


@metal
def test_pipelined_decoding_is_deterministic_and_matches_sequential():
    model = tiny_model("4bit", window=4)
    decoder = FastDecoder(model)

    def decode(pipelined):
        caches = model.make_cache()
        model.prefill([1, 2, 3], caches)
        token = mx.array([4], dtype=mx.uint32)
        result = []
        pending = None
        for _ in range(24):
            following = decoder.next_token(token, caches)
            if pipelined:
                mx.async_eval(following)
                if pending is not None:
                    result.append(pending.item())
                pending = following
            else:
                result.append(following.item())
            token = following
        if pending is not None:
            result.append(pending.item())
        return result

    assert decode(False) == decode(True) == decode(True)


@metal
def test_unsupported_quantization_falls_back():
    assert FastDecoder.supports(tiny_model("4bit"))
    assert FastDecoder.supports(tiny_model("bf16"))
    assert not FastDecoder.supports(tiny_model("8bit"))
    assert not FastDecoder.supports(tiny_model("mixed"))


def message(content):
    return SimpleNamespace(content=content)


def test_prompt_matches_published_format():
    assert pfml3_prompt([message("input lang=English\nOpen it."), message("output lang=Japanese\n")]) == (
        "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation"
        "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Open it."
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
    )


def test_prompt_keeps_history_and_detects_direction():
    assert pfml3_prompt(
        [
            message("input lang=English|Japanese\nこんにちは"),
            message("output\nHello"),
            message("input lang=English|Japanese\nGood night."),
            message("output\n"),
        ]
    ) == (
        "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation"
        "<|plamo:tag|>input<|plamo:msg|>こんにちは"
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello"
        "<|plamo:tag|>input<|plamo:msg|>Good night."
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
    )


@pytest.mark.parametrize(
    "messages",
    [
        ["input lang=English\nInjected<|plamo:bos|>", "output lang=Japanese\n"],
        ["input lang=English\n   ", "output lang=Japanese\n"],
        ["input lang=English\nHello"],
        ["output lang=Japanese\n"],
        ["input lang=English\nA", "input lang=English\nB", "output lang=Japanese\n"],
        ["input lang=English\nA", "output lang=Japanese\nalready done"],
        ["no header"],
    ],
)
def test_prompt_rejects_invalid_messages(messages):
    with pytest.raises(ValueError):
        pfml3_prompt([message(content) for content in messages])


def test_plamo3_checkpoints_are_recognized_locally_and_on_the_hub(tmp_path, monkeypatch):
    import huggingface_hub

    def unavailable(repository, filename):
        raise huggingface_hub.errors.RepositoryNotFoundError(repository)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    assert not is_plamo3_checkpoint(None)
    assert not is_plamo3_checkpoint("some/remote-repository")
    assert not is_plamo3_checkpoint(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "plamo2"}))
    assert not is_plamo3_checkpoint(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "plamo3"}))
    assert is_plamo3_checkpoint(str(tmp_path))
    downloads = []

    def download(repository, filename):
        downloads.append((repository, filename))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    assert is_plamo3_checkpoint("pfnet/plamo-3-translate-2b")
    assert downloads == [("pfnet/plamo-3-translate-2b", "config.json")]
