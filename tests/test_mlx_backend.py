"""Small numerical regression tests; no model downloads are needed."""

import numpy as np
import pytest
from dataclasses import asdict
from types import SimpleNamespace

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx_lm.models.cache import ArraysCache  # noqa: E402
from mlx_lm.models.plamo2 import Mamba, Model, ModelArgs  # noqa: E402

from plamo_translate.servers.mlx.loader import quantize_translation_model  # noqa: E402
from plamo_translate.servers.mlx.model import _fast_conv, _plamo_ssm, configure_model  # noqa: E402
from plamo_translate.servers.mlx import loader  # noqa: E402
from plamo_translate.servers.mlx.release import write_inference_assets  # noqa: E402


def tiny_model():
    return Model(
        ModelArgs(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size_per_head=16,
            mamba_num_heads=4,
            mamba_d_state=32,
            intermediate_size=128,
            vocab_size=64,
            tie_word_embeddings=False,
        )
    )


def test_checkpoint_rope_matches_reference_formula():
    model = tiny_model()
    config = {"model_type": "plamo2", "rope_theta": 10000, "rope_local_theta": 1000000}
    configure_model(model, config, optimize=False)
    x = mx.ones((1, 4, 1, 16))
    actual = model.layers[1].mixer.rope(x, offset=1000)
    theta = 1000 / (1000000 ** (np.arange(0, 16, 2) / 16))
    expected = np.concatenate([np.cos(theta) - np.sin(theta), np.cos(theta) + np.sin(theta)])
    np.testing.assert_allclose(np.array(actual)[0, 0, 0], expected, atol=5e-5)
    config["full_attention_idx"] = [1]
    configure_model(model, config, optimize=False)
    assert not mx.allclose(actual, model.layers[1].mixer.rope(x, offset=1000)).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_decode_convolution_and_state_match_upstream(dtype):
    if not mx.metal.is_available():
        pytest.skip("Metal kernel requires Apple Silicon")
    mx.random.seed(42)
    mixer = tiny_model().layers[0].mixer
    mixer.conv1d.weight = mixer.conv1d.weight.astype(dtype)
    initial = mx.random.normal((1, 3, 64)).astype(dtype)
    expected_cache, actual_cache = ArraysCache(size=2), ArraysCache(size=2)
    expected_cache[0] = initial
    actual_cache[0] = initial
    for _ in range(8):
        x = mx.random.normal((1, 1, 64)).astype(dtype)
        expected = Mamba._conv(mixer, x, expected_cache, None)
        actual = _fast_conv(mixer, x, actual_cache, None)
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected.astype(mx.float32)),
            atol=0.008 if dtype == mx.bfloat16 else 0.001 if dtype == mx.float16 else 1e-6,
            rtol=0.008 if dtype == mx.bfloat16 else 0.001 if dtype == mx.float16 else 1e-5,
        )
        assert mx.array_equal(expected_cache[0], actual_cache[0]).item()


@pytest.mark.parametrize("length", [1, 5])
def test_convolution_empty_cache_falls_back(length):
    mixer = tiny_model().layers[0].mixer
    x = mx.random.normal((1, length, 64))
    expected_cache, actual_cache = ArraysCache(size=2), ArraysCache(size=2)
    expected = Mamba._conv(mixer, x, expected_cache, None)
    actual = _fast_conv(mixer, x, actual_cache, None)
    assert mx.array_equal(expected, actual).item()
    assert mx.array_equal(expected_cache[0], actual_cache[0]).item()


def test_quantized_precision_cannot_be_silently_changed():
    model = tiny_model()
    config = {"model_type": "plamo2"}
    model, config = quantize_translation_model(model, config, "8bit")
    assert isinstance(model.lm_head, nn.QuantizedLinear)
    assert quantize_translation_model(model, config, "8bit")[0] is model
    with pytest.raises(ValueError, match="Checkpoint is 8bit"):
        quantize_translation_model(model, config, "4bit")
    with pytest.raises(ValueError, match="original weights"):
        quantize_translation_model(model, config, "bf16")


def test_mixed_precision_preserves_recurrent_projections():
    model, config = quantize_translation_model(tiny_model(), {"model_type": "plamo2"}, "mixed")
    assert model.layers[0].mlp.gate_up_proj.bits == 4
    assert model.layers[0].mixer.in_proj.bits == 8
    assert isinstance(model.layers[0].mixer.bcdt_proj, nn.Linear)
    assert isinstance(model.layers[0].mixer.dt_proj, nn.Linear)
    assert config["plamo_translate_precision"] == "mixed"


def test_ssm_preserves_small_time_steps_and_fp32_state():
    mixer = tiny_model().layers[0].mixer
    mixer.A_log = mx.zeros((4,))
    mixer.D = mx.zeros((4,))
    mixer.dt_bias = mx.zeros((4,))
    x = mx.ones((1, 1, 64))
    B = C = mx.ones((1, 1, 32))
    dt = mx.full((1, 1, 4), -20.0)
    cache = ArraysCache(size=2)
    actual = _plamo_ssm(mixer, x, B, C, dt, cache, None)
    # Reference recurrence: zero state -> softplus(-20) * B * x, then dot C.
    np.testing.assert_allclose(np.array(actual), np.logaddexp(0, -20) * 32, rtol=1e-5)
    assert cache[1].dtype == mx.float32
    actual = _plamo_ssm(mixer, x, B, C, dt, cache, None)
    np.testing.assert_allclose(np.array(actual), np.logaddexp(0, -20) * 64, rtol=1e-5)


@pytest.mark.parametrize("saved,requested,expected", [("bf16", None, "bf16"), (None, None, "4bit"),
                                                  ("bf16", "8bit", "8bit")])
def test_loader_preserves_saved_bf16_unless_explicitly_overridden(monkeypatch, saved, requested, expected):
    model = tiny_model()
    config = {"model_type": "plamo2"}
    if saved:
        config["plamo_translate_precision"] = saved
    tokenizer = SimpleNamespace(add_eos_token=lambda token: None)
    monkeypatch.setattr(loader, "load", lambda *args, **kwargs: (model, tokenizer, config))
    loaded, _, updated = loader.load_translation_model("unused", requested)
    assert updated["plamo_translate_precision"] == expected
    assert isinstance(loaded.lm_head, nn.QuantizedLinear) == (expected != "bf16")


def test_standalone_export_matches_corrected_cli_model(tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    model = tiny_model()
    config = {**asdict(model.config), "model_type": "plamo2", "rope_local_theta": 1000000, "rope_theta": 10000}
    configure_model(model, config)
    (tmp_path / "tokenizer_config.json").write_text("{}")
    write_inference_assets(tmp_path, config, [1, 4])
    mx.save_safetensors(str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    exported, _ = load_model(tmp_path)
    assert type(exported).__module__ == "custom_model"
    tokens = mx.array([[1, 2, 3, 4, 5]])
    np.testing.assert_allclose(np.array(model(tokens)), np.array(exported(tokens)), atol=1e-6, rtol=1e-6)
