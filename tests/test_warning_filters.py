import warnings

from plamo_translate.servers.warnings import (
    OPTIONAL_GPU_DEPENDENCY_WARNING_MESSAGES,
    build_optional_gpu_dependency_warning_options,
    suppress_optional_gpu_dependency_warnings,
)


def test_build_optional_gpu_dependency_warning_options():
    assert build_optional_gpu_dependency_warning_options() == [
        "-W",
        "ignore:mamba_ssm could not be imported:UserWarning",
        "-W",
        "ignore:causal_conv1d could not be imported:UserWarning",
    ]


def test_suppress_optional_gpu_dependency_warnings_only_hides_known_messages():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with suppress_optional_gpu_dependency_warnings():
            for message in OPTIONAL_GPU_DEPENDENCY_WARNING_MESSAGES:
                warnings.warn(message, UserWarning, stacklevel=1)
            warnings.warn("unexpected warning", UserWarning, stacklevel=1)

    assert [str(item.message) for item in captured] == ["unexpected warning"]
