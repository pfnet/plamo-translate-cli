import contextlib
import re
import warnings
from collections.abc import Iterator

OPTIONAL_GPU_DEPENDENCY_WARNING_MESSAGES = (
    "mamba_ssm could not be imported",
    "causal_conv1d could not be imported",
)


@contextlib.contextmanager
def suppress_optional_gpu_dependency_warnings() -> Iterator[None]:
    """Hide known optional dependency warnings emitted by remote model code."""
    with warnings.catch_warnings():
        for message in OPTIONAL_GPU_DEPENDENCY_WARNING_MESSAGES:
            warnings.filterwarnings(
                action="ignore",
                message=rf"^{re.escape(message)}$",
                category=UserWarning,
            )
        yield


def build_optional_gpu_dependency_warning_options() -> list[str]:
    """Build `python -W` options that suppress known optional dependency warnings."""
    options: list[str] = []
    for message in OPTIONAL_GPU_DEPENDENCY_WARNING_MESSAGES:
        options.extend(["-W", f"ignore:{message}:UserWarning"])
    return options
