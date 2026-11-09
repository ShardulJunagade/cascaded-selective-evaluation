#!/usr/bin/env python3
"""Fail early when the installed vLLM and Transformers releases are incompatible."""
from importlib.metadata import PackageNotFoundError, version
import shutil


EXPECTED_VLLM = "0.27.1"
EXPECTED_TRANSFORMERS = "5.5.3"
FULL_REPAIR = (
    'python -m pip install --no-cache-dir --upgrade '
    '"vllm==0.27.1" "transformers==5.5.3"'
)
TRANSFORMERS_REPAIR = (
    'python -m pip install --no-cache-dir --upgrade "transformers==5.5.3"'
)


def main() -> int:
    if shutil.which("ninja") is None:
        raise SystemExit(
            "Missing `ninja`, required to compile FlashInfer kernels during vLLM warmup.\n"
            "Repair with:\n  python -m pip install --no-cache-dir ninja"
        )

    try:
        vllm_version = version("vllm")
        transformers_version = version("transformers")
    except PackageNotFoundError as error:
        raise SystemExit(
            f"Missing benchmark runtime package: {error.name}.\n"
            f"Repair with:\n  {FULL_REPAIR}"
        ) from error

    if vllm_version != EXPECTED_VLLM:
        raise SystemExit(
            "Unsupported judge runtime: "
            f"vllm={vllm_version}, transformers={transformers_version}.\n"
            f"Repair with:\n  {FULL_REPAIR}"
        )
    if transformers_version != EXPECTED_TRANSFORMERS:
        raise SystemExit(
            "Unsupported Transformers release for vLLM 0.27.1: "
            f"transformers={transformers_version}.\n"
            f"Repair with:\n  {TRANSFORMERS_REPAIR}"
        )

    try:
        # vLLM imports this while registering LlavaNext models. Importing both the
        # symbol and LLM catches package incompatibility before a model is loaded.
        from transformers.models.pixtral.modeling_pixtral import (  # noqa: F401
            PixtralRotaryEmbedding,
        )
        from vllm import LLM  # noqa: F401
    except (ImportError, RuntimeError) as error:
        raise SystemExit(
            "Incompatible vLLM/Transformers installation: "
            f"vllm={vllm_version}, transformers={transformers_version}.\n"
            f"Repair with:\n  {FULL_REPAIR}"
        ) from error

    print(
        "Judge runtime verified: "
        f"vllm={vllm_version}, transformers={transformers_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
