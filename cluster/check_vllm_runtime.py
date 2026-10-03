#!/usr/bin/env python3
"""Fail early when the installed vLLM and Transformers releases are incompatible."""
from importlib.metadata import PackageNotFoundError, version


REPAIR = (
    'python -m pip install --no-cache-dir --upgrade '
    '"transformers==4.57.1"'
)


def main() -> int:
    try:
        vllm_version = version("vllm")
        transformers_version = version("transformers")
    except PackageNotFoundError as error:
        raise SystemExit(
            f"Missing benchmark runtime package: {error.name}.\nRepair with:\n  {REPAIR}"
        ) from error

    try:
        # vLLM 0.11.0 imports this while registering LlavaNext models. Importing it
        # here catches incompatible Transformers releases before a model is loaded.
        from transformers.models.pixtral.modeling_pixtral import (  # noqa: F401
            PixtralRotaryEmbedding,
        )
    except ImportError as error:
        raise SystemExit(
            "Incompatible vLLM/Transformers installation: "
            f"vllm={vllm_version}, transformers={transformers_version}.\n"
            f"Repair with:\n  {REPAIR}"
        ) from error

    print(
        "Judge runtime verified: "
        f"vllm={vllm_version}, transformers={transformers_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
