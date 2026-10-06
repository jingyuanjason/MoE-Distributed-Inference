"""Launch vLLM's OpenAI-compatible server from an external YAML file."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

import yaml


def _find_bundled_cuda_home() -> Path | None:
    """Locate the CUDA toolkit shipped inside this Python environment.

    The CUDA builds of vLLM/PyTorch install an ``nvcc`` under
    ``site-packages/nvidia/cu13``. JIT kernel compilation (flashinfer and
    similar) requires CUDA 12.8+ (it passes ``--compress-mode=size``), which
    is often newer than the system toolkit found on ``PATH``.
    """
    for key in ("purelib", "platlib"):
        candidate = Path(sysconfig.get_path(key)) / "nvidia" / "cu13"
        if (candidate / "bin" / "nvcc").is_file():
            return candidate
    return None


class VLLMServer:
    """Read a vLLM configuration; call serve() to load the model and serve it.

    YAML keys use vLLM's long-form CLI names (for example,
    ``tensor-parallel-size``). vLLM validates engine-specific parameters.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve(strict=True)
        with self.config_path.open(encoding="utf-8") as config_file:
            self.parameters = yaml.safe_load(config_file)

        if not isinstance(self.parameters, dict):
            raise ValueError("The YAML configuration must be a mapping.")
        if not all(isinstance(key, str) for key in self.parameters):
            raise ValueError("Configuration keys must be strings.")
        model = self.parameters.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("The configuration must contain a non-empty 'model' string.")

    def serve(self) -> None:
        """Run in the foreground, inheriting console output and raising on failure.

        vLLM reads the original YAML at launch, so do not modify it between
        construction and this call. Returning from construction does not mean
        that the model is loaded or that the HTTP endpoint is ready.
        """
        executable = shutil.which("vllm")
        if executable is None:
            raise RuntimeError(
                "vLLM is not available on PATH. Install vLLM in your serving "
                "environment and activate that environment before launching."
            )
        env = None
        if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
            bundled = _find_bundled_cuda_home()
            if bundled is not None:
                env = {**os.environ, "CUDA_HOME": str(bundled)}
        subprocess.run(
            [executable, "serve", "--config", str(self.config_path)],
            check=True,
            env=env,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="vLLM YAML file")
    args = parser.parse_args()
    VLLMServer(args.config).serve()


if __name__ == "__main__":
    main()