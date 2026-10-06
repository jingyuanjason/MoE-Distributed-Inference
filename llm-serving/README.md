# YAML-configured vLLM serving

## Docker images (NVIDIA GPUs)

The dependency image uses the official `vllm/vllm-openai` CUDA image, which
includes vLLM, PyTorch and PyYAML, and installs Tini for signal forwarding.
The application image adds the Python wrapper and a default YAML config.
Model weights are downloaded at runtime, not baked into either image.

Build the base image, then the application image:

```bash
docker build -t moe-vllm-base:local \
  -f /home/jingyuan_li/MoE-Distributed-Inference/llm-serving/dockerfiles/baseimage/Dockerfile \
  /home/jingyuan_li/MoE-Distributed-Inference/llm-serving

docker build -t moe-vllm-app:local \
  --build-arg BASE_IMAGE=moe-vllm-base:local \
  -f /home/jingyuan_li/MoE-Distributed-Inference/llm-serving/dockerfiles/appimage/Dockerfile \
  /home/jingyuan_li/MoE-Distributed-Inference/llm-serving
```

The upstream default is `vllm/vllm-openai:latest`. For production, pass
`--build-arg VLLM_IMAGE=<release-tag-or-digest-reference>` to the base build
to pin a version compatible with your GPU and host NVIDIA driver. Both
images use the serving directory as their build context. Rebuild the base
only when dependencies change; rebuild the app when code/config changes.

The host needs an NVIDIA driver and NVIDIA Container Toolkit configured for
Docker. Run with GPU access, shared memory for tensor parallelism, and a
persistent model cache:

```bash
docker run --rm --name moe-vllm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v moe-hf-cache:/root/.cache/huggingface \
  moe-vllm-app:local
```

The bundled container config binds to `0.0.0.0:8000` inside the container;
the command above publishes it only on the host's loopback interface.
For your own config, mount a YAML file read-only and override the default
arguments (the following uses the provided container config as an example):

```bash
docker run --rm --name moe-vllm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v moe-hf-cache:/root/.cache/huggingface \
  --mount type=bind,source=/home/jingyuan_li/MoE-Distributed-Inference/llm-serving/dockerfiles/appimage/config.yaml,target=/config/server.yaml,readonly \
  moe-vllm-app:local --config /config/server.yaml
```

Keep `host: "0.0.0.0"` in container configs, and adjust the port mapping if
you change the YAML port. Local model paths must be mounted into the
container and referenced by their container paths. For gated models, pass
`-e HF_TOKEN` with `HF_TOKEN` set in your host environment; never bake tokens
into images. Protect the API before publishing it beyond localhost.

Check readiness at `http://127.0.0.1:8000/health` after model loading. Stop
with `docker stop --timeout 60 moe-vllm`; Tini forwards SIGTERM to the wrapper
and its server child. A GPU-free entrypoint check is:

```bash
docker run --rm moe-vllm-app:local --help
```

## Running directly (uv)

This directory is a [uv](https://docs.astral.sh/uv/) project: `pyproject.toml`
declares **vLLM** and **PyYAML**, and `.python-version` pins Python 3.12 (the
default CUDA vLLM wheels do not yet support newer interpreters). With uv
installed, create the environment and install all locked dependencies with:

```bash
cd /home/jingyuan_li/MoE-Distributed-Inference/llm-serving
uv sync
```

`uv sync` creates `.venv/`, downloads a managed Python 3.12 if needed, and
installs the CUDA build of vLLM and PyTorch (several GB) from `uv.lock`.
For a non-CUDA accelerator, replace the `vllm` dependency with the build
appropriate for your hardware:
https://docs.vllm.ai/en/stable/getting_started/installation/.

Edit `/home/jingyuan_li/MoE-Distributed-Inference/llm-serving/config.example.yaml`
or supply your own YAML file, then run the server through uv (no manual
activation needed; `uv run` puts the `vllm` executable on `PATH`):

```bash
cd /home/jingyuan_li/MoE-Distributed-Inference/llm-serving
uv run python vllm_server.py --config config.example.yaml
```

If vLLM just-in-time compiles kernels (flashinfer, DeepGEMM, and similar), it
invokes the `nvcc` found via `CUDA_HOME`/`CUDA_PATH` or `PATH`. The flag set
requires CUDA 12.8 or newer (`--compress-mode=size`); an older system toolkit
fails with `nvcc fatal : Unknown option '--compress-mode=size'`. When neither
variable is set, `vllm_server.py` automatically points the server subprocess
at the CUDA 13 toolkit installed in the project environment
(`.venv/lib/python3.12/site-packages/nvidia/cu13`). Set `CUDA_HOME` yourself
only to override that choice, for example to use a different toolkit.

The bundled toolkit's compiler (nvcc 13.4) must match its CUDA runtime
headers; a mismatch fails JIT builds with "CUDA compiler and CUDA toolkit
headers are incompatible". `pyproject.toml` therefore pins
`nvidia-cuda-runtime==13.4.92` via a uv `override-dependencies` entry (vllm's
own constraint would otherwise resolve the older 13.0 headers). If you change
the vLLM version, keep the runtime version aligned with the bundled nvcc.

From Python, with the serving directory on your import path:

```python
from vllm_server import VLLMServer

server = VLLMServer(
    "/home/jingyuan_li/MoE-Distributed-Inference/llm-serving/config.example.yaml"
)
print(server.parameters["model"])
server.serve()
```

Run such scripts with `uv run python your_script.py` so they use the project's
environment.

Construction reads YAML safely and requires a non-empty `model` string.
`serve()` runs `vllm serve --config ...` in the foreground, loading the model
and exposing vLLM's OpenAI-compatible API. It blocks until the server exits;
use Ctrl+C in the terminal to stop it. Server logs are inherited and nonzero
exit codes raise `subprocess.CalledProcessError`. Launching does not provide a
readiness guarantee: wait for startup logs or check the `/health` endpoint.

Use vLLM's long-form, hyphenated option names in YAML. All options are passed
through the native configuration interface; vLLM validates their values and
compatibility. Do not change the file between construction and `serve()`:
vLLM rereads it at launch. Mutating `server.parameters` does not change the
server configuration. Relative model paths resolve from the launch directory.

The example binds only to localhost on port 8000. Set `host: "0.0.0.0"` for
remote access only with appropriate authentication and network protections.
Model download access and sufficient accelerator memory are also required.

Run CPU-only tests:

```bash
cd /home/jingyuan_li/MoE-Distributed-Inference/llm-serving
uv run python -m unittest discover -s . -p 'test_*.py' -v
```