# Override with a release tag or digest for reproducible deployments.
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

USER root

# The official image supplies vLLM, PyTorch, CUDA libraries and PyYAML.
# Tini forwards shutdown signals to the Python wrapper AND its vLLM child.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c "import yaml; from importlib.metadata import version; print('vLLM:', version('vllm'))" \
    && command -v vllm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Remove the upstream API-server entrypoint; the app image supplies its own.
ENTRYPOINT []
CMD ["python3", "--version"]