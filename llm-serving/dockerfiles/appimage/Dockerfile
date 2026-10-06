# Build the dependency image first, or supply a registry image reference.
ARG BASE_IMAGE=moe-vllm-base:local
FROM ${BASE_IMAGE}

WORKDIR /app

# Build context must be the llm-serving directory.
COPY vllm_server.py /app/vllm_server.py
COPY dockerfiles/appimage/config.yaml /app/config.yaml

EXPOSE 8000
STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/bin/tini", "-g", "--", "python3", "/app/vllm_server.py"]
CMD ["--config", "/app/config.yaml"]