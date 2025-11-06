FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Minimal OS dependencies for ComfyUI runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fetch ComfyUI sources
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git

# Install Python dependencies (PyTorch already provided by base image)
WORKDIR /app/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# Default server flags (overridable via CLI_ARGS env)
ENV CLI_ARGS="--listen 0.0.0.0 --port 8188"
EXPOSE 8188

CMD bash -lc "python3 main.py ${CLI_ARGS}"
