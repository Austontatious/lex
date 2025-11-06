# codex-managed: update via deployment automation
# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

# 0) Ensure /tmp exists and is writable/sticky; guard against /tmp being a file
RUN set -eux; \
    if [ ! -d /tmp ]; then rm -f /tmp; mkdir -p /tmp; fi; \
    chmod 1777 /tmp; \
    ls -ld /tmp

# 1) Drop CUDA/NVIDIA sources before first apt-get update (we can re-add later with signed-by)
RUN set -eux; rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true

# 2) Base tools & APT prep
ENV DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        apt-transport-https; \
    rm -rf /var/lib/apt/lists/*

# 3) (Optional) Re-add CUDA repo with signed-by keyring
# RUN set -eux; \
#     install -d -m 0755 /etc/apt/keyrings; \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
#         | gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda-archive-keyring.gpg; \
#     chmod a+r /etc/apt/keyrings/nvidia-cuda-archive-keyring.gpg; \
#     echo "deb [signed-by=/etc/apt/keyrings/nvidia-cuda-archive-keyring.gpg arch=amd64] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" \
#         > /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN set -eux; \
    apt-get update -y; \
    apt-get install -y --no-install-recommends \
        apt-transport-https \
        gnupg \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
        python3-pip \
        build-essential \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libmagic1 \
        file \
    ; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_ONLY_BINARY=:all:

COPY requirements.txt constraints.alpha.txt ./

RUN python3 -m pip install --upgrade pip==25.2 setuptools wheel \
    && python3 -m pip install --index-url https://pypi.org/simple --no-deps \
        https://files.pythonhosted.org/packages/source/s/sgmllib3k/sgmllib3k-1.0.0.tar.gz \
        https://files.pythonhosted.org/packages/source/P/PyPika/PyPika-0.48.9.tar.gz \
    && python3 -m pip install \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --constraint constraints.alpha.txt \
        --prefer-binary \
        -r requirements.txt

COPY backend /app/backend
COPY frontend/public /app/static
RUN mkdir -p /app/backend/lexi/static && cp -a /app/static/. /app/backend/lexi/static/

ENV PYTHONPATH=/app/backend

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "lexi.core.backend_core:app", "--host", "0.0.0.0", "--port", "9000", "--proxy-headers"]
