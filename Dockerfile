FROM ollama/ollama:latest

# 必要なソフトのインストール
RUN apt-get update && apt-get install -y \
    git \
    grep \
    curl \
    xz-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uvのインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Default command
CMD ["/bin/bash"]
