# 第一阶段：构建环境
FROM python:3.10-slim-bookworm as builder

WORKDIR /app
ENV PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python3-dev libgomp1 ffmpeg libsndfile1 libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 创建虚拟环境
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install -r requirements.txt

# 第二阶段：生产镜像
FROM python:3.10-slim-bookworm

# 创建非特权用户
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    mkdir /app && \
    chown appuser:appuser /app

WORKDIR /app
ENV PYTHONPATH=/app \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# 仅复制必要文件
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser . .

# 切换用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:15433/health || exit 1

# 启动命令
CMD ["chainlit", "run", "main.py", "--port", "15433", "--timeout", "300"]
