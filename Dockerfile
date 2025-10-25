FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ==== perbaikan pip: timeout & retries lebih tinggi ====
ENV PIP_DEFAULT_TIMEOUT=120 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .

# ==== install bertahap biar lebih andal ====
# 1) upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 2) install PyTorch CPU dari index resmi (lebih stabil & lebih kecil dari build CUDA)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 3) baru install deps lain (ultralytics, fastapi, opencv headless, dll)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=models/best.pt
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1
EXPOSE 8000
CMD ["sh","-c","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --limit-concurrency 4 --timeout-keep-alive 5"]
