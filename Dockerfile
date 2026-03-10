FROM nvcr.io/nvidia/pytorch:25.10-py3

WORKDIR /app

# Build args for proxy
ARG http_proxy
ARG https_proxy

RUN apt-get update && apt-get install -y --no-install-recommends curl git vim && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights during build
RUN pip uninstall -y apex
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cpu')" && \
    echo "Model downloaded successfully"

COPY main.py model.py ./
COPY vram_stress ./vram_stress

ENV MODEL_ID=BAAI/bge-m3
ENV DEVICE=cuda
ENV DTYPE=float16
ENV MAX_BATCH_SIZE=128
ENV MICRO_BATCH_SIZE=128
ENV PORT=80
ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_OFFLINE=1

EXPOSE 80

CMD ["python", "main.py"]
