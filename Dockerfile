FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Build args for proxy
ARG http_proxy
ARG https_proxy

RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py model.py ./

# Pre-download model weights during build
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cpu')" && \
    echo "Model downloaded successfully"

ENV MODEL_ID=BAAI/bge-m3
ENV DEVICE=cuda
ENV DTYPE=float16
ENV MAX_BATCH_SIZE=128
ENV PORT=8080
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8080

CMD ["python", "main.py"]
