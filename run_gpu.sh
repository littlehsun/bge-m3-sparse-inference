#!/bin/bash
# Run BGE-M3 Sparse Service - GPU Mode

export MODEL_ID="${MODEL_ID:-BAAI/bge-m3}"
export DEVICE="cuda"
export DTYPE="float16"
export MAX_BATCH_SIZE=128
export PORT="${PORT:-8080}"

echo "============================================"
echo "BGE-M3 Sparse Embedding Service (GPU Mode)"
echo "============================================"
echo "Model: $MODEL_ID"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Max Batch: $MAX_BATCH_SIZE"
echo "Port: $PORT"
echo "============================================"

python main.py
