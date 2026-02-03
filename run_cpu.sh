#!/bin/bash
# Run BGE-M3 Sparse Service - CPU Mode

export MODEL_ID="${MODEL_ID:-BAAI/bge-m3}"
export DEVICE="cpu"
export DTYPE="float32"
export MAX_BATCH_SIZE=128
export PORT="${PORT:-8080}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

echo "============================================"
echo "BGE-M3 Sparse Embedding Service (CPU Mode)"
echo "============================================"
echo "Model: $MODEL_ID"
echo "Device: $DEVICE"
echo "Dtype: $DTYPE"
echo "Max Batch: $MAX_BATCH_SIZE"
echo "Port: $PORT"
echo "Threads: $OMP_NUM_THREADS"
echo "============================================"

python main.py
