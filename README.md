# BGE-M3 Sparse Embedding Service

High-performance REST API service for generating sparse and dense embeddings using the [BGE-M3](https://huggingface.co/BAAI/bge-m3) model.

## Features

- GPU (CUDA) and CPU support with automatic optimization
- FP16 inference for GPU, FP32 for CPU
- Sparse and dense embedding endpoints
- TEI (text-embeddings-inference) compatible API format
- Batch processing with configurable batch size (up to 128)
- Docker support for easy deployment
- Optimized post-processing with vectorized operations

## Quick Start

### Using Docker (Recommended)

**GPU Version:**
```bash
# Build
docker build -t bge-m3-sparse:gpu .

# Run
docker run --gpus all -p 8080:8080 bge-m3-sparse:gpu
```

**CPU Version:**
```bash
# Build
docker build -f Dockerfile-cpu -t bge-m3-sparse:cpu .

# Run
docker run -p 8080:8080 bge-m3-sparse:cpu
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run (GPU)
export DEVICE=cuda
export DTYPE=float16
python main.py

# Run (CPU)
export DEVICE=cpu
export DTYPE=float32
python main.py
```

## Configuration

### Environment Variables

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MODEL_ID` | HuggingFace model identifier | `BAAI/bge-m3` | Any compatible model |
| `DEVICE` | Compute device | `cuda` if available, else `cpu` | `cuda`, `cpu` |
| `DTYPE` | Data type for inference | `float16` (GPU), `float32` (CPU) | `float16`, `bfloat16`, `float32` |
| `MAX_BATCH_SIZE` | Maximum allowed batch size per request | `128` | 1-128 |
| `MICRO_BATCH_SIZE` | Internal batch size for memory management | `64` (GPU), `8` (CPU) | Adjust based on GPU memory |
| `MAX_LENGTH` | Maximum token length | `8192` | 1-8192 |
| `PORT` | API server port | `8080` | Any valid port |
| `WORKERS` | Uvicorn worker processes | `1` | 1+ (recommend 1 for GPU) |

### CPU-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OMP_NUM_THREADS` | OpenMP thread count | `16` |
| `MKL_NUM_THREADS` | MKL thread count | `16` |

### Example Configurations

**High-memory GPU (A100 80GB):**
```bash
export DEVICE=cuda
export DTYPE=float16
export MICRO_BATCH_SIZE=128
export MAX_LENGTH=8192
```

**Low-memory GPU (RTX 3060 12GB):**
```bash
export DEVICE=cuda
export DTYPE=float16
export MICRO_BATCH_SIZE=16
export MAX_LENGTH=2048
```

**CPU with many cores:**
```bash
export DEVICE=cpu
export DTYPE=float32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export MICRO_BATCH_SIZE=8
```

## API Reference

### Health Check

```http
GET /health
```

**Response:**
```json
{"status": "ok"}
```

### Model Info

```http
GET /info
```

**Response:**
```json
{
  "model_id": "BAAI/bge-m3",
  "device": "cuda:0",
  "dtype": "torch.float16",
  "max_batch_size": 128,
  "vocab_size": 250002
}
```

### Sparse Embedding

```http
POST /embed_sparse
Content-Type: application/json
```

**Request Body:**
```json
{
  "inputs": "Your text here",
  "truncate": true
}
```

Or batch request:
```json
{
  "inputs": ["Text 1", "Text 2", "Text 3"],
  "truncate": true
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `inputs` | `string` or `string[]` | Yes | - | Input text(s) to embed |
| `truncate` | `boolean` | No | `true` | Truncate texts exceeding max length |

**Response:**
```json
[
  [
    {"index": 12, "value": 0.752341},
    {"index": 456, "value": 0.234567},
    ...
  ]
]
```

Each item contains sparse token indices and their weights.

### Dense Embedding

```http
POST /embed
Content-Type: application/json
```

**Request Body:**
```json
{
  "inputs": ["Text 1", "Text 2"],
  "truncate": true
}
```

**Response:**
```json
[
  [0.123, 0.456, 0.789, ...],
  [0.234, 0.567, 0.890, ...]
]
```

Returns 1024-dimensional normalized dense vectors.

## Usage Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8080"

# Single text - sparse embedding
response = requests.post(
    f"{BASE_URL}/embed_sparse",
    json={"inputs": "What is machine learning?"}
)
sparse_embedding = response.json()[0]
print(f"Non-zero tokens: {len(sparse_embedding)}")

# Batch - dense embedding
texts = [
    "First document about AI",
    "Second document about ML",
    "Third document about NLP"
]
response = requests.post(
    f"{BASE_URL}/embed",
    json={"inputs": texts}
)
dense_embeddings = response.json()
print(f"Embedding dimension: {len(dense_embeddings[0])}")
```

### cURL

```bash
# Health check
curl http://localhost:8080/health

# Single sparse embedding
curl -X POST http://localhost:8080/embed_sparse \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello world"}'

# Batch dense embedding
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Text 1", "Text 2", "Text 3"]}'
```

## Testing

```bash
# Basic tests
python test.py --url http://localhost:8080

# Full benchmark
python test.py --url http://localhost:8080 --benchmark
```

## Performance Tuning

### Timing Logs

The service outputs detailed timing information:
```
[TIMING] batch=100: tokenize=50ms, forward=2000ms, postprocess=5ms, build=10ms
```

| Stage | Description |
|-------|-------------|
| `tokenize` | Text to token conversion |
| `forward` | Model inference (GPU-accelerated) |
| `postprocess` | Sparse vector construction on GPU |
| `build` | Convert to JSON-serializable format |

### Optimization Tips

1. **Increase `MICRO_BATCH_SIZE`** for better GPU utilization (if memory allows)
2. **Decrease `MAX_LENGTH`** if your texts are short
3. **Use FP16** on GPU for 2x memory efficiency
4. **Batch your requests** - processing 100 texts at once is faster than 100 single requests

### Expected Performance

| Hardware | Batch=100 (short texts) | Notes |
|----------|-------------------------|-------|
| A100 80GB | ~1.5-2s | Full GPU |
| A100 MIG 1g.10gb | ~11s | 1/7 compute slice |
| RTX 4090 | ~2-3s | Consumer GPU |
| CPU (16 cores) | ~20-25s | Varies by CPU |

## Project Structure

```
bge-m3-sparse-inference/
├── main.py           # FastAPI application
├── model.py          # BGEM3SparseModel class
├── test.py           # Test client
├── requirements.txt  # Python dependencies
├── Dockerfile        # GPU Docker image
├── Dockerfile-cpu    # CPU Docker image
├── build.sh          # Docker build script
├── run_gpu.sh        # Local GPU run script
├── run_cpu.sh        # Local CPU run script
└── local_run.sh      # Docker run script
```

## Troubleshooting

### Model not using GPU

Check startup logs for:
```
[GPU] Model device AFTER .to(): cuda:0
[GPU] Model dtype: torch.float16
```

If it shows `cpu`, verify CUDA is available:
```python
import torch
print(torch.cuda.is_available())
```

### Out of Memory

Reduce `MICRO_BATCH_SIZE` or `MAX_LENGTH`:
```bash
export MICRO_BATCH_SIZE=16
export MAX_LENGTH=2048
```

### Slow Performance

1. Check GPU utilization: `nvidia-smi -l 1`
2. Verify FP16 is enabled (not FP32)
3. Ensure `MICRO_BATCH_SIZE` matches your batch size
4. Check input text lengths - long texts are slower (O(n²) attention)

## License

MIT License

## Acknowledgements

- [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) - BGE-M3 model implementation
- [BAAI](https://huggingface.co/BAAI) - Model weights
