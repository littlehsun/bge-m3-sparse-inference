# Copilot Instructions for BGE-M3 Sparse Inference

## Project Overview

This is a high-performance REST API service for generating sparse and dense embeddings using the BGE-M3 model. The service is optimized for both GPU (CUDA) and CPU inference with support for batch processing.

### Key Technologies
- **Framework**: FastAPI with Uvicorn
- **ML Framework**: PyTorch with HuggingFace Transformers
- **Model**: BAAI/bge-m3 (FlagEmbedding)
- **Deployment**: Docker support for GPU and CPU

## Architecture

### Core Components
1. **main.py**: FastAPI application with endpoints for sparse/dense embeddings
2. **model.py**: BGEM3SparseModel class with optimized inference
3. **test.py**: Test client for API validation
4. **test_lexical_weights.py**: Tests for lexical weight computations

### API Endpoints
- `GET /health`: Health check
- `GET /info`: Model information
- `POST /embed_sparse`: Generate sparse embeddings (TEI-compatible)
- `POST /embed`: Generate dense embeddings
- `GET /gpu_memory`: GPU memory usage stats

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for function parameters and return values
- Include docstrings for classes and public methods
- Keep functions focused and single-purpose
- Use descriptive variable names (e.g., `micro_batch_size`, `max_length`)

### Performance Considerations
- **GPU Optimization**: This service is heavily optimized for GPU performance
  - Use FP16 for GPU inference to save memory (2x reduction)
  - Minimize CPU-GPU transfers
  - Use vectorized operations (no Python loops on large tensors)
  - Pre-allocate tensors when possible
- **Thread Safety**: Use `inference_lock` to prevent GPU resource contention
- **Memory Management**: Clear GPU cache after inference to prevent accumulation
- **Batch Processing**: Support micro-batching for memory efficiency

### Testing
```bash
# Run basic tests
python test.py --url http://localhost:8080

# Run with benchmarks
python test.py --url http://localhost:8080 --benchmark

# Test lexical weights
python test_lexical_weights.py
```

### Running Locally
```bash
# GPU version
export DEVICE=cuda
export DTYPE=float16
python main.py

# CPU version
export DEVICE=cpu
export DTYPE=float32
python main.py
```

### Docker Usage
```bash
# Build GPU version
docker build -t bge-m3-sparse:gpu .

# Build CPU version
docker build -f Dockerfile-cpu -t bge-m3-sparse:cpu .

# Run with GPU
docker run --gpus all -p 8080:8080 bge-m3-sparse:gpu
```

## Configuration

### Environment Variables
- `MODEL_ID`: HuggingFace model identifier (default: `BAAI/bge-m3`)
- `DEVICE`: Compute device (`cuda` or `cpu`)
- `DTYPE`: Data type for inference (`float16`, `bfloat16`, `float32`)
- `MAX_BATCH_SIZE`: Maximum batch size per request (default: 128)
- `MICRO_BATCH_SIZE`: Internal batch size for memory management (GPU: 64, CPU: 8)
- `MAX_LENGTH`: Maximum token length (default: 8192)
- `PORT`: API server port (default: 8080)
- `WORKERS`: Uvicorn worker processes (default: 1, recommend 1 for GPU)

### CPU-Specific Variables
- `OMP_NUM_THREADS`: OpenMP thread count (default: 16)
- `MKL_NUM_THREADS`: MKL thread count (default: 16)

## Code Patterns

### Request Handling
- Use Pydantic models for request validation
- Support both single string and list of strings as input
- Validate batch size against `max_batch_size`
- Use async/await with thread pool executor for blocking ML operations
- Always use the `inference_lock` to prevent GPU contention

### Error Handling
- Return `HTTPException` with appropriate status codes:
  - 400: Invalid request (e.g., batch too large)
  - 503: Service unavailable (model not loaded)
- Log errors with appropriate log levels

### Model Operations
- Use `@torch.inference_mode()` for inference (disables gradient tracking)
- Implement micro-batching for memory-constrained scenarios
- Clear GPU cache after operations: `torch.cuda.empty_cache()`
- Use vectorized operations: `scatter_reduce`, `searchsorted`, etc.

## Optimization Best Practices

### GPU Memory Management
1. Use FP16 on GPU for 2x memory savings
2. Adjust `MICRO_BATCH_SIZE` based on available memory
3. Clear cache regularly to prevent accumulation
4. Monitor memory with `/gpu_memory` endpoint

### Inference Speed
1. Maximize `MICRO_BATCH_SIZE` without OOM
2. Use batch processing - 100 texts at once is much faster than 100 single requests
3. Decrease `MAX_LENGTH` if texts are short
4. Pre-allocate tensors and reuse when possible

### Code Optimization Principles
1. **Vectorization**: Replace Python loops with NumPy/PyTorch operations
2. **Minimal Transfers**: Only transfer non-zero values from GPU to CPU
3. **Caching**: Cache masks and static computations
4. **Type Consistency**: Keep tensors on same device until final transfer

## Dependencies

### Core Dependencies
- `torch>=2.0.0`: PyTorch for ML operations
- `transformers>=4.40.0,<4.46.0`: HuggingFace models
- `fastapi>=0.109.0`: Web framework
- `uvicorn[standard]>=0.27.0`: ASGI server
- `FlagEmbedding>=1.2.0,<1.3.0`: BGE model implementation

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Common Tasks

### Adding New Endpoints
1. Define Pydantic request model
2. Use `@app.post()` or `@app.get()` decorator
3. Add async function with proper error handling
4. Use `inference_lock` for model operations
5. Return JSON-serializable results
6. Add tests in `test.py`

### Modifying Model Behavior
1. Changes should be in `model.py`
2. Maintain backward compatibility with API
3. Document timing impacts in docstring
4. Test with both GPU and CPU
5. Verify memory usage hasn't increased

### Adding Configuration Options
1. Add environment variable with `os.environ.get()`
2. Provide sensible defaults
3. Document in README.md
4. Validate values at startup
5. Log configuration values

## Troubleshooting

### Model Not Using GPU
- Check startup logs for device confirmation
- Verify CUDA availability: `torch.cuda.is_available()`
- Check `DEVICE` environment variable

### Out of Memory Errors
- Reduce `MICRO_BATCH_SIZE`
- Reduce `MAX_LENGTH`
- Use FP16 instead of FP32
- Check for memory leaks with `/gpu_memory`

### Slow Performance
- Increase `MICRO_BATCH_SIZE` if memory allows
- Verify FP16 is enabled on GPU
- Use batch requests instead of single requests
- Check GPU utilization with `nvidia-smi`

## Security Considerations

- Input validation: All inputs are validated through Pydantic models
- Resource limits: `MAX_BATCH_SIZE` prevents excessive resource usage
- No user code execution: Only model inference, no eval/exec
- Docker isolation: Recommended deployment method

## Additional Notes

- The service uses a single inference lock to prevent GPU resource contention
- Worker count should be 1 for GPU to avoid device conflicts
- CPU mode can use multiple workers for parallelism
- The API format is compatible with text-embeddings-inference (TEI)
- Timing logs show performance breakdown: tokenize, forward, postprocess, build
