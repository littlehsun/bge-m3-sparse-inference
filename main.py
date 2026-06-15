"""
BGE-M3 Sparse Embedding Service - GPU/CPU Optimized
Handles 128 batch size with maximum speed optimization.

API compatible with text-embeddings-inference /embed_sparse
"""

import os
import asyncio
import logging
import threading
from typing import List, Optional, Union
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import BGEM3SparseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[BGEM3SparseModel] = None

# GPU inference lock - ensure only ONE inference runs at a time
# This prevents GPU resource contention and memory allocation conflicts
inference_lock = threading.Lock()


def _runtime_device() -> str:
    requested = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return str(device)


def _validate_worker_config(device: str, workers: int):
    if torch.device(device).type == "cuda" and workers != 1:
        raise RuntimeError(
            "GPU inference requires WORKERS=1. Multiple uvicorn workers each load "
            "a full model copy and inference_lock does not synchronize across processes."
        )


class EmbedSparseRequest(BaseModel):
    """Request format compatible with TEI"""
    inputs: Union[str, List[str]]
    truncate: bool = True
    truncation_direction: str = "Right"


class EmbedRequest(BaseModel):
    """Request format for dense embeddings"""
    inputs: Union[str, List[str]]
    truncate: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup"""
    global model

    model_id = os.environ.get("MODEL_ID", "BAAI/bge-m3")
    device = _runtime_device()
    device_type = torch.device(device).type
    dtype = os.environ.get("DTYPE", "float16" if device_type == "cuda" else "float32")
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "128"))
    workers = int(os.environ.get("WORKERS", "1"))
    _validate_worker_config(device, workers)

    # ==========================================================================
    # GPU Memory Utilization (similar to vLLM's gpu_memory_utilization)
    # This MUST be set before any CUDA memory allocation (i.e., before model load)
    #
    # When set, PyTorch will limit GPU memory usage to this fraction of total.
    # This allows sharing GPU with other services without OOM.
    #
    # Example: GPU_MEMORY_UTILIZATION=0.5 on 80GB GPU = limit to 40GB
    # ==========================================================================
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION")
    if device_type == "cuda" and torch.cuda.is_available() and gpu_memory_utilization:
        fraction = float(gpu_memory_utilization)
        if 0.0 < fraction <= 1.0:
            cuda_device = torch.device(device)
            torch.cuda.set_per_process_memory_fraction(fraction, device=cuda_device)
            total_memory = torch.cuda.get_device_properties(cuda_device).total_memory / 1024**3
            limited_memory = total_memory * fraction
            logger.info(f"[GPU Memory] Limiting to {fraction:.0%} of GPU memory")
            logger.info(f"[GPU Memory] Total: {total_memory:.1f}GB, Usable: {limited_memory:.1f}GB")
        else:
            logger.warning(f"[GPU Memory] Invalid GPU_MEMORY_UTILIZATION={fraction}, must be between 0.0 and 1.0")

    logger.info(f"Loading BGE-M3 model: {model_id}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Max batch: {max_batch_size}")
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    model = BGEM3SparseModel(
        model_id=model_id,
        device=device,
        dtype=dtype_map.get(dtype, torch.float16),
        max_batch_size=max_batch_size,
    )
    
    logger.info("Model loaded and ready!")
    yield
    
    # Cleanup
    loaded_model = model
    model = None
    del loaded_model
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="BGE-M3 Sparse Embedding Service",
    version="1.0.0",
    lifespan=lifespan,
)


def run_embed_sparse_sync(inputs: List[str], truncate: bool):
    """Run sparse embedding with lock to prevent GPU contention"""
    with inference_lock:
        try:
            return model.embed_sparse(inputs, truncate)
        finally:
            # Clear GPU cache after both successful requests and exceptions.
            current_model = model
            if current_model is not None and current_model.device.type == "cuda":
                torch.cuda.empty_cache()


def run_embed_dense_sync(inputs: List[str]):
    """Run dense embedding with lock to prevent GPU contention"""
    with inference_lock:
        try:
            return model.embed_dense(inputs)
        finally:
            # Clear GPU cache after both successful requests and exceptions.
            current_model = model
            if current_model is not None and current_model.device.type == "cuda":
                torch.cuda.empty_cache()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/embed_sparse")
async def embed_sparse(request: EmbedSparseRequest):
    """
    Generate sparse embeddings - TEI compatible API.
    
    Returns list of sparse embeddings, each containing (index, value) pairs.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Normalize input to list
    inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
    
    if len(inputs) == 0:
        return []
    
    if len(inputs) > model.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(inputs)} exceeds max {model.max_batch_size}"
        )
    
    # Run inference in thread pool with lock to prevent GPU contention
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, 
        run_embed_sparse_sync, 
        inputs,
        request.truncate,
    )
    
    return result


@app.post("/embed")
async def embed(request: EmbedRequest):
    """
    Generate dense embeddings.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    inputs = request.inputs if isinstance(request.inputs, list) else [request.inputs]
    
    if len(inputs) == 0:
        return []
    
    if len(inputs) > model.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(inputs)} exceeds max {model.max_batch_size}"
        )
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_embed_dense_sync, inputs)
    
    return result


@app.get("/info")
async def info():
    """Model info endpoint"""
    return {
        "model_id": model.model_id if model else None,
        "device": str(model.device) if model else None,
        "dtype": str(model.dtype) if model else None,
        "max_batch_size": model.max_batch_size if model else None,
        "vocab_size": model.vocab_size if model else None,
    }


@app.get("/gpu_memory")
async def gpu_memory():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "allocated_gb": 0,
            "reserved_gb": 0,
            "max_allocated_gb": 0,
        }

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Check if memory limit is set
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION")
    if gpu_memory_utilization:
        fraction = float(gpu_memory_utilization)
        limited_memory = total_memory * fraction
    else:
        fraction = 1.0
        limited_memory = total_memory

    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(total_memory, 3),
        "gpu_memory_utilization": fraction,
        "usable_memory_gb": round(limited_memory, 3),
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "max_allocated_gb": round(max_allocated, 3),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("WORKERS", "1"))
    runtime_device = _runtime_device()
    try:
        _validate_worker_config(runtime_device, workers)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info",
    )
