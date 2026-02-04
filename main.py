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
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = os.environ.get("DTYPE", "float16" if device == "cuda" else "float32")
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "128"))
    
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
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(
    title="BGE-M3 Sparse Embedding Service",
    version="1.0.0",
    lifespan=lifespan,
)


def run_embed_sparse_sync(inputs: List[str], truncate: bool):
    """Run sparse embedding with lock to prevent GPU contention"""
    with inference_lock:
        result = model.embed_sparse(inputs, truncate)
        # Clear GPU cache after each request to prevent memory accumulation
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def run_embed_dense_sync(inputs: List[str]):
    """Run dense embedding with lock to prevent GPU contention"""
    with inference_lock:
        result = model.embed_dense(inputs)
        # Clear GPU cache after each request to prevent memory accumulation
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("WORKERS", "1"))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info",
    )
