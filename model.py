"""
BGE-M3 Sparse Model - Maximum Speed Optimization

Key optimizations:
1. Uses FlagEmbedding BGEM3FlagModel for native sparse support
2. Vectorized sparse result building (no Python loops on vocab)
3. Pre-allocated tensors for sparse processing
4. No torch.unique() - uses scatter_reduce instead
5. Minimal CPU-GPU transfers (only transfer non-zero values, not full tensor)
6. Float16 inference on GPU
7. Cached special token mask (created once, not per-batch)
8. Vectorized np.round() instead of per-element Python round()
9. np.searchsorted for O(batch_size) grouping instead of O(n) loop
10. Higher default MICRO_BATCH_SIZE for GPU (64 vs 32)
"""

import os
import logging
from typing import List, Dict, Any, Set

import time
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Disable tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"


def log_gpu_info():
    """Log GPU information for debugging"""
    if torch.cuda.is_available():
        logger.info(f"[GPU] CUDA available: True")
        logger.info(f"[GPU] CUDA version: {torch.version.cuda}")
        logger.info(f"[GPU] Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"[GPU] Device {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")
        logger.info(f"[GPU] Current device: {torch.cuda.current_device()}")
    else:
        logger.warning("[GPU] CUDA NOT available!")


def log_gpu_memory():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"[GPU Memory] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


class BGEM3SparseModel:
    """
    Ultra-optimized BGE-M3 sparse embedding model using FlagEmbedding.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_batch_size: int = 128,
        max_length: int = int(os.environ.get("MAX_LENGTH", "8192")),
        compile_model: bool = False,  # Disabled - incompatible with FlagEmbedding
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        
        # Micro-batch size (optimization #3: increase default for GPU)
        # Higher micro_batch_size = fewer batches = less overhead
        default_micro_batch = "8" if device == "cpu" else "64"
        self.micro_batch_size = int(os.environ.get("MICRO_BATCH_SIZE", default_micro_batch))
        logger.info(f"[CONFIG] MICRO_BATCH_SIZE={self.micro_batch_size} (env={os.environ.get('MICRO_BATCH_SIZE', 'not set')})")

        # Log GPU info at startup
        log_gpu_info()

        # Import FlagEmbedding
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required. Install with: pip install FlagEmbedding"
            )

        # Configure CUDA for stability
        if self.device.type == "cuda":
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "max_split_size_mb:512,expandable_segments:True",
            )
            torch.cuda.empty_cache()

        use_fp16 = dtype in [torch.float16, torch.bfloat16]

        logger.info(f"Loading BGEM3FlagModel from {model_id}...")
        logger.info(f"[GPU] Requesting device: {self.device}, use_fp16: {use_fp16}")
        
        self.flag_model = BGEM3FlagModel(
            model_id,
            use_fp16=use_fp16,
            device=str(self.device),
        )

        # Get internal model for direct forward pass
        self._model = self.flag_model.model
        
        # Verify model device BEFORE .to()
        logger.info(f"[GPU] Model device BEFORE .to(): {next(self._model.parameters()).device}")
        
        self._model = self._model.to(self.device)
        self._model.eval()
        
        # Verify model device AFTER .to()
        logger.info(f"[GPU] Model device AFTER .to(): {next(self._model.parameters()).device}")
        logger.info(f"[GPU] Model dtype: {next(self._model.parameters()).dtype}")
        
        # Log GPU memory after model load
        log_gpu_memory()

        # Get tokenizer
        self.tokenizer = self.flag_model.tokenizer
        self.vocab_size = len(self.tokenizer)

        # Cache special token IDs
        self._special_token_ids: Set[int] = set()
        for attr in [
            "cls_token_id",
            "sep_token_id",
            "pad_token_id",
            "unk_token_id",
            "mask_token_id",
        ]:
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                self._special_token_ids.add(tid)

        # Pre-create special tokens mask on device (CRITICAL optimization!)
        self._special_mask = torch.zeros(
            self.vocab_size, dtype=torch.bool, device=self.device
        )
        for tid in self._special_token_ids:
            if 0 <= tid < self.vocab_size:
                self._special_mask[tid] = True
        
        logger.info(f"[GPU] Special mask device: {self._special_mask.device}")

        # Warmup
        self._warmup()

        logger.info(
            f"Model ready: device={self.device}, dtype={self.dtype}, vocab={self.vocab_size}, micro_batch={self.micro_batch_size}, max_len={self.max_length}"
        )

    def _warmup(self):
        """Warmup the model with dummy batches"""
        logger.info("Warming up model...")
        dummy_texts = ["warmup text"] * min(4, self.max_batch_size)

        with torch.inference_mode():
            _ = self._embed_sparse_batch(dummy_texts)
            _ = self._embed_dense_batch(dummy_texts)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

        log_gpu_memory()
        logger.info("Warmup complete")

    def _tokenize(
        self, texts: List[str], truncate: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts with padding"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=truncate,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "token_type_ids": torch.zeros_like(
                encoded["input_ids"], device=self.device
            ),
        }

    def _model_forward(self, inputs: Dict[str, torch.Tensor], return_dense: bool, return_sparse: bool) -> Dict[str, Any]:
        """Forward pass with compatibility for different FlagEmbedding versions."""
        try:
            outputs = self._model(
                inputs,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
                return_sparse_embedding=False,
            )
        except TypeError:
            outputs = self._model(
                inputs,
                return_dense=return_dense,
                return_sparse=return_sparse,
            )
        return outputs

    def _embed_sparse_batch(
        self,
        texts: List[str],
        truncate: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Process a single batch of texts for sparse embeddings - OPTIMIZED"""
        batch_size = len(texts)
        
        # === TIMING: Tokenize ===
        t0 = time.perf_counter()
        inputs = self._tokenize(texts, truncate)
        input_ids = inputs["input_ids"]
        t_tokenize = (time.perf_counter() - t0) * 1000

        # === TIMING: Model forward ===
        t0 = time.perf_counter()
        outputs = self._model_forward(inputs, return_dense=False, return_sparse=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_forward = (time.perf_counter() - t0) * 1000

        # === TIMING: Post-processing (VECTORIZED) ===
        t0 = time.perf_counter()
        
        token_weights = outputs["sparse_vecs"]
        if token_weights.dim() == 3:
            token_weights = token_weights.squeeze(-1)

        token_weights = torch.relu(token_weights)

        # Create dense (batch, vocab) tensor for max weights per token
        sparse_dense = torch.zeros(
            (batch_size, self.vocab_size),
            device=self.device,
            dtype=token_weights.dtype,
        )

        # Mask special tokens
        is_special = self._special_mask[input_ids]
        token_weights_masked = token_weights.masked_fill(is_special, 0.0)

        # Scatter max
        sparse_dense.scatter_reduce_(
            dim=1,
            index=input_ids.long(),
            src=token_weights_masked,
            reduce="amax",
            include_self=True,
        )

        # Apply threshold on GPU before transfer
        sparse_dense = sparse_dense * (sparse_dense > 0.001).float()
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_postprocess = (time.perf_counter() - t0) * 1000

        # === TIMING: Build results (FULLY VECTORIZED) ===
        t0 = time.perf_counter()

        # Find ALL non-zero positions at once using torch.nonzero
        nonzero_coords = torch.nonzero(sparse_dense, as_tuple=False)

        # Pre-allocate results list
        results: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]

        if len(nonzero_coords) > 0:
            # Optimization #4: Extract weights on GPU BEFORE transfer
            # This avoids transferring the entire (batch_size x vocab_size) tensor
            batch_indices_gpu = nonzero_coords[:, 0]
            token_indices_gpu = nonzero_coords[:, 1]
            weights_gpu = sparse_dense[batch_indices_gpu, token_indices_gpu]

            # Single CPU transfer for each array (optimization #4)
            batch_indices = batch_indices_gpu.cpu().numpy()
            token_indices = token_indices_gpu.cpu().numpy()
            weights = weights_gpu.cpu().numpy()

            # Optimization #2: Vectorized round using numpy (much faster than per-element)
            weights = np.round(weights, 6)

            # Optimization #1: Use searchsorted for batch grouping
            # torch.nonzero returns in row-major order, so batch_indices are sorted
            # This replaces the O(n) Python loop with O(batch_size) iterations
            boundaries = np.searchsorted(batch_indices, np.arange(batch_size + 1))

            for batch_idx in range(batch_size):
                start = boundaries[batch_idx]
                end = boundaries[batch_idx + 1]
                if start < end:
                    # List comprehension with zip is faster than append loop
                    results[batch_idx] = [
                        {"index": int(t), "value": float(w)}
                        for t, w in zip(token_indices[start:end], weights[start:end])
                    ]

        t_build = (time.perf_counter() - t0) * 1000

        logger.info(f"[TIMING] batch={batch_size}: tokenize={t_tokenize:.1f}ms, forward={t_forward:.1f}ms, postprocess={t_postprocess:.1f}ms, build={t_build:.1f}ms")

        # === MEMORY CLEANUP: Explicitly delete GPU tensors ===
        del inputs, input_ids, outputs, token_weights, sparse_dense
        del is_special, token_weights_masked, nonzero_coords
        # batch_indices_gpu etc. only exist when nonzero_coords was non-empty
        try:
            del batch_indices_gpu, token_indices_gpu, weights_gpu
        except NameError:
            pass

        return results

    @torch.inference_mode()
    def embed_sparse(
        self,
        texts: List[str],
        truncate: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Generate sparse embeddings with micro-batching."""
        start_time = time.perf_counter()
        total_size = len(texts)
        logger.info(f"[embed_sparse] Input batch size: {total_size}, device: {self.device}")

        if total_size <= self.micro_batch_size:
            results = self._embed_sparse_batch(texts, truncate)
        else:
            results = []
            for i in range(0, total_size, self.micro_batch_size):
                batch_texts = texts[i:i + self.micro_batch_size]
                batch_results = self._embed_sparse_batch(batch_texts, truncate)
                results.extend(batch_results)

                # Clear GPU cache periodically to prevent memory accumulation
                if self.device.type == "cuda" and (i // self.micro_batch_size + 1) % 4 == 0:
                    torch.cuda.empty_cache()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[embed_sparse] Completed in {elapsed_ms:.2f}ms for {total_size} texts ({elapsed_ms/total_size:.2f}ms/text)"
        )
        return results

    def _embed_dense_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts for dense embeddings"""
        inputs = self._tokenize(texts, truncate=True)
        outputs = self._model_forward(inputs, return_dense=True, return_sparse=False)
        dense_vecs = outputs["dense_vecs"]
        dense_vecs = F.normalize(dense_vecs, p=2, dim=-1)
        result = dense_vecs.cpu().tolist()

        # === MEMORY CLEANUP: Explicitly delete GPU tensors ===
        del inputs, outputs, dense_vecs

        return result

    @torch.inference_mode()
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings with normalization and micro-batching"""
        start_time = time.perf_counter()
        total_size = len(texts)
        logger.info(f"[embed_dense] Input batch size: {total_size}, device: {self.device}")

        if total_size <= self.micro_batch_size:
            results = self._embed_dense_batch(texts)
        else:
            results = []
            for i in range(0, total_size, self.micro_batch_size):
                batch_texts = texts[i:i + self.micro_batch_size]
                batch_results = self._embed_dense_batch(batch_texts)
                results.extend(batch_results)

                # Clear GPU cache periodically to prevent memory accumulation
                if self.device.type == "cuda" and (i // self.micro_batch_size + 1) % 4 == 0:
                    torch.cuda.empty_cache()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"[embed_dense] Completed in {elapsed_ms:.2f}ms for {total_size} texts ({elapsed_ms/total_size:.2f}ms/text)"
        )
        return results