"""
BGE-M3 Sparse Model - Maximum Speed Optimization

Key optimizations:
1. Uses FlagEmbedding BGEM3FlagModel for native sparse support
2. torch.compile() for GPU kernel fusion (PyTorch 2.0+)
3. Pre-allocated tensors for sparse processing
4. No torch.unique() - uses scatter_reduce instead
5. Minimal CPU-GPU transfers
6. Float16 inference on GPU
7. Cached special token mask (created once, not per-batch)
"""

import os
import logging
from typing import List, Dict, Any, Set

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Disable tqdm progress bars
os.environ["TQDM_DISABLE"] = "1"


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
        max_length: int = 8192,
        compile_model: bool = True,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_length = max_length
        
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
                "max_split_size_mb:512,expandable_segments:True"
            )
            torch.cuda.empty_cache()
        
        use_fp16 = dtype in [torch.float16, torch.bfloat16]
        
        logger.info(f"Loading BGEM3FlagModel from {model_id}...")
        self.flag_model = BGEM3FlagModel(
            model_id,
            use_fp16=use_fp16,
            device=str(self.device),
        )
        
        # Get internal model for direct forward pass
        self._model = self.flag_model.model
        self._model = self._model.to(self.device)
        self._model.eval()
        
        # Get tokenizer
        self.tokenizer = self.flag_model.tokenizer
        self.vocab_size = len(self.tokenizer)
        
        # Cache special token IDs
        self._special_token_ids: Set[int] = set()
        for attr in ["cls_token_id", "sep_token_id", "pad_token_id", "unk_token_id", "mask_token_id"]:
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                self._special_token_ids.add(tid)
        
        # Pre-create special tokens mask on device (CRITICAL optimization!)
        self._special_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
        for tid in self._special_token_ids:
            if 0 <= tid < self.vocab_size:
                self._special_mask[tid] = True
        
        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile') and self.device.type == "cuda":
            logger.info("Compiling model with torch.compile()...")
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
                logger.info("Model compiled successfully!")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}, continuing without compilation")
        
        # Warmup
        self._warmup()
        
        logger.info(f"Model ready: device={self.device}, dtype={self.dtype}, vocab={self.vocab_size}")
    
    def _warmup(self):
        """Warmup the model with dummy batches"""
        logger.info("Warming up model...")
        dummy_texts = ["warmup text"] * min(4, self.max_batch_size)
        
        with torch.inference_mode():
            # Warmup sparse
            _ = self.embed_sparse(dummy_texts)
            # Warmup dense  
            _ = self.embed_dense(dummy_texts)
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        
        logger.info("Warmup complete")
    
    def _tokenize(self, texts: List[str], truncate: bool = True) -> Dict[str, torch.Tensor]:
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
            "token_type_ids": torch.zeros_like(encoded["input_ids"], device=self.device),
        }
    
    @torch.inference_mode()
    def embed_sparse(
        self, 
        texts: List[str],
        truncate: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate sparse embeddings using optimized pipeline.
        
        Returns: List of sparse embeddings, each is List[{"index": int, "value": float}]
        """
        batch_size = len(texts)
        
        # Tokenize
        inputs = self._tokenize(texts, truncate)
        input_ids = inputs["input_ids"]
        
        # Forward pass to get sparse weights
        outputs = self._model(
            inputs,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            return_sparse_embedding=False,
        )
        
        # Get token weights
        token_weights = outputs["sparse_vecs"]
        if token_weights.dim() == 3:
            token_weights = token_weights.squeeze(-1)
        
        token_weights = torch.relu(token_weights)
        
        # ============================================================
        # FAST SPARSE EXTRACTION - No torch.unique()!
        # Uses scatter_reduce for GPU-native aggregation
        # ============================================================
        
        # Create dense (batch, vocab) tensor for max weights per token
        sparse_dense = torch.zeros(
            (batch_size, self.vocab_size),
            device=self.device,
            dtype=token_weights.dtype,
        )
        
        # Mask special tokens (use pre-cached mask)
        is_special = self._special_mask[input_ids]  # (batch, seq_len)
        token_weights_masked = token_weights.masked_fill(is_special, 0.0)
        
        # Scatter max: aggregate max weight per token ID
        # CRITICAL: scatter_reduce requires int64 index
        sparse_dense.scatter_reduce_(
            dim=1,
            index=input_ids.long(),
            src=token_weights_masked,
            reduce='amax',
            include_self=True,
        )
        
        # Get non-zero mask
        nonzero_mask = sparse_dense > 0
        
        # Transfer to CPU ONCE (single bulk transfer)
        sparse_dense_cpu = sparse_dense.cpu()
        nonzero_mask_cpu = nonzero_mask.cpu()
        
        # Build results from CPU tensors
        results = []
        for i in range(batch_size):
            mask_i = nonzero_mask_cpu[i]
            if mask_i.any():
                indices = torch.where(mask_i)[0].tolist()
                weights = sparse_dense_cpu[i, mask_i].tolist()
                # Filter very small weights for efficiency
                sparse_values = [
                    {"index": idx, "value": round(w, 6)}
                    for idx, w in zip(indices, weights)
                    if w > 0.001
                ]
            else:
                sparse_values = []
            results.append(sparse_values)
        
        return results
    
    @torch.inference_mode()
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings with normalization"""
        inputs = self._tokenize(texts, truncate=True)
        
        outputs = self._model(
            inputs,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        
        dense_vecs = outputs["dense_vecs"]
        dense_vecs = F.normalize(dense_vecs, p=2, dim=-1)
        
        return dense_vecs.cpu().tolist()
