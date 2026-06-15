import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def source(path: str) -> str:
    return (ROOT / path).read_text()


def function_source(path: str, name: str) -> str:
    text = source(path)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"Function {name} not found in {path}")


def class_method_source(path: str, class_name: str, method_name: str) -> str:
    text = source(path)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(text, item)
    raise AssertionError(f"Method {class_name}.{method_name} not found in {path}")


class RuntimeGuardTests(unittest.TestCase):
    def test_request_wrappers_clear_cuda_cache_in_finally(self):
        for name in ("run_embed_sparse_sync", "run_embed_dense_sync"):
            body = function_source("main.py", name)
            self.assertIn("finally:", body)
            self.assertIn("empty_cache", body)

    def test_gpu_multi_worker_guard_is_present(self):
        body = source("main.py")
        self.assertIn("workers != 1", body)
        self.assertIn("DEVICE", body)
        self.assertIn('torch.device(device).type == "cuda"', body)
        self.assertNotIn('device == "cuda" and workers != 1', body)

    def test_gpu_multi_worker_guard_handles_indexed_cuda_devices(self):
        body = function_source("main.py", "_validate_worker_config")
        self.assertIn("torch.device(device).type", body)
        self.assertNotIn('device == "cuda"', body)

    def test_sparse_threshold_uses_in_place_mask_without_fp32_promotion(self):
        body = class_method_source("model.py", "BGEM3SparseModel", "_embed_sparse_batch")
        self.assertIn("masked_fill_", body)
        self.assertNotIn("(sparse_dense > 0.001).float()", body)

    def test_sparse_batch_reuses_dense_buffer(self):
        body = source("model.py")
        self.assertIn("canonicalize_torch_device", body)
        self.assertIn("self.device = canonicalize_torch_device(device)", body)
        self.assertIn("_get_sparse_dense_buffer", body)
        sparse_body = class_method_source("model.py", "BGEM3SparseModel", "_embed_sparse_batch")
        self.assertIn("_get_sparse_dense_buffer", sparse_body)
        self.assertNotIn("torch.zeros(", sparse_body)

    def test_sparse_weight_transfer_casts_only_nonzero_weights_to_float32(self):
        body = class_method_source("model.py", "BGEM3SparseModel", "_embed_sparse_batch")
        self.assertIn("weights_gpu.float().cpu().numpy()", body)

    def test_cuda_timing_sync_is_guarded(self):
        body = class_method_source("model.py", "BGEM3SparseModel", "_embed_sparse_batch")
        self.assertIn("ENABLE_TIMING", source("model.py"))
        self.assertNotIn("torch.cuda.synchronize()", body)
        self.assertIn("_sync_for_timing", body)


if __name__ == "__main__":
    unittest.main()
