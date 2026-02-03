#!/usr/bin/env python3
"""
Test script to compare custom sparse embedding implementation
against official FlagEmbedding lexical_weights output.

Usage:
    python test_lexical_weights.py
    python test_lexical_weights.py --verbose
    python test_lexical_weights.py --threshold 1e-4
"""

import argparse
import os
import sys
from typing import Dict, List, Any, Tuple

import torch
import numpy as np


def load_models(model_id: str = "BAAI/bge-m3", device: str = None):
    """Load both official FlagEmbedding model and custom implementation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading models on {device}...")

    # Load official FlagEmbedding model
    from FlagEmbedding import BGEM3FlagModel
    flag_model = BGEM3FlagModel(
        model_id,
        use_fp16=(device == "cuda"),
        device=device,
    )

    # Load custom implementation
    from model import BGEM3SparseModel
    custom_model = BGEM3SparseModel(
        model_id=model_id,
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("Models loaded.\n")
    return flag_model, custom_model


def get_official_lexical_weights(flag_model, texts: List[str]) -> List[Dict[int, float]]:
    """Get lexical weights using official FlagEmbedding API."""
    output = flag_model.encode(texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
    return output['lexical_weights']


def get_custom_sparse_output(custom_model, texts: List[str]) -> List[Dict[int, float]]:
    """Get sparse output from custom implementation and convert to dict format."""
    results = custom_model.embed_sparse(texts, truncate=True)

    # Convert list of {index, value} to dict {index: value}
    dict_results = []
    for text_result in results:
        token_dict = {item['index']: item['value'] for item in text_result}
        dict_results.append(token_dict)

    return dict_results


def compare_lexical_weights(
    official: Dict[int, float],
    custom: Dict[int, float],
    threshold: float = 1e-4,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare official lexical_weights with custom implementation output.

    Returns:
        (is_match, details_dict)
    """
    all_tokens = set(official.keys()) | set(custom.keys())

    missing_in_custom = set(official.keys()) - set(custom.keys())
    extra_in_custom = set(custom.keys()) - set(official.keys())
    common_tokens = set(official.keys()) & set(custom.keys())

    # Check value differences for common tokens
    value_diffs = []
    for token_id in common_tokens:
        official_val = official[token_id]
        custom_val = custom[token_id]
        diff = abs(official_val - custom_val)
        if diff > threshold:
            value_diffs.append({
                'token_id': token_id,
                'official': official_val,
                'custom': custom_val,
                'diff': diff,
            })

    # Check if missing tokens have significant weights
    significant_missing = []
    for token_id in missing_in_custom:
        weight = official[token_id]
        if weight > threshold:
            significant_missing.append({'token_id': token_id, 'weight': weight})

    significant_extra = []
    for token_id in extra_in_custom:
        weight = custom[token_id]
        if weight > threshold:
            significant_extra.append({'token_id': token_id, 'weight': weight})

    is_match = (
        len(significant_missing) == 0 and
        len(significant_extra) == 0 and
        len(value_diffs) == 0
    )

    details = {
        'total_official_tokens': len(official),
        'total_custom_tokens': len(custom),
        'common_tokens': len(common_tokens),
        'missing_in_custom': len(missing_in_custom),
        'extra_in_custom': len(extra_in_custom),
        'significant_missing': significant_missing,
        'significant_extra': significant_extra,
        'value_differences': value_diffs,
        'is_match': is_match,
    }

    return is_match, details


def print_comparison_result(
    text: str,
    details: Dict[str, Any],
    tokenizer=None,
    verbose: bool = False,
):
    """Print comparison results in a readable format."""
    is_match = details['is_match']
    status = "PASS" if is_match else "FAIL"

    print(f"  [{status}] \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    print(f"       Official tokens: {details['total_official_tokens']}, Custom tokens: {details['total_custom_tokens']}")

    if not is_match or verbose:
        if details['significant_missing']:
            print(f"       Missing in custom ({len(details['significant_missing'])}):")
            for item in details['significant_missing'][:5]:
                token_str = ""
                if tokenizer:
                    token_str = f" ({tokenizer.decode([item['token_id']])})"
                print(f"         - ID {item['token_id']}{token_str}: {item['weight']:.6f}")
            if len(details['significant_missing']) > 5:
                print(f"         ... and {len(details['significant_missing']) - 5} more")

        if details['significant_extra']:
            print(f"       Extra in custom ({len(details['significant_extra'])}):")
            for item in details['significant_extra'][:5]:
                token_str = ""
                if tokenizer:
                    token_str = f" ({tokenizer.decode([item['token_id']])})"
                print(f"         - ID {item['token_id']}{token_str}: {item['weight']:.6f}")
            if len(details['significant_extra']) > 5:
                print(f"         ... and {len(details['significant_extra']) - 5} more")

        if details['value_differences']:
            print(f"       Value differences ({len(details['value_differences'])}):")
            for item in details['value_differences'][:5]:
                token_str = ""
                if tokenizer:
                    token_str = f" ({tokenizer.decode([item['token_id']])})"
                print(f"         - ID {item['token_id']}{token_str}: official={item['official']:.6f}, custom={item['custom']:.6f}, diff={item['diff']:.6f}")
            if len(details['value_differences']) > 5:
                print(f"         ... and {len(details['value_differences']) - 5} more")


def run_tests(
    flag_model,
    custom_model,
    threshold: float = 1e-4,
    verbose: bool = False,
):
    """Run comparison tests on various test cases."""

    test_cases = [
        # Basic English
        "What is BGE M3?",
        "Definition of BM25",
        "Hello world, this is a test for sparse embeddings.",

        # Repeated tokens
        "the cat sat on the mat and the dog sat on the rug",

        # Technical content
        "Machine learning models use neural networks for deep learning tasks.",

        # Multi-language (if supported)
        "這是一個中文測試句子。",
        "これは日本語のテスト文です。",

        # Long text
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to the natural intelligence displayed by animals including humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals.",

        # Special characters
        "Hello! How are you? I'm fine, thanks. #AI @ML $100",

        # Numbers
        "The year 2024 has 365 days and 12 months.",

        # Empty-ish
        "a",
        "   ",
    ]

    print("=" * 60)
    print("LEXICAL WEIGHTS COMPARISON TEST")
    print(f"Threshold: {threshold}")
    print("=" * 60)
    print()

    passed = 0
    failed = 0

    tokenizer = custom_model.tokenizer

    for text in test_cases:
        if not text.strip():
            print(f"  [SKIP] Empty text")
            continue

        try:
            # Get outputs from both implementations
            official_weights = get_official_lexical_weights(flag_model, [text])[0]
            custom_weights = get_custom_sparse_output(custom_model, [text])[0]

            # Compare
            is_match, details = compare_lexical_weights(
                official_weights,
                custom_weights,
                threshold=threshold,
                verbose=verbose,
            )

            print_comparison_result(text, details, tokenizer, verbose)

            if is_match:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  [ERROR] \"{text[:30]}...\": {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def run_batch_test(
    flag_model,
    custom_model,
    batch_size: int = 32,
    threshold: float = 1e-4,
):
    """Test batch processing consistency."""
    print()
    print("=" * 60)
    print(f"BATCH PROCESSING TEST (batch_size={batch_size})")
    print("=" * 60)

    texts = [f"Test sentence number {i} for batch processing." for i in range(batch_size)]

    # Get batch outputs
    official_weights = get_official_lexical_weights(flag_model, texts)
    custom_weights = get_custom_sparse_output(custom_model, texts)

    passed = 0
    failed = 0

    for i, (official, custom) in enumerate(zip(official_weights, custom_weights)):
        is_match, details = compare_lexical_weights(official, custom, threshold=threshold)
        if is_match:
            passed += 1
        else:
            failed += 1
            print(f"  [FAIL] Batch item {i}: {len(details['value_differences'])} diffs, "
                  f"{len(details['significant_missing'])} missing, {len(details['significant_extra'])} extra")

    print(f"\n  Batch result: {passed}/{batch_size} passed")
    print("=" * 60)

    return failed == 0


def run_numerical_analysis(
    flag_model,
    custom_model,
    text: str = "Machine learning models use neural networks for deep learning.",
):
    """Detailed numerical analysis of a single text."""
    print()
    print("=" * 60)
    print("DETAILED NUMERICAL ANALYSIS")
    print(f"Text: \"{text}\"")
    print("=" * 60)

    tokenizer = custom_model.tokenizer

    # Get outputs
    official_weights = get_official_lexical_weights(flag_model, [text])[0]
    custom_weights = get_custom_sparse_output(custom_model, [text])[0]

    # Get tokenization for reference
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=True)

    print(f"\nTokenization: {tokens}")
    print(f"Token IDs: {token_ids}")

    # Compare all tokens
    all_token_ids = sorted(set(official_weights.keys()) | set(custom_weights.keys()))

    print(f"\n{'Token ID':<10} {'Token':<15} {'Official':<12} {'Custom':<12} {'Diff':<12} {'Status'}")
    print("-" * 75)

    for token_id in all_token_ids:
        token_str = tokenizer.decode([token_id]).strip()
        if len(token_str) > 12:
            token_str = token_str[:12] + "..."

        official_val = official_weights.get(token_id, 0.0)
        custom_val = custom_weights.get(token_id, 0.0)
        diff = abs(official_val - custom_val)

        status = "OK" if diff < 1e-4 else "DIFF"
        if token_id not in official_weights:
            status = "EXTRA"
        elif token_id not in custom_weights:
            status = "MISSING"

        print(f"{token_id:<10} {token_str:<15} {official_val:<12.6f} {custom_val:<12.6f} {diff:<12.6f} {status}")

    # Summary statistics
    official_values = list(official_weights.values())
    custom_values = list(custom_weights.values())

    print(f"\nStatistics:")
    print(f"  Official: count={len(official_values)}, sum={sum(official_values):.4f}, "
          f"max={max(official_values):.4f}, mean={np.mean(official_values):.4f}")
    print(f"  Custom:   count={len(custom_values)}, sum={sum(custom_values):.4f}, "
          f"max={max(custom_values):.4f}, mean={np.mean(custom_values):.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare lexical weights implementations")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Model ID")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Comparison threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--batch-test", action="store_true", help="Run batch test")
    parser.add_argument("--numerical", action="store_true", help="Run detailed numerical analysis")
    args = parser.parse_args()

    # Load models
    flag_model, custom_model = load_models(args.model, args.device)

    # Run tests
    all_passed = True

    all_passed &= run_tests(flag_model, custom_model, args.threshold, args.verbose)

    if args.batch_test:
        all_passed &= run_batch_test(flag_model, custom_model, threshold=args.threshold)

    if args.numerical:
        run_numerical_analysis(flag_model, custom_model)

    print()
    if all_passed:
        print("All tests PASSED!")
        sys.exit(0)
    else:
        print("Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
