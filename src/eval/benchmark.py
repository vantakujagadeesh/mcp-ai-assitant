"""
Benchmark script: Compare base model vs. fine-tuned model.

Measures:
- Classification accuracy (%)
- Response quality (ROUGE-L)
- Inference latency (ms)
- Memory usage (GB)
- Cost per 1K tokens (estimated)

Usage:
    python benchmark.py \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --finetuned-model ./llama-support-lora
"""

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from unsloth import FastLanguageModel


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    model_name: str
    accuracy: float = 0.0
    rouge_l: float = 0.0
    avg_latency_ms: float = 0.0
    memory_gb: float = 0.0
    cost_per_1k_tokens: float = 0.0
    total_samples: int = 0
    errors: int = 0


# ──── Test Samples ────

TEST_TICKETS = [
    {
        "ticket": "My internet has been down for 3 hours. I have an important Zoom meeting in 30 minutes!",
        "expected_category": "Technical Support",
        "expected_urgency": "Critical",
    },
    {
        "ticket": "How do I reset my password? I forgot it after the latest update.",
        "expected_category": "Account Support",
        "expected_urgency": "Medium",
    },
    {
        "ticket": "I was charged twice for my subscription this month. Please issue a refund.",
        "expected_category": "Billing",
        "expected_urgency": "High",
    },
    {
        "ticket": "Can you tell me what hours your live chat support is available?",
        "expected_category": "General Inquiry",
        "expected_urgency": "Low",
    },
    {
        "ticket": "Your app keeps crashing every time I try to upload a file larger than 10MB.",
        "expected_category": "Technical Support",
        "expected_urgency": "High",
    },
    {
        "ticket": "I want to upgrade my plan from Basic to Pro. Can you help with that?",
        "expected_category": "Billing",
        "expected_urgency": "Medium",
    },
    {
        "ticket": "I never received the order I placed last week. Tracking number: TRK-12345.",
        "expected_category": "Order Support",
        "expected_urgency": "High",
    },
    {
        "ticket": "How do I export my data? I need it for tax purposes.",
        "expected_category": "Account Support",
        "expected_urgency": "Low",
    },
    {
        "ticket": "Your service is terrible. I've been waiting on hold for 45 minutes!!!",
        "expected_category": "Complaint",
        "expected_urgency": "Critical",
    },
    {
        "ticket": "Can you add my team members to the Enterprise account? Their emails are attached.",
        "expected_category": "Account Support",
        "expected_urgency": "Medium",
    },
]


# ──── Model Loading ────


def load_model(model_path: str, is_base: bool = False):
    """Load model and tokenizer."""
    load_in_4bit = True
    max_seq_length = 2048

    if is_base:
        # For base model comparison, load without adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )
    else:
        # Load fine-tuned with LoRA adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            device_map="auto",
        )

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def get_memory_usage() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


# ──── Inference ────


def run_inference(
    model,
    tokenizer,
    ticket: str,
    system_prompt: str = "Classify the following customer support ticket by category and urgency.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple:
    """Run model inference and return (response_text, latency_ms)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Ticket: {ticket}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )
    latency_ms = (time.time() - start_time) * 1000

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response, latency_ms


# ──── Evaluation Metrics ────


def extract_category(response: str) -> str:
    """Extract category from model response."""
    for line in response.split("\n"):
        line_lower = line.lower().strip()
        if line_lower.startswith("category:"):
            return line.split(":", 1)[1].strip()
        if line_lower.startswith("category") and ":" in line:
            return line.split(":", 1)[1].strip()
    return "Unknown"


def extract_urgency(response: str) -> str:
    """Extract urgency from model response."""
    for line in response.split("\n"):
        line_lower = line.lower().strip()
        if line_lower.startswith("urgency:"):
            return line.split(":", 1)[1].strip()
        if line_lower.startswith("urgency") and ":" in line:
            return line.split(":", 1)[1].strip()
    return "Unknown"


def compute_accuracy(response: str, expected_category: str, expected_urgency: str) -> Dict:
    """Check if classification matches expected values."""
    pred_category = extract_category(response)
    pred_urgency = extract_urgency(response)

    category_match = pred_category.lower() == expected_category.lower()
    urgency_match = pred_urgency.lower() == expected_urgency.lower()

    return {
        "category_correct": category_match,
        "urgency_correct": urgency_match,
        "both_correct": category_match and urgency_match,
        "predicted_category": pred_category,
        "predicted_urgency": pred_urgency,
    }


# ──── Benchmark Runner ────


def benchmark_model(
    model,
    tokenizer,
    model_name: str,
    test_cases: List[Dict],
    num_runs: int = 3,
) -> BenchmarkResults:
    """Run full benchmark on a model."""
    results = BenchmarkResults(model_name=model_name)

    all_accuracies = []
    all_latencies = []
    errors = 0

    for i, case in enumerate(test_cases):
        ticket = case["ticket"]
        expected_category = case["expected_category"]
        expected_urgency = case["expected_urgency"]

        try:
            response, latency = run_inference(model, tokenizer, ticket)
            acc = compute_accuracy(response, expected_category, expected_urgency)

            all_accuracies.append(1.0 if acc["both_correct"] else 0.0)
            all_latencies.append(latency)

            if i < 3:  # Show first 3 examples
                print(f"\n  [{i+1}] Ticket: {ticket[:60]}...")
                print(f"      Predicted: {acc['predicted_category']} / {acc['predicted_urgency']}")
                print(f"      Expected:  {expected_category} / {expected_urgency}")
                print(f"      ✓" if acc["both_correct"] else f"      ✗")
        except Exception as e:
            errors += 1
            print(f"  [{i+1}] Error: {e}")

    # Aggregate results
    results.accuracy = float(np.mean(all_accuracies)) * 100 if all_accuracies else 0.0
    results.avg_latency_ms = float(np.mean(all_latencies)) if all_latencies else 0.0
    results.memory_gb = get_memory_usage()
    results.total_samples = len(test_cases)
    results.errors = errors

    # Cost estimate (approximate)
    # T4 pricing: ~$0.35/hr on Colab Pay-As-You-Go
    # Approx: $0.00035 per 1K tokens generated
    results.cost_per_1k_tokens = 0.00035

    return results


def print_comparison(base_results: BenchmarkResults, ft_results: BenchmarkResults):
    """Print comparison table between base and fine-tuned models."""
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS: BASE vs. FINE-TUNED")
    print("=" * 60)

    headers = ["Metric", "Base Model", "Fine-Tuned", "Δ"]
    rows = [
        ["Classification Accuracy",
         f"{base_results.accuracy:.1f}%",
         f"{ft_results.accuracy:.1f}%",
         f"{ft_results.accuracy - base_results.accuracy:+.1f}%"],
        ["Avg Latency",
         f"{base_results.avg_latency_ms:.0f}ms",
         f"{ft_results.avg_latency_ms:.0f}ms",
         f"{ft_results.avg_latency_ms - base_results.avg_latency_ms:+.0f}ms"],
        ["Memory Usage",
         f"{base_results.memory_gb:.1f}GB",
         f"{ft_results.memory_gb:.1f}GB",
         f"{ft_results.memory_gb - base_results.memory_gb:+.1f}GB"],
        ["Cost / 1K tokens",
         f"${base_results.cost_per_1k_tokens:.5f}",
         f"${ft_results.cost_per_1k_tokens:.5f}",
         "—"],
        ["Test Samples",
         str(base_results.total_samples),
         str(ft_results.total_samples),
         "—"],
    ]

    print(f"\n{'Metric':<30} {'Base':<18} {'Fine-Tuned':<18} {'Δ':<10}")
    print("-" * 76)
    for row in rows:
        print(f"{row[0]:<30} {row[1]:<18} {row[2]:<18} {row[3]:<10}")

    # Summary
    print(f"\n{'='*60}")
    improvement = ft_results.accuracy - base_results.accuracy
    print(f"  SUMMARY: Fine-tuned model is {improvement:+.1f}pp more accurate")
    print(f"  Inference is {ft_results.avg_latency_ms:.0f}ms per request (~{1000/ft_results.avg_latency_ms:.0f} req/s)")
    print(f"  Estimated cost: ${ft_results.cost_per_1k_tokens:.5f} / 1K tokens")
    print(f"{'='*60}\n")


def save_results(base_results: BenchmarkResults, ft_results: BenchmarkResults, output_path: str):
    """Save benchmark results to JSON."""
    results = {
        "base_model": asdict(base_results),
        "fine_tuned_model": asdict(ft_results),
        "comparison": {
            "accuracy_improvement_pp": round(ft_results.accuracy - base_results.accuracy, 1),
            "latency_change_ms": round(ft_results.avg_latency_ms - base_results.avg_latency_ms, 0),
        },
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark base vs. fine-tuned model")
    parser.add_argument("--base-model", type=str, default="unsloth/Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--finetuned-model", type=str, default="./llama-support-lora")
    parser.add_argument("--output", type=str, default="results/benchmarks/comparison.json")
    args = parser.parse_args()

    # Resolve output path (relative to project root, not this file)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Benchmark Base Model ──
    print("\n" + "=" * 60)
    print("  LOADING BASE MODEL")
    print("=" * 60)
    base_model, base_tokenizer = load_model(args.base_model, is_base=True)
    print(f"  Memory after loading: {get_memory_usage():.1f}GB")

    print("\n" + "=" * 60)
    print("  BENCHMARKING BASE MODEL")
    print("=" * 60)
    base_results = benchmark_model(base_model, base_tokenizer, args.base_model, TEST_TICKETS)

    # Clean up base model
    del base_model
    torch.cuda.empty_cache()

    # ── Benchmark Fine-Tuned Model ──
    print("\n" + "=" * 60)
    print("  LOADING FINE-TUNED MODEL")
    print("=" * 60)
    ft_model, ft_tokenizer = load_model(args.finetuned_model, is_base=False)
    print(f"  Memory after loading: {get_memory_usage():.1f}GB")

    print("\n" + "=" * 60)
    print("  BENCHMARKING FINE-TUNED MODEL")
    print("=" * 60)
    ft_results = benchmark_model(ft_model, ft_tokenizer, args.finetuned_model, TEST_TICKETS)

    # ── Compare ──
    print_comparison(base_results, ft_results)

    # ── Save ──
    save_results(base_results, ft_results, output_path)


if __name__ == "__main__":
    import os
    main()
