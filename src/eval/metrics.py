"""
Evaluation metrics for comparing base model vs. fine-tuned model.

Metrics:
- Classification accuracy (exact match, partial)
- ROUGE-L for response similarity
- BERTScore for semantic similarity
- Response length statistics
- Confidence scoring
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional metric libraries
try:
    import evaluate
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("evaluate library not installed. Install with: pip install evaluate rouge-score")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("bert-score not installed. Install with: pip install bert-score")


# ──── Classification Metrics ────


def extract_field(response: str, field: str) -> str:
    """Extract a field value from a structured response (e.g., 'Category: Billing')."""
    pattern = rf"(?:^|\n){field}\s*:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for any line containing the field name
    for line in response.split("\n"):
        if field.lower() in line.lower() and ":" in line:
            return line.split(":", 1)[1].strip()
    return "Unknown"


def classification_accuracy(
    responses: List[str],
    expected_categories: List[str],
    expected_urgencies: List[str],
) -> Dict[str, float]:
    """Compute classification accuracy for category and urgency.

    Returns:
        Dict with category_accuracy, urgency_accuracy, overall_accuracy.
    """
    correct_cat = 0
    correct_urg = 0
    correct_both = 0
    total = len(responses)

    for i, response in enumerate(responses):
        pred_cat = extract_field(response, "Category")
        pred_urg = extract_field(response, "Urgency")

        if pred_cat.lower() == expected_categories[i].lower():
            correct_cat += 1
        if pred_urg.lower() == expected_urgencies[i].lower():
            correct_urg += 1
        if (
            pred_cat.lower() == expected_categories[i].lower()
            and pred_urg.lower() == expected_urgencies[i].lower()
        ):
            correct_both += 1

    return {
        "category_accuracy": correct_cat / total if total > 0 else 0.0,
        "urgency_accuracy": correct_urg / total if total > 0 else 0.0,
        "overall_accuracy": correct_both / total if total > 0 else 0.0,
        "total_samples": total,
    }


# ──── Response Quality Metrics ────


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

    Args:
        predictions: Model-generated responses.
        references: Ground truth responses.

    Returns:
        Dict with rouge1, rouge2, rougeL f-measures.
    """
    if not ROUGE_AVAILABLE:
        logger.warning("ROUGE not available. Returning zeros.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True,
    )
    return {
        "rouge1": results.get("rouge1", 0.0),
        "rouge2": results.get("rouge2", 0.0),
        "rougeL": results.get("rougeL", 0.0),
    }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
) -> Dict[str, float]:
    """Compute BERTScore (precision, recall, F1).

    Args:
        predictions: Model-generated responses.
        references: Ground truth responses.
        lang: Language code.

    Returns:
        Dict with precision, recall, f1.
    """
    if not BERTSCORE_AVAILABLE:
        logger.warning("BERTScore not available. Returning zeros.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


# ──── Confidence Scoring ────


def compute_confidence(response: str) -> float:
    """Estimate confidence of a generated response.

    Heuristic based on:
    - Structure adherence (has Category/Urgency fields)
    - Response length (too short = low confidence)
    - Presence of uncertainty markers

    Returns:
        Float between 0.0 and 1.0.
    """
    score = 1.0

    # Structure adherence (+0.2 each)
    if "Category:" in response or "category:" in response:
        score += 0.2
    else:
        score -= 0.2

    if "Urgency:" in response or "urgency:" in response:
        score += 0.2
    else:
        score -= 0.1

    # Length penalty
    words = response.split()
    if len(words) < 10:
        score -= 0.3
    elif len(words) > 500:
        score -= 0.1  # Too verbose may be rambling

    # Uncertainty markers
    uncertainty_phrases = [
        "i'm not sure", "i don't know", "maybe", "perhaps",
        "i think", "not certain", "unclear",
    ]
    for phrase in uncertainty_phrases:
        if phrase in response.lower():
            score -= 0.15
            break

    return max(0.0, min(1.0, score))


# ──── Comprehensive Evaluation ────


def evaluate_all(
    responses: List[str],
    reference_categories: List[str],
    reference_urgencies: List[str],
    reference_responses: Optional[List[str]] = None,
) -> Dict:
    """Run all evaluation metrics.

    Args:
        responses: Model-generated responses.
        reference_categories: Expected categories.
        reference_urgencies: Expected urgencies.
        reference_responses: Optional ground truth responses for ROUGE/BERTScore.

    Returns:
        Dict with all metrics.
    """
    results = {}

    # Classification accuracy
    results["classification"] = classification_accuracy(
        responses, reference_categories, reference_urgencies
    )

    # Confidence
    confidences = [compute_confidence(r) for r in responses]
    results["confidence"] = {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "low_confidence_ratio": float(sum(1 for c in confidences if c < 0.5) / len(confidences))
        if confidences
        else 0.0,
    }

    # Response quality (if references available)
    if reference_responses:
        results["rouge"] = compute_rouge(responses, reference_responses)
        results["bertscore"] = compute_bertscore(responses, reference_responses)

    # Response statistics
    word_counts = [len(r.split()) for r in responses]
    results["response_stats"] = {
        "mean_words": float(np.mean(word_counts)),
        "median_words": float(np.median(word_counts)),
        "min_words": int(np.min(word_counts)),
        "max_words": int(np.max(word_counts)),
    }

    return results


def compare_models(
    base_results: Dict,
    ft_results: Dict,
) -> Dict:
    """Compare base model vs. fine-tuned model results.

    Returns:
        Dict with delta metrics.
    """
    comparison = {}

    for metric in ["classification", "confidence", "rouge", "bertscore", "response_stats"]:
        if metric in base_results and metric in ft_results:
            comparison[metric] = {}
            for key in ft_results[metric]:
                if key in base_results[metric] and isinstance(ft_results[metric][key], (int, float)):
                    comparison[metric][f"{key}_delta"] = (
                        ft_results[metric][key] - base_results[metric][key]
                    )

    if "classification" in comparison:
        acc_delta = comparison["classification"].get("overall_accuracy_delta", 0)
        comparison["summary"] = f"Fine-tuned model is {acc_delta * 100:+.1f}% more accurate"

    return comparison


if __name__ == "__main__":
    # Quick test
    test_responses = [
        "Category: Billing\nUrgency: High\nResponse: We'll issue a refund.",
        "Category: Technical Support\nUrgency: Low\nResponse: Please restart your router.",
    ]
    test_categories = ["Billing", "Technical Support"]
    test_urgencies = ["High", "Low"]

    results = evaluate_all(test_responses, test_categories, test_urgencies)
    print("Evaluation results:")
    import json
    print(json.dumps(results, indent=2))
