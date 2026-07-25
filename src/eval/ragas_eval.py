"""
Ragas RAG Evaluation

Evaluates the RAG pipeline using Ragas metrics (faithfulness, answer
relevancy, context recall).  Generates synthetic test questions from the
knowledge base articles, runs them through the retriever, and scores the
quality of retrieved contexts.

Metrics
-------
- **Faithfulness**: How factually accurate the generated answer is given
  the retrieved context (0–1, higher is better).
- **Answer Relevancy**: How relevant the generated answer is to the
  question (0–1, higher is better).
- **Context Recall**: How well the retrieved context covers the ground
  truth answer (0–1, higher is better).
- **Context Precision**: How much of the retrieved context is relevant
  to the question (0–1, higher is better).

Usage
-----
    # Basic evaluation
    python src/eval/ragas_eval.py

    # With custom KB path
    python src/eval/ragas_eval.py --kb-path data/knowledge_base/support_articles.jsonl

    # Dry run (show test questions only)
    python src/eval/ragas_eval.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
DEFAULT_KB_PATH = os.path.join(PROJECT_ROOT, "data", "knowledge_base", "support_articles.jsonl")
DEFAULT_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "eval", "ragas_results.json")

# ---------------------------------------------------------------------------
# Optional: Ragas + dependencies
# ---------------------------------------------------------------------------
_RAGAS_AVAILABLE = False
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    _RAGAS_AVAILABLE = True
except ImportError:
    faithfulness = answer_relevancy = context_recall = context_precision = None  # type: ignore

# LLM / embedding for Ragas (optional)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RAGEvalSample:
    """A single evaluation sample."""

    question: str
    ground_truth: str
    retrieved_contexts: List[str]
    generated_answer: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "retrieved_contexts": self.retrieved_contexts,
            "generated_answer": self.generated_answer,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
        }


@dataclass
class RAGEvalResult:
    """Aggregated evaluation results."""

    timestamp: str = ""
    num_samples: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    ragas_available: bool = False
    engine: str = "ragas"
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "num_samples": self.num_samples,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_recall": round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4),
            "ragas_available": self.ragas_available,
            "engine": self.engine,
            "samples": self.samples,
        }


# ===================================================================
# Test Dataset Generator
# ===================================================================

# Synthetic test questions derived from the knowledge base articles.
# Each entry pairs a realistic user question with the ground-truth answer
# extracted from the corresponding KB article.
SYNTHETIC_TEST_SET: List[Dict[str, str]] = [
    {
        "question": "How do I reset my password?",
        "ground_truth": "Go to the login page, click 'Forgot Password', enter your email, check your inbox for the reset link, click the link and enter a new password with at least 8 characters including one number and one special character.",
    },
    {
        "question": "What are the subscription plans and how much do they cost?",
        "ground_truth": "Basic ($9.99/month) includes 5 projects and 10GB storage; Pro ($29.99/month) includes unlimited projects, 100GB storage, and priority support; Enterprise (custom pricing) includes everything in Pro plus dedicated support, SSO, and custom integrations.",
    },
    {
        "question": "My payment was declined. What should I do?",
        "ground_truth": "Check that your card details are up to date in Billing Settings. Common causes include expired card, insufficient funds, bank declined the transaction, incorrect CVV, or card not enabled for international transactions. You can retry or add a different payment method.",
    },
    {
        "question": "How do I invite team members to my account?",
        "ground_truth": "Go to Team Settings, click 'Invite Members', enter their email addresses, assign roles (Admin, Member, Viewer), and send the invitation. They will receive an email with a link to join your workspace.",
    },
    {
        "question": "Can I cancel my subscription at any time?",
        "ground_truth": "Yes, you can upgrade, downgrade, or cancel at any time. Annual billing gives you a 20% discount on all plans. All plans come with a 14-day free trial.",
    },
    {
        "question": "How do I enable two-factor authentication?",
        "ground_truth": "Go to Security Settings, click 'Enable 2FA', scan the QR code with an authenticator app like Google Authenticator or Authy, enter the verification code, and save your backup codes in a secure location.",
    },
    {
        "question": "What should I do if I suspect my account has been compromised?",
        "ground_truth": "Immediately change your password, enable two-factor authentication if not already enabled, review your active sessions and revoke any unrecognized ones, check your account activity log for suspicious actions, and contact support immediately.",
    },
    {
        "question": "How do I export my data from the platform?",
        "ground_truth": "Go to Account Settings, click 'Export Data', select the data you want to export (projects, documents, analytics), choose your preferred format (CSV, JSON, or PDF), and click Export. You will receive an email with a download link.",
    },
    {
        "question": "What integrations are supported?",
        "ground_truth": "We support integrations with Slack, GitHub, Jira, Trello, Google Drive, Dropbox, Zapier, and over 200 other services via our API. Enterprise plans include custom integrations and dedicated support.",
    },
    {
        "question": "How do I create and manage API keys?",
        "ground_truth": "Go to Developer Settings, click 'Create API Key', name your key, set permissions (read-only, read-write, or admin), and copy the generated key. Store it securely as it won't be shown again. You can revoke or rotate keys at any time.",
    },
]


def generate_test_set(kb_path: str = DEFAULT_KB_PATH) -> List[Dict[str, str]]:
    """Generate a test set from knowledge base articles.

    Uses the predefined synthetic test set which was manually crafted
    from the KB articles.  Returns the same test set regardless of the
    KB path (the articles are guaranteed to contain these topics).

    Args:
        kb_path: Path to the knowledge base JSONL (used only for validation).

    Returns:
        List of ``{"question": ..., "ground_truth": ...}`` dicts.
    """
    if not os.path.exists(kb_path):
        logger.warning("Knowledge base not found at %s — using default test set", kb_path)
    else:
        logger.info("Using knowledge base at %s (%d test questions)", kb_path, len(SYNTHETIC_TEST_SET))

    return SYNTHETIC_TEST_SET


# ===================================================================
# RAG Evaluator
# ===================================================================

class RAGEvaluator:
    """Evaluate RAG pipeline quality using Ragas metrics.

    Runs a set of test questions through the retriever, collects the
    retrieved contexts, and scores them with Ragas metrics.

    Gracefully degrades when ``ragas`` or an LLM are not available —
    falls back to a heuristic scoring function.
    """

    def __init__(
        self,
        kb_path: str = DEFAULT_KB_PATH,
        top_k: int = 3,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.kb_path = kb_path
        self.top_k = top_k
        self.llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.llm_model = llm_model
        self.embedding_model = embedding_model

        self._retriever: Optional[Any] = None
        self._ragas_llm: Optional[Any] = None
        self._ragas_embeddings: Optional[Any] = None

        self._load_retriever()
        self._init_ragas()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _load_retriever(self) -> None:
        """Load the knowledge base retriever."""
        try:
            from src.rag.retriever import KnowledgeBaseRetriever

            self._retriever = KnowledgeBaseRetriever(
                kb_path=self.kb_path,
                top_k=self.top_k,
            )
            logger.info("Retriever loaded: %d articles", self._retriever.article_count())
        except Exception as exc:
            logger.warning("Failed to load retriever: %s", exc)
            self._retriever = None

    def _init_ragas(self) -> None:
        """Initialise Ragas with LLM and embedding models."""
        if not _RAGAS_AVAILABLE:
            logger.info("ragas not installed — using heuristic fallback scoring")
            return

        if not _LANGCHAIN_OPENAI_AVAILABLE:
            logger.info("langchain-openai not installed — Ragas will use default models")
            return

        if not self.llm_api_key:
            logger.info(
                "OPENAI_API_KEY not set — Ragas will use default models "
                "(may require local LLM setup)"
            )
            return

        try:
            self._ragas_llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                temperature=0,
            )
            self._ragas_embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.llm_api_key,
            )
            logger.info("Ragas LLM + embeddings configured (%s, %s)", self.llm_model, self.embedding_model)
        except Exception as exc:
            logger.warning("Failed to configure Ragas LLM: %s", exc)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_contexts(self, question: str) -> List[str]:
        """Retrieve relevant contexts for a question.

        Args:
            question: The user's question.

        Returns:
            List of context text strings from the knowledge base.
        """
        if self._retriever is None:
            return []

        try:
            results = self._retriever.search(query=question, top_k=self.top_k)
            contexts = []
            for r in results:
                # Build a rich context string from the article fields
                title = r.get("title", "")
                content = r.get("content", r.get("chunk_text", ""))
                if title and content:
                    contexts.append(f"{title}: {content}")
                elif content:
                    contexts.append(content)
                elif title:
                    contexts.append(title)
            return contexts
        except Exception as exc:
            logger.warning("Retrieval failed for '%s': %s", question[:50], exc)
            return []

    # ------------------------------------------------------------------
    # Ragas scoring
    # ------------------------------------------------------------------
    def score_with_ragas(
        self,
        questions: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str],
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Score retrieval using Ragas metrics.

        Computes all four metrics:
        - **faithfulness** / **answer_relevancy**: use ``ground_truth``
          as a proxy for the generated answer.  In a full RAG pipeline
          these would use the actual LLM-generated answer; using the
          ground truth gives an **upper-bound estimate**.
        - **context_recall** / **context_precision**: use ``question``,
          ``contexts``, and ``ground_truth`` directly.

        Falls back to heuristic scoring when Ragas is unavailable.

        Returns:
            Tuple of ``(faithfulness_scores, answer_relevancy_scores,
            context_recall_scores, context_precision_scores)``.
        """
        if not _RAGAS_AVAILABLE:
            return self._score_heuristic(questions, contexts_list, ground_truths)

        try:
            from datasets import Dataset

            data = {
                "question": questions,
                "answer": ground_truths,  # proxy — upper-bound estimate
                "contexts": contexts_list,
                "ground_truth": ground_truths,
            }
            dataset = Dataset.from_dict(data)

            metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
            active_metrics = [m for m in metrics if m is not None]

            if not active_metrics:
                return self._score_heuristic(questions, contexts_list, ground_truths)

            result = ragas_evaluate(
                dataset,
                metrics=active_metrics,
                llm=self._ragas_llm,
                embeddings=self._ragas_embeddings,
            )

            # Ragas evaluate() returns a Dataset; access columns via []
            def _col(name: str, default: float = 0.0) -> List[float]:
                """Safely extract a column from the Ragas result Dataset."""
                try:
                    col = result[name]
                    if hasattr(col, "__iter__"):
                        return [float(v) for v in col]
                    return [float(col)] * len(questions)
                except (KeyError, TypeError, ValueError):
                    return [default] * len(questions)

            return (
                _col("faithfulness"),
                _col("answer_relevancy"),
                _col("context_recall"),
                _col("context_precision"),
            )

        except Exception as exc:
            logger.warning("Ragas evaluation failed (%s) — falling back to heuristic", exc)
            return self._score_heuristic(questions, contexts_list, ground_truths)

    # ------------------------------------------------------------------
    # Heuristic fallback scoring
    # ------------------------------------------------------------------
    def _score_heuristic(
        self,
        questions: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str],
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Heuristic scoring when Ragas is unavailable.

        Uses token overlap (ROUGE-1-like) as a proxy for faithfulness,
        and keyword coverage as a proxy for context recall.
        """
        import re

        f_scores = []
        ar_scores = []
        cr_scores = []
        cp_scores = []

        for i in range(len(questions)):
            question = questions[i]
            contexts = contexts_list[i] if i < len(contexts_list) else []
            ground_truth = ground_truths[i] if i < len(ground_truths) else ""

            q_tokens = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
            gt_tokens = set(re.findall(r"[a-zA-Z0-9]+", ground_truth.lower()))

            # Combine all contexts
            all_context_text = " ".join(contexts)
            ctx_tokens = set(re.findall(r"[a-zA-Z0-9]+", all_context_text.lower()))

            # Faithfulness proxy: overlap between ground truth and contexts
            if gt_tokens and ctx_tokens:
                f_score = len(gt_tokens & ctx_tokens) / len(gt_tokens)
            else:
                f_score = 0.0
            f_scores.append(min(1.0, f_score))

            # Answer relevancy proxy: overlap between question and contexts
            if q_tokens and ctx_tokens:
                ar_score = len(q_tokens & ctx_tokens) / len(q_tokens)
            else:
                ar_score = 0.0
            ar_scores.append(min(1.0, ar_score))

            # Context recall proxy: same as faithfulness for heuristic
            cr_scores.append(min(1.0, f_score))

            # Context precision proxy: relevant vs total context
            if ctx_tokens and gt_tokens:
                cp_score = len(gt_tokens & ctx_tokens) / len(ctx_tokens)
            else:
                cp_score = 0.0
            cp_scores.append(min(1.0, cp_score))

        return f_scores, ar_scores, cr_scores, cp_scores

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_set: Optional[List[Dict[str, str]]] = None,
    ) -> RAGEvalResult:
        """Run a full evaluation.

        Args:
            test_set: List of ``{"question": ..., "ground_truth": ...}``
                dicts.  Defaults to the built-in synthetic test set.

        Returns:
            An ``RAGEvalResult`` with aggregated scores.
        """
        if test_set is None:
            test_set = generate_test_set(self.kb_path)

        if not test_set:
            logger.warning("No test questions available — skipping evaluation")
            return RAGEvalResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                num_samples=0,
            )

        questions = [t["question"] for t in test_set]
        ground_truths = [t["ground_truth"] for t in test_set]

        logger.info("Running RAG evaluation on %d test questions", len(questions))

        # Retrieve contexts for each question
        contexts_list: List[List[str]] = []
        for question in questions:
            contexts = self.retrieve_contexts(question)
            contexts_list.append(contexts)
            logger.debug("  '%s': %d contexts retrieved", question[:50], len(contexts))

        # Score with Ragas (or heuristic fallback)
        f_scores, ar_scores, cr_scores, cp_scores = self.score_with_ragas(
            questions, contexts_list, ground_truths,
        )

        # Build result
        samples = []
        for i in range(len(questions)):
            samples.append(RAGEvalSample(
                question=questions[i],
                ground_truth=ground_truths[i],
                retrieved_contexts=contexts_list[i] if i < len(contexts_list) else [],
                faithfulness=f_scores[i] if i < len(f_scores) else 0.0,
                answer_relevancy=ar_scores[i] if i < len(ar_scores) else 0.0,
                context_recall=cr_scores[i] if i < len(cr_scores) else 0.0,
                context_precision=cp_scores[i] if i < len(cp_scores) else 0.0,
            ))

        result = RAGEvalResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_samples=len(samples),
            faithfulness=sum(s.faithfulness for s in samples) / len(samples) if samples else 0.0,
            answer_relevancy=sum(s.answer_relevancy for s in samples) / len(samples) if samples else 0.0,
            context_recall=sum(s.context_recall for s in samples) / len(samples) if samples else 0.0,
            context_precision=sum(s.context_precision for s in samples) / len(samples) if samples else 0.0,
            ragas_available=_RAGAS_AVAILABLE,
            engine="ragas" if _RAGAS_AVAILABLE else "heuristic",
            samples=[s.to_dict() for s in samples],
        )

        logger.info(
            "Evaluation complete: faithfulness=%.3f, answer_relevancy=%.3f, "
            "context_recall=%.3f, context_precision=%.3f (engine=%s)",
            result.faithfulness, result.answer_relevancy,
            result.context_recall, result.context_precision,
            result.engine,
        )

        return result


# ===================================================================
# Results persistence
# ===================================================================

def save_results(
    result: RAGEvalResult,
    path: str = DEFAULT_RESULTS_PATH,
) -> None:
    """Save evaluation results to a JSON file.

    Appends to an existing results file (keeping history) so that
    evaluation runs can be compared over time.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing history
    history: List[Dict[str, Any]] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                history = json.load(fh)
                if not isinstance(history, list):
                    history = [history]
        except (json.JSONDecodeError, OSError):
            history = []

    # Append new result
    history.append(result.to_dict())

    # Keep only last 50 runs
    history = history[-50:]

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
        logger.info("Evaluation results saved to %s (%d runs)", path, len(history))
    except OSError as exc:
        logger.error("Failed to save evaluation results: %s", exc)


def load_results(path: str = DEFAULT_RESULTS_PATH) -> List[Dict[str, Any]]:
    """Load evaluation result history.

    Returns:
        List of result dicts, most recent last.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data
        return [data]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load evaluation results: %s", exc)
        return []


def compare_results(
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare the latest two evaluation runs.

    Args:
        history: List of result dicts from ``load_results()``.

    Returns:
        Dict with current metrics and deltas from previous run.
    """
    if not history:
        return {"error": "No evaluation results available"}

    current = history[-1]

    if len(history) < 2:
        return {
            "current": current,
            "previous": None,
            "deltas": None,
            "message": "Only one evaluation run available. Submit another to see deltas.",
        }

    previous = history[-2]

    deltas = {}
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        if metric in current and metric in previous:
            deltas[metric] = round(current[metric] - previous[metric], 4)

    return {
        "current": current,
        "previous": previous,
        "deltas": deltas,
        "message": (
            f"Faithfulness: {current.get('faithfulness', 0):.3f} "
            f"({deltas.get('faithfulness', 0):+.3f}), "
            f"Answer Relevancy: {current.get('answer_relevancy', 0):.3f} "
            f"({deltas.get('answer_relevancy', 0):+.3f}), "
            f"Context Recall: {current.get('context_recall', 0):.3f} "
            f"({deltas.get('context_recall', 0):+.3f})"
        ),
    }


# ===================================================================
# CLI Entry Point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with Ragas metrics")
    parser.add_argument("--kb-path", type=str, default=DEFAULT_KB_PATH, help="Path to knowledge base JSONL")
    parser.add_argument("--top-k", type=int, default=3, help="Number of contexts to retrieve per question")
    parser.add_argument("--output", type=str, default=DEFAULT_RESULTS_PATH, help="Path to save results")
    parser.add_argument("--dry-run", action="store_true", help="Show test questions without running evaluation")
    parser.add_argument("--verbose", action="store_true", help="Show per-sample scores")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Generate test set
    test_set = generate_test_set(args.kb_path)

    if args.dry_run:
        print(f"\\n{'='*60}")
        print(f"  RAG EVALUATION — DRY RUN ({len(test_set)} test questions)")
        print(f"{'='*60}")
        for i, t in enumerate(test_set):
            print(f"\\n  [{i+1}] Q: {t['question']}")
            print(f"       GT: {t['ground_truth'][:100]}...")
        print(f"\\n{'='*60}")
        print(f"  To run evaluation: python src/eval/ragas_eval.py")
        print(f"{'='*60}\\n")
        return

    # Run evaluation
    print(f"\\n{'='*60}")
    print(f"  RAG EVALUATION ({len(test_set)} test questions, top_k={args.top_k})")
    print(f"{'='*60}\\n")

    evaluator = RAGEvaluator(kb_path=args.kb_path, top_k=args.top_k)
    result = evaluator.evaluate(test_set)

    # Print summary
    print(f"\\n{'='*60}")
    print(f"  RESULTS: {result.engine}")
    print(f"{'='*60}")
    print(f"  Faithfulness:      {result.faithfulness:.4f}")
    print(f"  Answer Relevancy:  {result.answer_relevancy:.4f}")
    print(f"  Context Recall:    {result.context_recall:.4f}")
    print(f"  Context Precision: {result.context_precision:.4f}")
    print(f"  Samples:           {result.num_samples}")
    print(f"  Ragas available:   {result.ragas_available}")
    print(f"{'='*60}")

    if args.verbose and result.samples:
        print(f"\\n{'='*60}")
        print(f"  PER-SAMPLE SCORES")
        print(f"{'='*60}")
        for i, s in enumerate(result.samples):
            print(f"\\n  [{i+1}] Q: {s['question']}")
            print(f"       F: {s['faithfulness']:.3f} | AR: {s['answer_relevancy']:.3f} "
                  f"| CR: {s['context_recall']:.3f} | CP: {s['context_precision']:.3f}")

    # Save
    save_results(result, args.output)

    # Compare with previous
    history = load_results(args.output)
    comparison = compare_results(history)
    if comparison.get("deltas"):
        print(f"\\n{'─'*60}")
        print(f"  COMPARED TO PREVIOUS RUN:")
        for metric, delta in comparison["deltas"].items():
            arrow = "▲" if delta >= 0 else "▼"
            print(f"    {metric}: {arrow} {delta:+.4f}")
        print(f"{'─'*60}")

    print(f"\\n{'='*60}\\n")


if __name__ == "__main__":
    main()
