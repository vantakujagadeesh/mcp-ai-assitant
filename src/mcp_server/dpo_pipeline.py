"""
DPO Preference Pair Builder

Reads feedback logs (``data/feedback/feedback_log.jsonl``), groups positive
and negative feedback by ticket context, and generates DPO preference triples
``(prompt, chosen, rejected)`` suitable for direct preference optimization.

Also tracks when enough pairs have accumulated and can trigger a fine-tuning
run via the existing ``src/finetune/train.py`` infrastructure.

Architecture::

    Feedback Log (JSONL)                     DPO Pipeline
    ┌─────────────────────┐      ┌──────────────────────────────┐
    │ ticket_id: "TKT-1"  │      │  1. Collect & Aggregate      │
    │ rating: "thumbs_up" │─────▶│     - Group by ticket_id     │
    │ ticket_text: "..."  │      │     - Separate +/− responses │
    │ response: "..."     │      │                              │
    ├─────────────────────┤      │  2. Build Preference Pairs   │
    │ ticket_id: "TKT-1"  │      │     - prompt = ticket_text   │
    │ rating: "thumbs_down"│────▶│     - chosen = positive resp │
    │ response: "... alt" │      │     - rejected = negative    │
    └─────────────────────┘      │                              │
                                 │  3. Trigger Training         │
                                 │     - When ≥ threshold pairs  │
                                 │     - Launch finetune task    │
                                 └──────────────────────────────┘
"""

import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_FEEDBACK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "feedback", "feedback_log.jsonl",
)

DEFAULT_PAIRS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "training", "dpo_pairs.jsonl",
)

DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "training", "dpo_state.json",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_PAIRS_FOR_TRAINING = 50        # minimum preference pairs to trigger training
MAX_PAIRS_PER_BATCH = 500          # cap per batch to avoid OOM
TRAINING_COOLDOWN_HOURS = 24       # minimum hours between training runs
MIN_TICKETS_FOR_PAIR = 1           # min feedback entries per ticket to form a pair

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """A single DPO preference triple."""

    prompt: str
    chosen: str          # preferred (thumbs-up) response
    rejected: str        # dispreferred (thumbs-down) response
    category: str = ""
    urgency: str = ""
    source_ticket: str = ""     # ticket_id this pair was derived from
    confidence: float = 0.9     # how confident we are this is a valid pair
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "category": self.category,
            "urgency": self.urgency,
            "source_ticket": self.source_ticket,
            "confidence": self.confidence,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class DPOPipelineState:
    """Persistent state tracking DPO pipeline progress."""

    total_feedback_processed: int = 0
    total_pairs_generated: int = 0
    total_training_runs: int = 0
    last_processed_index: int = 0  # line index in feedback_log.jsonl
    last_training_run_at: Optional[str] = None
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_feedback_processed": self.total_feedback_processed,
            "total_pairs_generated": self.total_pairs_generated,
            "total_training_runs": self.total_training_runs,
            "last_processed_index": self.last_processed_index,
            "last_training_run_at": self.last_training_run_at,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DPOPipelineState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ===================================================================
# Feedback Reader
# ===================================================================

def read_feedback_log(path: str = DEFAULT_FEEDBACK_PATH) -> List[Dict[str, Any]]:
    """Read all feedback entries from the JSONL log file.

    Returns:
        List of parsed feedback dicts, each containing at minimum
        ``ticket_id``, ``rating``, and (optionally) ``ticket_text``,
        ``response``, ``category``, ``urgency``, ``confidence``.
    """
    if not os.path.exists(path):
        logger.info("Feedback log not found at %s — no entries to process", path)
        return []

    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed feedback line: %s", exc)
    except OSError as exc:
        logger.error("Failed to read feedback log: %s", exc)
        return []

    return entries


def read_unprocessed_feedback(
    path: str = DEFAULT_FEEDBACK_PATH,
    last_index: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Read only new (unprocessed) feedback entries.

    Args:
        path: Path to the feedback JSONL file.
        last_index: Line index to start from (0-based).

    Returns:
        A tuple ``(new_entries, total_line_count)`` where ``new_entries``
        are the entries from ``last_index`` onwards.
    """
    if not os.path.exists(path):
        return [], 0

    entries: List[Dict[str, Any]] = []
    total_lines = 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                total_lines += 1
                if total_lines <= last_index:
                    continue
                if stripped:
                    try:
                        entries.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return [], total_lines

    return entries, total_lines


# ===================================================================
# Preference Pair Builder
# ===================================================================

class DPOPreferenceBuilder:
    """Build DPO preference pairs from feedback data.

    Groups feedback by ``ticket_id``, separates positive (``thumbs_up``)
    and negative (``thumbs_down``) responses, and pairs them.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        max_pairs: int = MAX_PAIRS_PER_BATCH,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_pairs = max_pairs

    def build_pairs(self, feedback_entries: List[Dict[str, Any]]) -> List[PreferencePair]:
        """Build preference pairs from feedback entries.

        Args:
            feedback_entries: List of parsed feedback dicts.

        Returns:
            List of ``PreferencePair`` objects.
        """
        # Group by ticket_id
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: {"positive": [], "negative": []}
        )

        for entry in feedback_entries:
            ticket_id = entry.get("ticket_id", "unknown")
            rating = entry.get("rating", "")
            confidence = entry.get("confidence", 0.0)

            if confidence < self.min_confidence:
                continue

            if rating == "thumbs_up":
                grouped[ticket_id]["positive"].append(entry)
            elif rating == "thumbs_down":
                grouped[ticket_id]["negative"].append(entry)
            # Ignore "unknown" or other ratings

        # Build pairs: for each ticket with both positive and negative feedback
        pairs: List[PreferencePair] = []

        for ticket_id, groups in grouped.items():
            positives = groups["positive"]
            negatives = groups["negative"]

            if not positives or not negatives:
                continue

            # Use the most confident positive as "chosen"
            best_positive = max(positives, key=lambda e: e.get("confidence", 0))
            # Use the most confident negative as "rejected"
            worst_negative = max(negatives, key=lambda e: e.get("confidence", 0))

            prompt = best_positive.get(
                "ticket_text", best_positive.get("input", "")
            ) or worst_negative.get("ticket_text", worst_negative.get("input", ""))

            # Build the prompt from the ticket text + context
            full_prompt = f"Ticket: {prompt}"
            category = best_positive.get("category", "") or worst_negative.get("category", "")
            urgency = best_positive.get("urgency", "") or worst_negative.get("urgency", "")

            # Chosen = positive response, Rejected = negative response
            chosen = best_positive.get("response", best_positive.get("output", ""))
            rejected = worst_negative.get("response", worst_negative.get("output", ""))

            if not prompt or not chosen or not rejected:
                continue

            pair = PreferencePair(
                prompt=full_prompt,
                chosen=chosen,
                rejected=rejected,
                category=category,
                urgency=urgency,
                source_ticket=ticket_id,
                confidence=min(
                    best_positive.get("confidence", 0.9),
                    worst_negative.get("confidence", 0.9),
                ),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            pairs.append(pair)

        # Limit and sort by confidence
        pairs.sort(key=lambda p: p.confidence, reverse=True)
        if len(pairs) > self.max_pairs:
            pairs = pairs[: self.max_pairs]

        return pairs


# ===================================================================
# Persistence
# ===================================================================

def save_pairs(pairs: List[PreferencePair], path: str = DEFAULT_PAIRS_PATH) -> int:
    """Save preference pairs to a JSONL file.

    Args:
        pairs: List of preference pairs to save.
        path: Output JSONL path.

    Returns:
        Number of pairs written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Append to existing
    existing_ids: Set[str] = set()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            existing_ids.add(json.loads(line).get("source_ticket", ""))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    count = 0
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for pair in pairs:
                if pair.source_ticket in existing_ids:
                    continue  # skip duplicates
                fh.write(json.dumps(pair.to_dict()) + "\n")
                existing_ids.add(pair.source_ticket)
                count += 1
        logger.info("Saved %d new DPO preference pairs to %s", count, path)
    except OSError as exc:
        logger.error("Failed to save DPO pairs: %s", exc)

    return count


def load_pairs(path: str = DEFAULT_PAIRS_PATH) -> List[Dict[str, Any]]:
    """Load all saved DPO preference pairs.

    Returns:
        List of pair dicts.
    """
    if not os.path.exists(path):
        return []

    pairs: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError as exc:
        logger.error("Failed to load DPO pairs: %s", exc)

    return pairs


def count_pairs(path: str = DEFAULT_PAIRS_PATH) -> int:
    """Count saved DPO preference pairs without loading all of them."""
    return len(load_pairs(path))


# ===================================================================
# State management
# ===================================================================

def load_state(path: str = DEFAULT_STATE_PATH) -> DPOPipelineState:
    """Load pipeline state from disk."""
    if not os.path.exists(path):
        return DPOPipelineState()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return DPOPipelineState.from_dict(data)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load DPO state (%s), starting fresh", exc)
        return DPOPipelineState()


def save_state(state: DPOPipelineState, path: str = DEFAULT_STATE_PATH) -> None:
    """Save pipeline state to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2)
    except OSError as exc:
        logger.error("Failed to save DPO state: %s", exc)


# ===================================================================
# Full pipeline run
# ===================================================================

def run_pipeline(
    feedback_path: str = DEFAULT_FEEDBACK_PATH,
    pairs_path: str = DEFAULT_PAIRS_PATH,
    state_path: str = DEFAULT_STATE_PATH,
    min_pairs: int = MIN_PAIRS_FOR_TRAINING,
    cooldown_hours: int = TRAINING_COOLDOWN_HOURS,
) -> Dict[str, Any]:
    """Run one cycle of the DPO pipeline.

    1. Read new (unprocessed) feedback entries.
    2. Build preference pairs.
    3. Save new pairs to disk.
    4. Check if we have enough pairs to trigger training.
    5. If training is needed and cooldown has elapsed, trigger it.

    Returns:
        A dict summary of what was done.
    """
    state = load_state(state_path)
    builder = DPOPreferenceBuilder()

    # Step 1: Read new feedback
    new_entries, total_lines = read_unprocessed_feedback(feedback_path, state.last_processed_index)
    if not new_entries:
        return {
            "action": "nothing_to_process",
            "new_entries": 0,
            "new_pairs": 0,
            "total_pairs": count_pairs(pairs_path),
            "message": "No new feedback entries to process",
        }

    logger.info("DPO pipeline: processing %d new feedback entries", len(new_entries))

    # Step 2: Build preference pairs
    pairs = builder.build_pairs(new_entries)
    logger.info("DPO pipeline: built %d preference pairs", len(pairs))

    # Step 3: Save new pairs
    saved = save_pairs(pairs, pairs_path)

    # Update state
    state.last_processed_index = total_lines
    state.total_feedback_processed += len(new_entries)
    state.total_pairs_generated += saved

    # Step 4: Check if we should trigger training
    total_pairs = count_pairs(pairs_path)
    should_train = total_pairs >= min_pairs

    training_triggered = False
    cooldown_ok = True

    if should_train:
        # Check cooldown
        if state.last_training_run_at:
            last_run = datetime.fromisoformat(state.last_training_run_at)
            elapsed_hours = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
            cooldown_ok = elapsed_hours >= cooldown_hours

        if cooldown_ok:
            # Step 5: Trigger training
            logger.info(
                "DPO pipeline: %d pairs >= %d threshold, triggering training",
                total_pairs, min_pairs,
            )
            try:
                result = _trigger_training(pairs_path)
                if result:
                    state.total_training_runs += 1
                    state.last_training_run_at = datetime.now(timezone.utc).isoformat()
                    training_triggered = True
                    logger.info("DPO pipeline: training triggered successfully")
                else:
                    logger.warning("DPO pipeline: training trigger returned no result")
            except Exception as exc:
                error_msg = f"Training trigger failed: {exc}"
                logger.error(error_msg)
                state.last_error = error_msg
        else:
            remaining_hours = cooldown_hours - elapsed_hours
            logger.info(
                "DPO pipeline: cooldown active (%d hours remaining)",
                int(remaining_hours),
            )

    # Save state
    save_state(state, state_path)

    return {
        "action": "training_triggered" if training_triggered else "pairs_saved",
        "new_entries": len(new_entries),
        "new_pairs": saved,
        "total_pairs": total_pairs,
        "training_triggered": training_triggered,
        "cooldown_ok": cooldown_ok,
        "total_training_runs": state.total_training_runs,
        "last_training_run_at": state.last_training_run_at,
        "message": (
            f"DPO training triggered with {total_pairs} pairs"
            if training_triggered
            else f"Saved {saved} new pairs ({total_pairs} total, need {min_pairs} for training)"
        ),
    }


def _trigger_training(pairs_path: str) -> Optional[Dict[str, Any]]:
    """Trigger a DPO fine-tuning run.

    This spawns a subprocess that runs the DPO training script.  In a
    production deployment this would be sent to a separate GPU worker.
    When Celery is available, this is wrapped in a Celery task instead.

    Returns:
        A dict with training job info, or ``None`` on failure.
    """
    import subprocess

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "finetune", "dpo_train.py",
    )

    if not os.path.exists(script_path):
        logger.warning(
            "DPO training script not found at %s — training will be a no-op",
            script_path,
        )
        return {
            "status": "skipped",
            "reason": f"Training script not found: {script_path}",
            "pairs_path": pairs_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    try:
        result = subprocess.run(
            [sys.executable, script_path, "--pairs-path", pairs_path],
            capture_output=True,
            text=True,
            timeout=30,  # just the launch, training takes much longer
        )
        return {
            "status": "launched",
            "returncode": result.returncode,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except FileNotFoundError:
        return {
            "status": "failed",
            "reason": "Python not found",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "launched",
            "reason": "Subprocess timed out, but may still be running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ===================================================================
# Pipeline status (for health endpoint / dashboard)
# ===================================================================

def pipeline_status(
    pairs_path: str = DEFAULT_PAIRS_PATH,
    state_path: str = DEFAULT_STATE_PATH,
) -> Dict[str, Any]:
    """Return current DPO pipeline status."""
    state = load_state(state_path)
    total_pairs = count_pairs(pairs_path)

    return {
        "enabled": True,
        "total_feedback_processed": state.total_feedback_processed,
        "total_pairs_generated": state.total_pairs_generated,
        "total_pairs_available": total_pairs,
        "total_training_runs": state.total_training_runs,
        "last_training_run_at": state.last_training_run_at,
        "min_pairs_for_training": MIN_PAIRS_FOR_TRAINING,
        "ready_for_training": total_pairs >= MIN_PAIRS_FOR_TRAINING,
        "last_error": state.last_error,
        "pairs_path": pairs_path,
        "feedback_path": DEFAULT_FEEDBACK_PATH,
    }
