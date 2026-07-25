"""
Celery Worker — DPO Training Pipeline

Processes negative feedback in the background, builds DPO preference pairs,
and triggers fine-tuning runs.  Uses Celery with Redis as the broker when
available, falling back to direct synchronous execution when Celery is not
installed (useful for development / single-server deployments).

Architecture::

    POST /api/feedback_log
           │
           ▼
    ┌─────────────────────────────┐
    │  trigger_dpo_pipeline.delay │  (Celery task or direct call)
    │  ─ OR ─                    │
    │  trigger_dpo_pipeline()    │  (sync fallback)
    └──────────┬──────────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  DPO Pipeline               │
    │  1. Read new feedback       │
    │  2. Build preference pairs  │
    │  3. Save to JSONL           │
    │  4. Trigger training if ≥50 │
    └─────────────────────────────┘

Usage::

    # Start Celery worker (with Redis):
    celery -A src.mcp_server.dpo_worker worker --loglevel=info

    # Trigger from code:
    from src.mcp_server.dpo_worker import trigger_dpo_pipeline
    trigger_dpo_pipeline()  # sync fallback if Celery not available
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: Celery
# ---------------------------------------------------------------------------
try:
    from celery import Celery

    _CELERY_AVAILABLE = True
    _celery_app = Celery(
        "mcp_dpo",
        broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1"),
        backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    )
    _celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
    )
except ImportError:
    _CELERY_AVAILABLE = False
    _celery_app = None  # type: ignore


# ===================================================================
# DPO Pipeline Task
# ===================================================================

def trigger_dpo_pipeline(
    feedback_path: Optional[str] = None,
    pairs_path: Optional[str] = None,
    min_pairs: int = 50,
    cooldown_hours: int = 24,
) -> Dict[str, Any]:
    """Run the DPO pipeline: process feedback → build pairs → maybe train.

    This is the core function.  When Celery is available it's exposed as
    a Celery task; otherwise it runs synchronously.

    Args:
        feedback_path: Path to feedback JSONL (default: data/feedback/).
        pairs_path: Path to save pairs (default: data/training/).
        min_pairs: Minimum pairs before triggering training.
        cooldown_hours: Minimum hours between training runs.

    Returns:
        Dict summary of what was done.
    """
    from src.mcp_server.dpo_pipeline import run_pipeline

    result = run_pipeline(
        feedback_path=feedback_path,
        pairs_path=pairs_path,
        min_pairs=min_pairs,
        cooldown_hours=cooldown_hours,
    )
    logger.info("DPO pipeline result: %s", result.get("message", str(result)))
    return result


def get_dpo_status() -> Dict[str, Any]:
    """Return DPO pipeline status for health endpoint / dashboard."""
    from src.mcp_server.dpo_pipeline import pipeline_status

    return pipeline_status()


# ===================================================================
# Celery task (when Celery is installed)
# ===================================================================

if _CELERY_AVAILABLE and _celery_app is not None:

    @_celery_app.task(
        bind=True,
        max_retries=3,
        default_retry_delay=60,
        name="dpo.trigger_pipeline",
    )
    def trigger_dpo_pipeline_task(
        self,
        feedback_path: Optional[str] = None,
        pairs_path: Optional[str] = None,
        min_pairs: int = 50,
        cooldown_hours: int = 24,
    ) -> Dict[str, Any]:
        """Celery task wrapper around ``trigger_dpo_pipeline()``."""
        try:
            return trigger_dpo_pipeline(
                feedback_path=feedback_path,
                pairs_path=pairs_path,
                min_pairs=min_pairs,
                cooldown_hours=cooldown_hours,
            )
        except Exception as exc:
            logger.error("DPO pipeline task failed: %s", exc)
            raise self.retry(exc=exc)

    def schedule_dpo_pipeline() -> None:
        """Schedule the DPO pipeline to run now (non-blocking).

        When Celery is available, sends a task to the broker.
        Otherwise runs synchronously.
        """
        logger.info("Scheduling DPO pipeline via Celery task")
        trigger_dpo_pipeline_task.delay()
else:
    # No Celery — we'll use asyncio for async execution when available,
    # or run synchronously as a fallback.

    def schedule_dpo_pipeline() -> None:
        """Schedule the DPO pipeline.

        Without Celery, runs synchronously in-process.
        This is fine for development / low-traffic deployments.
        """
        logger.info("Running DPO pipeline synchronously (no Celery)")
        try:
            result = trigger_dpo_pipeline()
            logger.info("DPO pipeline completed: %s", result.get("message", ""))
        except Exception as exc:
            logger.error("DPO pipeline failed: %s", exc)


# ===================================================================
# Convenience: schedule from the REST API
# ===================================================================

def try_schedule_pipeline() -> Dict[str, Any]:
    """Try to schedule the DPO pipeline.

    Returns a status dict indicating whether it was scheduled or not.
    This is safe to call from the REST API endpoint handlers — it never
    blocks for a significant amount of time.
    """
    try:
        if _CELERY_AVAILABLE:
            trigger_dpo_pipeline_task.delay()
            return {
                "scheduled": True,
                "method": "celery",
                "message": "DPO pipeline scheduled via Celery task",
            }
        else:
            # Run synchronously — will block but we don't want to block
            # the API response, so we tell the caller it's running.
            import threading

            thread = threading.Thread(target=trigger_dpo_pipeline, daemon=True)
            thread.start()
            return {
                "scheduled": True,
                "method": "thread",
                "message": "DPO pipeline started in background thread",
            }
    except Exception as exc:
        logger.error("Failed to schedule DPO pipeline: %s", exc)
        return {
            "scheduled": False,
            "method": "none",
            "message": f"Failed: {exc}",
        }
