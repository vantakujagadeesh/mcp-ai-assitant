"""
MCP Server — Support Assistant

Contains the ``SupportAssistant`` class that loads a fine-tuned (or base)
model and exposes tools for ticket classification, response generation,
and knowledge-base search.

Called by ``rest_api.py`` on startup.  Can also be run standalone with
``--transport sse`` for MCP-native clients.
"""

import argparse
import logging
import os
import sys
import json
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup for sibling imports
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = os.path.join(_project_root, "configs", "mcp_server.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load server configuration from a YAML (or JSON) file.

    Falls back to sensible defaults if the file is missing.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    config: Dict[str, Any] = {
        "transport": "rest",
        "host": "0.0.0.0",
        "port": 8000,
        "model": {
            "name": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "max_seq_length": 2048,
            "load_in_4bit": True,
            "lora_path": None,  # set to a local path or HF repo when fine-tuned
        },
        "rag": {
            "kb_path": None,  # defaults to data/knowledge_base/support_articles.jsonl
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": 5,
        },
    }

    if not os.path.exists(path):
        logger.info("Config file not found at %s — using defaults", path)
        return config

    try:
        import yaml

        with open(path, "r") as fh:
            loaded = yaml.safe_load(fh) or {}
        config.update(loaded)
        logger.info("Loaded config from %s", path)
    except ImportError:
        logger.warning("PyYAML not installed, trying JSON fallback for config")
        try:
            with open(path, "r") as fh:
                loaded = json.load(fh) or {}
            config.update(loaded)
        except Exception:
            logger.warning("Could not parse config — using defaults")
    except Exception as exc:
        logger.warning("Failed to load config (%s) — using defaults", exc)

    return config


# ---------------------------------------------------------------------------
# SupportAssistant — the reasoning core
# ---------------------------------------------------------------------------
class SupportAssistant:
    """Wraps a (potentially fine-tuned) LLM for customer support tasks.

    Usage inside REST API::

        cfg = load_config()
        assistant = SupportAssistant(cfg)
        result = assistant.handle_classify_ticket("My payment failed")
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Load the base model + optional LoRA adapters via Unsloth.

        Gracefully degrades to ``model = None`` if dependencies are missing
        (e.g. outside Colab).
        """
        model_cfg = self.config.get("model", {})
        model_name = model_cfg.get("name", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
        max_seq = model_cfg.get("max_seq_length", 2048)
        lora_path = model_cfg.get("lora_path")

        try:
            from unsloth import FastLanguageModel

            logger.info("Loading base model: %s", model_name)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq,
                dtype=None,           # auto-detect
                load_in_4bit=model_cfg.get("load_in_4bit", True),
            )

            if lora_path and os.path.exists(lora_path):
                logger.info("Loading LoRA adapters from %s", lora_path)
                from unsloth import FastLanguageModel as FLM

                self.model = FLM.from_pretrained(
                    self.model, lora_path
                )

            # Enable inference mode
            from unsloth import FastLanguageModel as FLM

            FLM.for_inference(self.model)
            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning(
                "unsloth not installed — model features disabled. "
                "Install with: uv pip install unsloth --torch-backend=auto --system"
            )
            self.model = None
            self.tokenizer = None
        except Exception as exc:
            logger.warning("Failed to load model (%s) — model features disabled", exc)
            self.model = None
            self.tokenizer = None

    # ------------------------------------------------------------------
    # Tool: classify ticket
    # ------------------------------------------------------------------
    def handle_classify_ticket(self, ticket_text: str) -> str:
        """Classify a support ticket into category + urgency.

        Returns the raw model-generated text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a support ticket classifier. "
                    "Given a ticket, output ONLY:\n"
                    "Category: <one of billing, account, technical, security, feature_request, general>\n"
                    "Urgency: <one of critical, high, medium, low>\n"
                    "Do not explain. Do not add extra text."
                ),
            },
            {"role": "user", "content": f"Ticket: {ticket_text}"},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True,
        )

        return self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Tool: generate response
    # ------------------------------------------------------------------
    def handle_generate_response(
        self,
        ticket_text: str,
        category: str = "",
        urgency: str = "",
    ) -> str:
        """Generate a support response for the given ticket.

        Returns the raw model-generated text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful customer support agent. "
                    "Write a professional, empathetic response to the customer's ticket. "
                    "Be concise but thorough. "
                    "Output ONLY your response, starting with 'Response:'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Category: {category}\n"
                    f"Urgency: {urgency}\n"
                    f"Ticket: {ticket_text}\n\n"
                    f"Write a support response:"
                ),
            },
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )

        return self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


# ===================================================================
# Standalone entry point (for MCP-native clients over SSE)
# ===================================================================

def run_sse_server(config: Dict[str, Any]) -> None:
    """Run the MCP server over SSE transport (not yet implemented)."""
    raise NotImplementedError(
        "SSE transport is a stub. Use --transport rest for the FastAPI REST server."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP AI Assistant Server")
    parser.add_argument("--transport", choices=["sse", "rest"], default="rest")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.transport == "rest":
        # Delegate to the FastAPI REST server
        from src.mcp_server.rest_api import create_app
        import uvicorn

        app = create_app()
        print(
            f"\n🚀 MCP AI Assistant starting on http://{args.host}:{args.port}"
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    else:
        run_sse_server(config)


if __name__ == "__main__":
    main()
