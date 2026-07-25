"""
Guardrails — PII Masking & Safety / Injection Detection

Provides two layers of protection for user-input text before it reaches
the LLM or gets stored in logs / conversation history:

1. **PII Masking** (``PIISanitizer``)
   - Uses Microsoft Presidio when available (``pip install presidio-analyzer
     presidio-anonymizer``) for NER-based PII detection.
   - Falls back to regex-based detection when Presidio is not installed.
   - Detects & masks: email addresses, phone numbers, credit card numbers,
     SSNs, IP addresses, and API keys / secrets.

2. **Safety / Injection Guardrails** (``SafetyChecker``)
   - Uses NVIDIA NeMo Guardrails when available (``pip install
     nemoguardrails``) for semantic jailbreak / prompt injection detection.
   - Falls back to keyword + heuristic patterns.
   - Detects & blocks: prompt injection attempts, jailbreak attempts,
     harmful content, system prompt override attempts.

Both layers gracefully degrade — if a dependency is missing they log a
warning and use a lightweight built-in fallback.  This means the server
starts and runs without requiring ``presidio-analyzer`` or
``nemoguardrails`` to be installed.

Usage::

    from src.mcp_server.guardrails import PIISanitizer, SafetyChecker

    pii = PIISanitizer()
    safe = SafetyChecker()

    text = "My email is john@example.com and SSN is 123-45-6789"
    result = pii.sanitize(text)
    # result.sanitized_text -> "My email is [EMAIL] and SSN is [SSN]"
    # result.entities_found -> [{"type": "EMAIL", ...}, {"type": "SSN", ...}]

    check = safe.check_input("Tell me how to hack a server")
    # check.blocked -> True
    # check.reason -> "Prompt injection attempt detected"
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class GuardrailAction(Enum):
    """Action taken by a guardrail check."""

    PASS = "pass"           # No issues, text is safe
    SANITIZED = "sanitized"  # PII was masked
    BLOCKED = "blocked"     # Content was blocked
    FLAGGED = "flagged"     # Content was flagged but not blocked


@dataclass
class SanitizeResult:
    """Result of a PII sanitization operation."""

    sanitized_text: str
    entities_found: List[Dict[str, Any]] = field(default_factory=list)
    action: GuardrailAction = GuardrailAction.PASS


@dataclass
class SafetyCheckResult:
    """Result of a safety / injection check."""

    blocked: bool = False
    flagged: bool = False
    reason: str = ""
    sanitized_text: str = ""
    action: GuardrailAction = GuardrailAction.PASS


# ---------------------------------------------------------------------------
# PII regex patterns (fallback when Presidio is not available)
# ---------------------------------------------------------------------------

PII_PATTERNS: List[Tuple[str, str, str]] = [
    # (entity_type, regex_pattern, replacement_template)
    ("EMAIL_ADDRESS",
     r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
     "[EMAIL]"),
    ("PHONE_NUMBER",
     r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
     "[PHONE]"),
    ("CREDIT_CARD",
     r"\b(?:\d{4}[-\s]?){3}\d{4,7}\b",
     "[CREDIT_CARD]"),
    ("SSN",
     r"\b\d{3}-\d{2}-\d{4}\b",
     "[SSN]"),
    ("IP_ADDRESS",
     r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
     "[IP_ADDRESS]"),
    ("API_KEY",
     r"(?i)\b(sk-[a-zA-Z0-9]{20,}|[a-zA-Z0-9_-]{32,}|AKIA[0-9A-Z]{16})\b",
     "[API_KEY]"),
    ("US_PASSPORT",
     r"\b\d{9}\b",
     "[PASSPORT]"),
]

# ---------------------------------------------------------------------------
# Prompt injection / jailbreak patterns (fallback when NeMo is unavailable)
# ---------------------------------------------------------------------------

# Heuristic patterns that suggest prompt injection or jailbreak attempts
INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    # Direct system prompt override attempts
    (r"(?i)ignore\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?)", "system_override", 0.8),
    (r"(?i)forget\s+(?:all\s+)?(?:previous|above|prior)\s+(?:instructions?|prompts?)", "system_override", 0.8),
    (r"(?i)you\s+(?:are\s+)?(?:now\s+)?(?:a\s+)?(?:free|unbounded|unrestricted)", "jailbreak", 0.7),
    (r"(?i)act\s+as\s+(?:if\s+)?you\s+(?:are|were)\s+(?:a|an)\s+(?:different|new)", "jailbreak", 0.6),
    (r"(?i)new\s+(?:instructions?|prompts?|system prompt)", "system_override", 0.7),
    (r"(?i)you\s+(?:must|have\s+to|need\s+to)\s+(?:ignore|forget|disregard)", "jailbreak", 0.7),

    # Role-playing / DAN-style jailbreaks
    (r"(?i)do\s+anything\s+now", "jailbreak", 0.5),
    (r"(?i)no\s+(?:restrictions?|limits?|rules?|boundaries?)", "jailbreak", 0.6),
    (r"(?i)you\s+(?:can\s+)?(?:finally\s+)?(?:be\s+)?(?:free|unleashed)", "jailbreak", 0.7),
    (r"(?i)d.a.n|dan\s+mode|jailbreak\s+mode", "jailbreak", 0.9),

    # Attempts to extract system prompt
    (r"(?i)what\s+(?:is|are)\s+(?:your\s+)?(?:system\s+)?(?:prompts?|instructions?)", "prompt_extraction", 0.6),
    (r"(?i)print\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", "prompt_extraction", 0.7),
    (r"(?i)repeat\s+(?:the\s+)?(?:above|previous|initial)\s+(?:text|prompt|instructions?)", "prompt_extraction", 0.6),
    (r"(?i)show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)", "prompt_extraction", 0.6),

    # Harmful content
    (r"(?i)how\s+to\s+(?:hack|steal|cheat|scam|phish|crack)", "harmful", 0.7),
    (r"(?i)(?:bomb|explosive|weapon|terrorist)\s+(?:making?|instructions?|recipe)", "harmful", 0.9),
    (r"(?i)sql\s+injection|cross.site.scripting|buffer\s+overflow", "harmful", 0.6),
]


# ===================================================================
# PII Sanitizer
# ===================================================================

class PIISanitizer:
    """Sanitize PII from text using Presidio (or regex fallback).

    Usage::

        pii = PIISanitizer()
        result = pii.sanitize("My email is john@example.com")
        # result.sanitized_text -> "My email is [EMAIL]"
        # result.entities_found -> [{"type": "EMAIL_ADDRESS", ...}]
    """

    def __init__(
        self,
        enabled: bool = True,
        use_presidio: bool = True,
        mask_char: str = "*",
        show_first_n_chars: int = 0,
    ) -> None:
        self.enabled = enabled
        self.mask_char = mask_char
        self.show_first_n_chars = show_first_n_chars

        self._presidio_available = False
        self._analyzer = None
        self._anonymizer = None
        self._spacy_available = False

        if enabled and use_presidio:
            self._init_presidio()

        # If Presidio didn't load, compile regex patterns now
        if not self._presidio_available:
            self._regex_patterns = [
                (re.compile(pattern), entity_type, replacement)
                for entity_type, pattern, replacement in PII_PATTERNS
            ]

    # ------------------------------------------------------------------
    # Presidio initialisation
    # ------------------------------------------------------------------
    def _init_presidio(self) -> None:
        """Attempt to load Presidio analyzer + anonymizer."""
        try:
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()
            logger.info("Presidio AnalyzerEngine loaded")
        except ImportError:
            logger.info(
                "presidio-analyzer not installed — using regex PII fallback. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )
            return
        except Exception as exc:
            logger.warning("Failed to initialise Presidio analyzer: %s", exc)
            return

        try:
            from presidio_anonymizer import AnonymizerEngine

            self._anonymizer = AnonymizerEngine()
            logger.info("Presidio AnonymizerEngine loaded")
        except ImportError:
            logger.warning(
                "presidio-anonymizer not installed — can detect PII but cannot anonymize"
            )

        self._presidio_available = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sanitize(self, text: str) -> SanitizeResult:
        """Sanitize PII in the given text.

        Returns:
            A ``SanitizeResult`` with the sanitized text and a list of
            entities found (with type, start, end, confidence).
        """
        if not self.enabled or not text:
            return SanitizeResult(sanitized_text=text, entities_found=[])

        if self._presidio_available and self._analyzer is not None:
            return self._sanitize_presidio(text)

        return self._sanitize_regex(text)

    def mask(self, text: str) -> str:
        """Convenience: sanitize and return only the sanitized text."""
        return self.sanitize(text).sanitized_text

    # ------------------------------------------------------------------
    # Presidio-based sanitization
    # ------------------------------------------------------------------
    def _sanitize_presidio(self, text: str) -> SanitizeResult:
        """Use Presidio for NER-based PII detection and anonymization."""
        entities_found: List[Dict[str, Any]] = []

        try:
            # Analyze
            results = self._analyzer.analyze(
                text=text,
                language="en",
            )

            for r in results:
                entities_found.append({
                    "type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": round(r.score, 4),
                })

            # Anonymize
            if self._anonymizer is not None and results:
                anonymized = self._anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                )
                sanitized = anonymized.text
            else:
                # If only analyzer is available, manually mask
                sanitized = self._manual_mask_presidio_results(text, results)

        except Exception as exc:
            logger.warning("Presidio sanitization failed (%s) — falling back to regex", exc)
            return self._sanitize_regex(text)

        if entities_found:
            return SanitizeResult(
                sanitized_text=sanitized,
                entities_found=entities_found,
                action=GuardrailAction.SANITIZED,
            )
        return SanitizeResult(sanitized_text=text, entities_found=[])

    @staticmethod
    def _manual_mask_presidio_results(text: str, results: Any) -> str:
        """Manually mask PII spans from Presidio analyzer results."""
        # Sort results in reverse order so we don't invalidate indices
        sorted_results = sorted(results, key=lambda r: r.start, reverse=True)
        chars = list(text)
        for r in sorted_results:
            for i in range(r.start, r.end):
                if i < len(chars):
                    chars[i] = "*"
        return "".join(chars)

    # ------------------------------------------------------------------
    # Regex-based sanitization (fallback)
    # ------------------------------------------------------------------
    def _sanitize_regex(self, text: str) -> SanitizeResult:
        """Use regex patterns for PII detection and masking."""
        entities_found: List[Dict[str, Any]] = []
        sanitized = text

        for compiled, entity_type, replacement in self._regex_patterns:
            for match in compiled.finditer(text):
                entities_found.append({
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.85,  # regex patterns are fairly confident
                    "match": match.group(),
                })
            # Replace all occurrences
            sanitized = compiled.sub(replacement, sanitized)

        if entities_found:
            return SanitizeResult(
                sanitized_text=sanitized,
                entities_found=entities_found,
                action=GuardrailAction.SANITIZED,
            )
        return SanitizeResult(sanitized_text=text, entities_found=[])

    @property
    def is_presidio_loaded(self) -> bool:
        """Whether Presidio is available (vs. regex fallback)."""
        return self._presidio_available

    @property
    def status(self) -> Dict[str, Any]:
        """Return status information for health / dashboard."""
        return {
            "enabled": self.enabled,
            "engine": "presidio" if self._presidio_available else "regex",
            "presidio_loaded": self._presidio_available,
        }


# ===================================================================
# Safety / Injection Checker
# ===================================================================

class SafetyChecker:
    """Check input text for prompt injection, jailbreaks, and harmful content.

    Uses NeMo Guardrails when available, falling back to regex/heuristic
    pattern matching.

    Usage::

        safe = SafetyChecker()
        result = safe.check_input("Tell me how to hack a server")
        # result.blocked -> True
        # result.reason -> "harmful: harmful content detected"
    """

    def __init__(
        self,
        enabled: bool = True,
        use_nemo: bool = True,
        block_on_high_confidence: bool = True,
        flag_on_medium_confidence: bool = True,
        injection_threshold: float = 0.7,
    ) -> None:
        self.enabled = enabled
        self.block_on_high_confidence = block_on_high_confidence
        self.flag_on_medium_confidence = flag_on_medium_confidence
        self.injection_threshold = injection_threshold

        self._nemo_available = False
        self._nemo_rails = None

        if enabled and use_nemo:
            self._init_nemo()

        if not self._nemo_available:
            self._injection_patterns = [
                (re.compile(pattern), category, confidence)
                for pattern, category, confidence in INJECTION_PATTERNS
            ]

    # ------------------------------------------------------------------
    # NeMo Guardrails initialisation
    # ------------------------------------------------------------------
    def _init_nemo(self) -> None:
        """Attempt to load NeMo Guardrails."""
        try:
            from nemoguardrails import RailsConfig

            # Look for guardrails config directory, or use default inline config
            config_dir = self._find_guardrails_config()
            if config_dir:
                self._nemo_rails = self._create_nemo_rails(config_dir)
                self._nemo_available = True
                logger.info("NeMo Guardrails loaded from %s", config_dir)
            else:
                logger.info(
                    "NeMo Guardrails config not found — using heuristic fallback. "
                    "Place ./configs/guardrails/ with config.yml to enable."
                )
        except ImportError:
            logger.info(
                "nemoguardrails not installed — using heuristic injection detection. "
                "Install with: pip install nemoguardrails"
            )
        except Exception as exc:
            logger.warning("Failed to initialise NeMo Guardrails: %s", exc)

    @staticmethod
    def _find_guardrails_config() -> Optional[str]:
        """Locate the NeMo Guardrails configuration directory."""
        # First check ./configs/guardrails/
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "configs", "guardrails",
        )
        config_yml = os.path.join(local_path, "config.yml")
        if os.path.exists(config_yml):
            return local_path

        # Then check environment variable
        env_path = os.environ.get("NEMO_GUARDRAILS_CONFIG")
        if env_path and os.path.exists(os.path.join(env_path, "config.yml")):
            return env_path

        return None

    @staticmethod
    def _create_nemo_rails(config_dir: str) -> Any:
        """Create a NeMo Guardrails LLMRails instance."""
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_path(config_dir)
        return LLMRails(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_input(self, text: str) -> SafetyCheckResult:
        """Check user input for safety / injection issues.

        Args:
            text: The user's input text.

        Returns:
            A ``SafetyCheckResult`` indicating whether the input should
            be blocked, flagged, or allowed.
        """
        if not self.enabled or not text:
            return SafetyCheckResult()

        if self._nemo_available and self._nemo_rails is not None:
            return self._check_nemo(text)

        return self._check_heuristic(text)

    def check_output(self, text: str) -> SafetyCheckResult:
        """Check LLM output for safety issues.

        Args:
            text: The LLM-generated output text.

        Returns:
            A ``SafetyCheckResult`` indicating whether the output should
            be blocked or allowed.
        """
        if not self.enabled or not text:
            return SafetyCheckResult()

        # For now, re-use the same heuristic (output injection is rare)
        if self._nemo_available and self._nemo_rails is not None:
            return self._check_nemo_output(text)

        return self._check_output_heuristic(text)

    # ------------------------------------------------------------------
    # NeMo-based checking
    # ------------------------------------------------------------------
    def _check_nemo(self, text: str) -> SafetyCheckResult:
        """Use NeMo Guardrails for semantic safety checking."""
        try:
            response = self._nemo_rails.generate(
                messages=[{"role": "user", "content": text}]
            )
            # NeMo returns the response; if it's a canned refusal, it was blocked
            refusal_keywords = [
                "i'm sorry", "i cannot", "i can't", "i am unable",
                "i'm not able", "it would be inappropriate",
            ]
            lower = response.lower()
            for kw in refusal_keywords:
                if kw in lower:
                    return SafetyCheckResult(
                        blocked=True,
                        reason="Guided refusal: input was blocked by guardrails",
                        action=GuardrailAction.BLOCKED,
                    )
            return SafetyCheckResult(
                sanitized_text=response,
                action=GuardrailAction.PASS,
            )
        except Exception as exc:
            logger.warning("NeMo Guardrails check failed (%s) — falling back to heuristic", exc)

        return self._check_heuristic(text)

    def _check_nemo_output(self, text: str) -> SafetyCheckResult:
        """Use NeMo Guardrails for output safety checking."""
        # NeMo doesn't have a separate output-only check easily accessible
        # Use heuristic for output for now
        return self._check_output_heuristic(text)

    # ------------------------------------------------------------------
    # Heuristic-based checking (fallback)
    # ------------------------------------------------------------------
    def _check_heuristic(self, text: str) -> SafetyCheckResult:
        """Use regex/heuristic patterns for injection detection."""
        max_confidence = 0.0
        matched_categories: Dict[str, float] = {}
        matched_patterns: List[str] = []

        for compiled, category, confidence in self._injection_patterns:
            match = compiled.search(text)
            if match:
                matched_categories[category] = max(
                    matched_categories.get(category, 0),
                    confidence,
                )
                max_confidence = max(max_confidence, confidence)
                matched_patterns.append(match.group())

        if not matched_categories:
            return SafetyCheckResult()

        # Determine action based on confidence
        if max_confidence >= self.injection_threshold:
            # High confidence — block
            top_category = max(matched_categories, key=matched_categories.get)
            return SafetyCheckResult(
                blocked=True,
                flagged=True,
                reason=f"{top_category}: injection/jailbreak detected (confidence={max_confidence:.2f})",
                action=GuardrailAction.BLOCKED,
            )

        if self.flag_on_medium_confidence and max_confidence >= 0.5:
            # Medium confidence — flag but don't block
            top_category = max(matched_categories, key=matched_categories.get)
            return SafetyCheckResult(
                blocked=False,
                flagged=True,
                reason=f"{top_category}: suspicious input flagged (confidence={max_confidence:.2f})",
                action=GuardrailAction.FLAGGED,
            )

        return SafetyCheckResult()

    def _check_output_heuristic(self, text: str) -> SafetyCheckResult:
        """Check LLM output for harmful content."""
        # Simple output check — look for harmful disclosures
        harmful_output = [
            (r"(?i)here are the steps to (?:hack|steal|cheat)", "harmful_instructions"),
            (r"(?i)your (?:credit card|ssn|password) is", "pii_disclosure"),
            (r"(?i)i can help you (?:hack|steal|scam)", "harmful_assistance"),
        ]

        for compiled, category in [(re.compile(p), c) for p, c in harmful_output]:
            if compiled.search(text):
                return SafetyCheckResult(
                    blocked=True,
                    reason=f"{category}: harmful output detected",
                    action=GuardrailAction.BLOCKED,
                )

        return SafetyCheckResult()

    @property
    def is_nemo_loaded(self) -> bool:
        """Whether NeMo Guardrails is available (vs. heuristic fallback)."""
        return self._nemo_available

    @property
    def status(self) -> Dict[str, Any]:
        """Return status information for health / dashboard."""
        return {
            "enabled": self.enabled,
            "engine": "nemo" if self._nemo_available else "heuristic",
            "nemo_loaded": self._nemo_available,
            "threshold": self.injection_threshold,
            "block_on_high_confidence": self.block_on_high_confidence,
        }


# ===================================================================
# Convenience: create default instances
# ===================================================================

_default_pii: Optional[PIISanitizer] = None
_default_safety: Optional[SafetyChecker] = None


def get_pii_sanitizer(config: Optional[Dict[str, Any]] = None) -> PIISanitizer:
    """Get or create the default PII sanitizer instance.

    Args:
        config: Optional config dict with a ``guardrails.pii`` section.
    """
    global _default_pii

    if _default_pii is not None and config is None:
        return _default_pii

    pii_config = config.get("guardrails", {}).get("pii", {}) if config else {}
    _default_pii = PIISanitizer(
        enabled=pii_config.get("enabled", True),
        use_presidio=pii_config.get("use_presidio", True),
    )
    return _default_pii


def get_safety_checker(config: Optional[Dict[str, Any]] = None) -> SafetyChecker:
    """Get or create the default safety checker instance.

    Args:
        config: Optional config dict with a ``guardrails.safety`` section.
    """
    global _default_safety

    if _default_safety is not None and config is None:
        return _default_safety

    safety_config = config.get("guardrails", {}).get("safety", {}) if config else {}
    _default_safety = SafetyChecker(
        enabled=safety_config.get("enabled", True),
        use_nemo=safety_config.get("use_nemo", True),
        block_on_high_confidence=safety_config.get("block_on_high_confidence", True),
        injection_threshold=safety_config.get("injection_threshold", 0.7),
    )
    return _default_safety


def reset_guardrails() -> None:
    """Reset default guardrail instances (useful for testing)."""
    global _default_pii, _default_safety
    _default_pii = None
    _default_safety = None
