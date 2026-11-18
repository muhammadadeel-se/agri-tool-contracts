"""Safety Guardrails, Input Sanitization, Refusal Templates, and Security Logging.

Enforces agricultural domain boundaries before LLM invocation and validates
fertilizer dosage ceilings on tool outputs to prevent hazardous recommendations.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_INPUT_LENGTH = 1000

PERMITTED_TOPICS: tuple[str, ...] = (
    "Local simulated weather forecasts, humidity, wind, and soil moisture by 5-digit US ZIP code",
    "Soil N-P-K nutrient deficit analysis and standard fertilizer rate calculations",
    "Soil pH assessment and standard lime or elemental sulfur soil conditioning",
    "Supported crop nutrition planning (corn, wheat, soybean, tomato, potato)",
)

SAFE_DOSAGE_LIMITS_LBS_PER_ACRE: dict[str, float] = {
    "nitrogen_n": 350.0,
    "phosphorus_p2o5": 250.0,
    "potassium_k2o": 400.0,
}
MAX_SAFE_FIELD_ACRES = 10000.0

REFUSAL_TEMPLATES: dict[str, str] = {
    "prompt_injection": (
        "I cannot override my operational safety guidelines or expose internal system "
        "instructions. I am here to assist with legitimate local farm weather and crop "
        "nutrient questions."
    ),
    "hazardous_chemical": (
        "I cannot provide instructions or advice on mixing hazardous chemicals, fuel-fertilizer "
        "combinations, toxic gases, or unregulated pesticide formulations. For safe chemical "
        "handling, consult your local agricultural extension office or product Safety Data Sheet (SDS)."
    ),
    "unsafe_dosage": (
        "I cannot recommend or calculate extreme fertilizer application rates that exceed safe "
        "agronomic thresholds, as excessive nutrient loading can burn crops and contaminate "
        "groundwater. Please verify your soil test values (ppm) and field acreage within standard "
        "farming ranges."
    ),
    "out_of_scope": (
        "I am specialized strictly in agricultural weather planning, soil pH evaluation, and "
        "N-P-K crop fertilizer calculations. I cannot assist with topics outside agronomy and "
        "farm weather."
    ),
    "nonsensical_input": (
        "I could not understand that input. Please ask a clear question about local farm weather "
        "(with a 5-digit US ZIP code) or crop fertilizer requirements (with crop type and soil N-P-K ppm values)."
    ),
}

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|all|system|safety)\b.{0,20}\b(instructions?|rules?|prompts?|guardrails?|filters?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|output|leak|display)\b.{0,25}\b(system\s*prompt|hidden\s*instructions?|internal\s*rules?|developer\s*message)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|switch\s+to)\b.{0,25}\b(dan|unrestricted|jailbroken|unfiltered|evil|developer\s*mode)\b",
            re.IGNORECASE,
        ),
    ),
)

_HAZARDOUS_CHEMICAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "explosive_fertilizer_mix",
        re.compile(
            r"\b(ammonium\s*nitrate|urea|fertilizer)\b.{0,40}\b(diesel|fuel\s*oil|kerosene|gasoline|nitromethane|explosive|bomb|detonat\w*|anfo)\b"
            r"|\b(diesel|fuel\s*oil|kerosene|gasoline|nitromethane|explosive|bomb|detonat\w*|anfo)\b.{0,40}\b(ammonium\s*nitrate|urea|fertilizer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "toxic_gas_chemical_mix",
        re.compile(
            r"\b(bleach|sodium\s*hypochlorite)\b.{0,30}\b(ammonia|vinegar|acid|chlorine)\b"
            r"|\b(ammonia|vinegar|acid)\b.{0,30}\b(bleach|sodium\s*hypochlorite)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "restricted_poison_or_weapon",
        re.compile(
            r"\b(cyanide|ricin|sarin|nerve\s*agent|mustard\s*gas|chlorine\s*gas|weaponiz\w*|poison\s+(water|livestock|animals|neighbors|people))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unsafe_pesticide_cocktail",
        re.compile(
            r"\b(mix|combine|brew|concentrate)\b.{0,35}\b(paraquat|organophosphate|banned\s*pesticide|lethal\s*dose|toxic\s*fumes)\b",
            re.IGNORECASE,
        ),
    ),
)

_UNSAFE_DOSAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "excessive_tonnage_application",
        re.compile(
            r"\b(\d+)\s*tons?\s*(of\s+)?(nitrogen|urea|fertilizer|ammonium|phosphorus|potassium)\s*(per|\/|an?)\s*acre\b",
            re.IGNORECASE,
        ),
    ),
    (
        "intentional_crop_or_soil_harm",
        re.compile(
            r"\b(burn\s+the\s+crops?|overdose\s+(the\s+)?(soil|field|crop)|contaminate\s+groundwater|10x\s+recommended)\b",
            re.IGNORECASE,
        ),
    ),
)

_HIGH_POUNDS_PER_ACRE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\s*(?:of\s+)?(?:nitrogen|urea|fertilizer|phosphorus|potassium|n-p-k|npk)\s*(?:per|\/|an?)\s*acre\b",
    re.IGNORECASE,
)

_AGRI_DOMAIN_KEYWORDS: frozenset[str] = frozenset(
    {
        "weather",
        "rain",
        "forecast",
        "temperature",
        "humidity",
        "wind",
        "moisture",
        "soil",
        "fertilizer",
        "fertilize",
        "nitrogen",
        "phosphorus",
        "potassium",
        "npk",
        "ph",
        "lime",
        "sulfur",
        "crop",
        "crops",
        "corn",
        "wheat",
        "soybean",
        "tomato",
        "potato",
        "acre",
        "acres",
        "farm",
        "farming",
        "farmer",
        "field",
        "harvest",
        "plant",
        "planting",
        "irrigation",
        "nutrient",
        "nutrients",
        "yield",
        "zip",
    }
)

_FOLLOWUP_KEYWORDS: frozenset[str] = frozenset(
    {
        "today",
        "tomorrow",
        "apply",
        "application",
        "rate",
        "dosage",
        "amount",
        "metric",
        "imperial",
        "units",
    }
)


@dataclass(frozen=True)
class GuardrailDecision:
    """Structured outcome from evaluating a user prompt or tool payload."""

    allowed: bool
    category: str
    sanitized_text: str
    trigger: str | None = None
    refusal_message: str | None = None


class SecurityAuditLogger:
    """Records blocked prompts and guardrail triggers to a structured log file."""

    def __init__(self, log_path: str | Path = "guardrail_events.log") -> None:
        self.log_path = Path(log_path)
        self.logger = logging.getLogger(f"agri_guardrails.{self.log_path.resolve()}")
        self.logger.setLevel(logging.WARNING)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def log_block(self, decision: GuardrailDecision, stage: str = "input_filter") -> dict[str, Any]:
        """Persist a JSON-encoded security audit entry for a blocked event."""
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "category": decision.category,
            "trigger": decision.trigger,
            "sanitized_preview": decision.sanitized_text[:200],
        }
        self.logger.warning(json.dumps(event))
        return event


class SafetyGuardrail:
    """Sanitizes inputs, enforces agricultural boundaries, and blocks unsafe dosages."""

    def __init__(self, audit_logger: SecurityAuditLogger | None = None) -> None:
        self.audit_logger = audit_logger or SecurityAuditLogger()

    @staticmethod
    def sanitize_input(raw_text: Any) -> str:
        """Normalize Unicode, strip control characters, and collapse whitespace."""
        if not isinstance(raw_text, str):
            return ""
        normalized = unicodedata.normalize("NFKC", raw_text)
        no_control = _CONTROL_CHARS_RE.sub("", normalized)
        return " ".join(no_control.split()).strip()

    def _block(
        self,
        category: str,
        sanitized_text: str,
        trigger: str,
        stage: str = "input_filter",
    ) -> GuardrailDecision:
        decision = GuardrailDecision(
            allowed=False,
            category=category,
            sanitized_text=sanitized_text,
            trigger=trigger,
            refusal_message=REFUSAL_TEMPLATES[category],
        )
        self.audit_logger.log_block(decision, stage=stage)
        return decision

    def evaluate_input(self, raw_text: Any, has_prior_context: bool = False) -> GuardrailDecision:
        """Run sanitization and layered safety checks before the agent loop executes."""
        sanitized = self.sanitize_input(raw_text)

        if not sanitized:
            return self._block("nonsensical_input", "", "empty_after_sanitization")

        if len(sanitized) > MAX_INPUT_LENGTH:
            return self._block("prompt_injection", sanitized, f"exceeded_max_length:{len(sanitized)}")

        alpha_count = sum(1 for ch in sanitized if ch.isalpha())
        longest_token = max((len(tok) for tok in sanitized.split()), default=0)
        if alpha_count < 3 or longest_token > 30:
            return self._block("nonsensical_input", sanitized, "non_linguistic_or_gibberish")

        for rule_name, pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(sanitized):
                return self._block("prompt_injection", sanitized, f"injection:{rule_name}")

        for rule_name, pattern in _HAZARDOUS_CHEMICAL_PATTERNS:
            if pattern.search(sanitized):
                return self._block("hazardous_chemical", sanitized, f"chemical:{rule_name}")

        for rule_name, pattern in _UNSAFE_DOSAGE_PATTERNS:
            if pattern.search(sanitized):
                return self._block("unsafe_dosage", sanitized, f"dosage:{rule_name}")

        lbs_match = _HIGH_POUNDS_PER_ACRE_RE.search(sanitized)
        if lbs_match and float(lbs_match.group(1)) > SAFE_DOSAGE_LIMITS_LBS_PER_ACRE["nitrogen_n"]:
            return self._block(
                "unsafe_dosage",
                sanitized,
                f"dosage:exceeded_lbs_per_acre:{lbs_match.group(1)}",
            )

        tokens = set(re.findall(r"[a-z0-9]+", sanitized.lower()))
        in_domain = bool(tokens & _AGRI_DOMAIN_KEYWORDS)
        valid_followup = has_prior_context and bool(tokens & _FOLLOWUP_KEYWORDS)
        if not in_domain and not valid_followup:
            return self._block("out_of_scope", sanitized, "outside_agricultural_boundary")

        return GuardrailDecision(
            allowed=True,
            category="allowed",
            sanitized_text=sanitized,
        )

    def validate_fertilizer_dosage(
        self,
        recommended_lbs_per_acre: dict[str, float],
        field_acres: float,
    ) -> GuardrailDecision:
        """Post-calculation safety check blocking extreme nutrient rates or acreage."""
        summary = f"acres={field_acres}, rates={recommended_lbs_per_acre}"
        if field_acres > MAX_SAFE_FIELD_ACRES:
            return self._block(
                "unsafe_dosage",
                summary,
                f"field_acres_exceeded:{field_acres}",
                stage="output_guardrail",
            )

        for nutrient, ceiling in SAFE_DOSAGE_LIMITS_LBS_PER_ACRE.items():
            rate = float(recommended_lbs_per_acre.get(nutrient, 0.0))
            if rate > ceiling:
                return self._block(
                    "unsafe_dosage",
                    summary,
                    f"{nutrient}_exceeded:{rate}>{ceiling}",
                    stage="output_guardrail",
                )

        return GuardrailDecision(
            allowed=True,
            category="allowed",
            sanitized_text=summary,
        )