# Agricultural Safety Guardrails & Refusal Handling

A multi-layered safety guardrail system wrapping an agricultural tool-calling agent to sanitize inputs, enforce agronomy domain boundaries, block hazardous chemical or dosage queries, and log security triggers.

## Boundary Rules & Guardrail Layers

1. **Permitted Agricultural Scope (`PERMITTED_TOPICS`):**
   - Local simulated weather, wind, humidity, and soil moisture lookups (5-digit US ZIP code).
   - Standard N-P-K fertilizer recommendations for supported crops (`corn`, `wheat`, `soybean`, `tomato`, `potato`).
   - Soil pH evaluation and standard lime/sulfur amendments.
2. **Pre-LLM Sanitization & Classification (`SafetyGuardrail.evaluate_input`):**
   - **Sanitization:** Normalizes Unicode (`NFKC`), strips ASCII control characters, collapses whitespace, and enforces a 1,000-character length ceiling.
   - **Prompt Injection Defense:** Blocks system-prompt extraction, rule-override attempts, and jailbreak personas before reaching the LLM.
   - **Hazardous Chemical & Dosage Filtering:** Blocks fuel-fertilizer explosive mixtures (e.g., ammonium nitrate + diesel), toxic gas combinations (bleach + ammonia), restricted poisons, and extreme fertilizer overdoses.
   - **Domain Boundary & Nonsensical Input Gate:** Politely declines non-agricultural requests (finance, coding, human medicine, politics) and non-linguistic gibberish without wasting LLM tokens.
3. **Post-Tool Dosage Guardrail (`SafetyGuardrail.validate_fertilizer_dosage`):**
   - Enforces upper agronomic ceilings (`350 lbs/acre N`, `250 lbs/acre P2O5`, `400 lbs/acre K2O`, and `10,000 acres`) on computed tool outputs before they reach the farmer.
4. **Structured Security Audit Logging (`SecurityAuditLogger`):**
   - Appends JSON-formatted events (`timestamp_utc`, `stage`, `category`, `trigger`, `sanitized_preview`) to `guardrail_events.log` whenever an input or tool output is blocked.

## Setup & Running Tests

```bash
pip install -r requirements.txt
pytest -v
```