# Agricultural Agentic Loop & Tool Integration

A multi-turn LLM advisory agent that translates natural language farmer questions into schema-validated weather and soil nutrition tool calls, executes local Python functions, and synthesizes grounded responses.

## Architecture & Execution Loop

1. **System Prompting (`SYSTEM_PROMPT`):** Instructs the model to ground all metrics in tool outputs, ask clarifying questions when required inputs (5-digit ZIP code or N-P-K ppm values) are missing, and link context across turns.
2. **Dynamic Tool Routing (`AgriAdvisorAgent.run_turn`):** Sends conversation history and `TOOL_SCHEMAS` from `agri_tools.py` to an OpenAI-compatible chat completions endpoint (`tool_choice="auto"`).
3. **Contract-Validated Execution (`_safe_execute_tool`):** Parses model JSON payloads, enforces JSON Schema input rules via `execute_tool_call`, and appends structured `role: "tool"` messages back into the conversation history.
4. **Fallback & Loop Safety:**
   - Malformed JSON (`invalid_json_arguments`), schema violations (`schema_validation_error`), and unregistered tool names (`unknown_tool`) are caught and returned to the model as structured JSON error objects with recovery hints rather than crashing the process.
   - A `max_iterations` guard prevents infinite tool-calling loops, and `_trim_history` bounds context length without splitting assistant/tool message pairs.

## Setup & Testing

```bash
pip install -r requirements.txt
cp .env.example .env   # Add your OPENAI_API_KEY (or local OPENAI_BASE_URL)

# Run automated unit tests (uses mocked LLM responses; no API key required)
pytest -v

# Run scripted 3-turn weather + crop nutrition demo
python cli.py --demo

# Start an interactive multi-turn farmer session
python cli.py