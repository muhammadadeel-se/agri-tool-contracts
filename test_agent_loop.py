"""Unit tests for AgriAdvisorAgent loop, history management, and fallbacks."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_loop import AgriAdvisorAgent, SYSTEM_PROMPT


def _make_completion(content: str | None = None, tool_calls: list | None = None):
    """Build a lightweight mock mirroring OpenAI's ChatCompletion object."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _make_tool_call(call_id: str, name: str, arguments: str):
    """Build a mock tool call entry."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_single_turn_weather_tool_invocation():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_completion(
            tool_calls=[
                _make_tool_call("call_1", "get_simulated_weather", '{"zip_code": "90210"}')
            ]
        ),
        _make_completion(
            content="In ZIP 90210 (West Coast), it is currently Partly Cloudy with a soil moisture index of 0.39."
        ),
    ]

    agent = AgriAdvisorAgent(client=mock_client, model="gpt-4o-mini")
    reply = agent.run_turn("Check the weather in 90210.")

    assert "90210" in reply
    assert mock_client.chat.completions.create.call_count == 2

    # Verify history structure: system -> user -> assistant(tool_calls) -> tool -> assistant(final)
    roles = [msg["role"] for msg in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_payload = json.loads(agent.messages[3]["content"])
    assert tool_payload["status"] == "success"
    assert tool_payload["zip_code"] == "90210"


def test_multi_turn_history_preserves_weather_and_fertilizer_context():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        # Turn 1: Weather tool call -> answer
        _make_completion(
            tool_calls=[
                _make_tool_call("call_w1", "get_simulated_weather", '{"zip_code": "50014"}')
            ]
        ),
        _make_completion(content="Weather in 50014 is 56.8°F with 0.45 soil moisture."),
        # Turn 2: Fertilizer tool call -> answer
        _make_completion(
            tool_calls=[
                _make_tool_call(
                    "call_f1",
                    "calculate_fertilizer_recommendation",
                    json.dumps(
                        {
                            "crop_type": "corn",
                            "soil_nitrogen_ppm": 20,
                            "soil_phosphorus_ppm": 15,
                            "soil_potassium_ppm": 100,
                            "field_acres": 10,
                        }
                    ),
                )
            ]
        ),
        _make_completion(content="For 10 acres of corn, apply 800 lbs N, 300 lbs P2O5, and 1000 lbs K2O."),
        # Turn 3: Follow-up answered directly from retained conversation context
        _make_completion(content="With 0.45 soil moisture in 50014, field conditions are suitable for application."),
    ]

    agent = AgriAdvisorAgent(client=mock_client)
    r1 = agent.run_turn("How is the weather in 50014?")
    r2 = agent.run_turn("Calculate corn fertilizer for 10 acres: N=20, P=15, K=100.")
    r3 = agent.run_turn("Can I apply that fertilizer today given the moisture?")

    assert "50014" in r1
    assert "800 lbs N" in r2
    assert "0.45" in r3
    assert agent.messages[0]["content"] == SYSTEM_PROMPT
    assert sum(1 for m in agent.messages if m["role"] == "user") == 3


def test_fallback_handles_invalid_json_arguments_and_recovers():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        # Step 1: Model emits broken JSON
        _make_completion(
            tool_calls=[
                _make_tool_call("call_bad_json", "get_simulated_weather", '{"zip_code": 90210')
            ]
        ),
        # Step 2: Model sees invalid_json_arguments error and retries with valid JSON
        _make_completion(
            tool_calls=[
                _make_tool_call("call_fixed", "get_simulated_weather", '{"zip_code": "90210"}')
            ]
        ),
        # Step 3: Final grounded response
        _make_completion(content="Recovered and fetched weather for 90210."),
    ]

    agent = AgriAdvisorAgent(client=mock_client)
    reply = agent.run_turn("Weather for 90210?")

    assert "90210" in reply
    first_tool_error = json.loads(agent.messages[3]["content"])
    assert first_tool_error["status"] == "error"
    assert first_tool_error["error_type"] == "invalid_json_arguments"


def test_fallback_handles_schema_validation_error_on_bad_inputs():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        # Model hallucinates an unsupported crop and negative nitrogen
        _make_completion(
            tool_calls=[
                _make_tool_call(
                    "call_bad_schema",
                    "calculate_fertilizer_recommendation",
                    json.dumps(
                        {
                            "crop_type": "mango",
                            "soil_nitrogen_ppm": -10,
                            "soil_phosphorus_ppm": 20,
                            "soil_potassium_ppm": 100,
                        }
                    ),
                )
            ]
        ),
        # Model reads schema_validation_error and asks the farmer for valid parameters
        _make_completion(
            content="Mango is not currently supported, and nitrogen cannot be negative. Please provide a valid crop (corn, wheat, soybean, tomato, potato) and non-negative ppm values."
        ),
    ]

    agent = AgriAdvisorAgent(client=mock_client)
    reply = agent.run_turn("Fertilize my mango trees with -10 ppm nitrogen.")

    tool_error = json.loads(agent.messages[3]["content"])
    assert tool_error["status"] == "error"
    assert tool_error["error_type"] == "schema_validation_error"
    assert "not currently supported" in reply


def test_fallback_handles_unknown_tool_name():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _make_completion(
            tool_calls=[
                _make_tool_call("call_unknown", "get_pest_forecast", '{"zip_code": "90210"}')
            ]
        ),
        _make_completion(content="I only have access to local weather and fertilizer calculation tools."),
    ]

    agent = AgriAdvisorAgent(client=mock_client)
    reply = agent.run_turn("Are there locusts in 90210?")

    tool_error = json.loads(agent.messages[3]["content"])
    assert tool_error["error_type"] == "unknown_tool"
    assert "weather and fertilizer" in reply


def test_max_iterations_guard_prevents_infinite_tool_loop():
    mock_client = MagicMock()
    # Model keeps calling tools endlessly on every step
    mock_client.chat.completions.create.return_value = _make_completion(
        tool_calls=[
            _make_tool_call("call_loop", "get_simulated_weather", '{"zip_code": "90210"}')
        ]
    )

    agent = AgriAdvisorAgent(client=mock_client, max_iterations=2)
    reply = agent.run_turn("Loop forever")

    assert "maximum tool-calling steps" in reply
    assert mock_client.chat.completions.create.call_count == 2


def test_rejects_empty_user_message():
    agent = AgriAdvisorAgent(client=MagicMock())
    with pytest.raises(ValueError):
        agent.run_turn("   ")