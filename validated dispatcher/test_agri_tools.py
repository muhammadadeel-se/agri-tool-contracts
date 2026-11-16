"""Unit tests for agri_tools functions and JSON Schema contracts."""

import jsonschema
import pytest

from agri_tools import (
    OUTPUT_SCHEMAS,
    TOOL_SCHEMAS,
    calculate_fertilizer_recommendation,
    execute_tool_call,
    get_simulated_weather,
)


# ---------------------------------------------------------------------------
# Schema Metadata Tests
# ---------------------------------------------------------------------------


def test_tool_schemas_are_valid_draft7_schemas():
    assert len(TOOL_SCHEMAS) == 2
    for tool_def in TOOL_SCHEMAS:
        assert tool_def["type"] == "function"
        fn = tool_def["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        jsonschema.Draft7Validator.check_schema(fn["parameters"])


# ---------------------------------------------------------------------------
# Weather Tool Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zip_code,units", [("90210", "imperial"), ("10001", "metric"), ("00501", "imperial"), ("99950", "metric")])
def test_get_simulated_weather_valid_and_matches_output_schema(zip_code, units):
    result = execute_tool_call("get_simulated_weather", {"zip_code": zip_code, "units": units})
    jsonschema.validate(instance=result, schema=OUTPUT_SCHEMAS["get_simulated_weather"])
    assert result["zip_code"] == zip_code
    assert result["units"] == units
    assert len(result["forecast_3day"]) == 3


def test_get_simulated_weather_is_deterministic():
    first = get_simulated_weather("60601", units="imperial")
    second = get_simulated_weather("60601", units="imperial")
    assert first == second


def test_get_simulated_weather_metric_conversion_consistency():
    imp = get_simulated_weather("80202", units="imperial")
    met = get_simulated_weather("80202", units="metric")
    expected_c = round((imp["current_conditions"]["temperature"] - 32.0) * 5.0 / 9.0, 1)
    assert met["current_conditions"]["temperature"] == expected_c


@pytest.mark.parametrize(
    "bad_payload",
    [
        {},                                      # Missing required zip_code
        {"zip_code": "1234"},                    # Too short
        {"zip_code": "123456"},                  # Too long
        {"zip_code": "ABCDE"},                   # Non-numeric
        {"zip_code": 90210},                     # Integer instead of string
        {"zip_code": "90210", "units": "kelvin"},# Invalid enum value
        {"zip_code": "90210", "extra": True},    # Unexpected property
    ],
)
def test_get_simulated_weather_rejects_bad_inputs(bad_payload):
    with pytest.raises(jsonschema.ValidationError):
        execute_tool_call("get_simulated_weather", bad_payload)


# ---------------------------------------------------------------------------
# Fertilizer Tool Tests
# ---------------------------------------------------------------------------


def test_calculate_fertilizer_corn_deficit_and_acreage_scaling():
    payload = {
        "crop_type": "corn",
        "soil_nitrogen_ppm": 20.0,    # Target 60 -> 40 ppm deficit -> 80 lbs/acre
        "soil_phosphorus_ppm": 10.0,  # Target 30 -> 20 ppm deficit -> 40 lbs/acre
        "soil_potassium_ppm": 100.0,  # Target 150 -> 50 ppm deficit -> 100 lbs/acre
        "soil_ph": 6.4,
        "field_acres": 10.0,
    }
    result = execute_tool_call("calculate_fertilizer_recommendation", payload)
    jsonschema.validate(
        instance=result,
        schema=OUTPUT_SCHEMAS["calculate_fertilizer_recommendation"],
    )

    assert result["nutrient_deficits_ppm"] == {"N": 40.0, "P": 20.0, "K": 50.0}
    assert result["recommended_lbs_per_acre"] == {
        "nitrogen_n": 80.0,
        "phosphorus_p2o5": 40.0,
        "potassium_k2o": 100.0,
    }
    assert result["total_field_lbs"] == {
        "nitrogen_n": 800.0,
        "phosphorus_p2o5": 400.0,
        "potassium_k2o": 1000.0,
    }


def test_calculate_fertilizer_zero_deficit_when_nutrients_exceed_targets():
    result = calculate_fertilizer_recommendation(
        crop_type="soybean",
        soil_nitrogen_ppm=50.0,
        soil_phosphorus_ppm=60.0,
        soil_potassium_ppm=220.0,
    )
    jsonschema.validate(
        instance=result,
        schema=OUTPUT_SCHEMAS["calculate_fertilizer_recommendation"],
    )
    assert result["nutrient_deficits_ppm"] == {"N": 0.0, "P": 0.0, "K": 0.0}
    assert result["recommended_lbs_per_acre"] == {
        "nitrogen_n": 0.0,
        "phosphorus_p2o5": 0.0,
        "potassium_k2o": 0.0,
    }
    assert any("no fertilizer required" in note.lower() for note in result["agronomic_notes"])


@pytest.mark.parametrize(
    "soil_ph,expected_keyword",
    [
        (4.8, "lime"),
        (8.1, "sulfur"),
    ],
)
def test_calculate_fertilizer_ph_advisories(soil_ph, expected_keyword):
    result = calculate_fertilizer_recommendation(
        crop_type="wheat",
        soil_nitrogen_ppm=30.0,
        soil_phosphorus_ppm=20.0,
        soil_potassium_ppm=100.0,
        soil_ph=soil_ph,
    )
    assert any(expected_keyword in note.lower() for note in result["agronomic_notes"])


@pytest.mark.parametrize(
    "bad_payload",
    [
        # Missing required parameters
        {"crop_type": "corn", "soil_nitrogen_ppm": 20, "soil_phosphorus_ppm": 10},
        {"soil_nitrogen_ppm": 20, "soil_phosphorus_ppm": 10, "soil_potassium_ppm": 100},
        # Unsupported crop
        {"crop_type": "avocado", "soil_nitrogen_ppm": 20, "soil_phosphorus_ppm": 10, "soil_potassium_ppm": 100},
        # Negative nutrient value
        {"crop_type": "corn", "soil_nitrogen_ppm": -5, "soil_phosphorus_ppm": 10, "soil_potassium_ppm": 100},
        # Out-of-range pH
        {"crop_type": "corn", "soil_nitrogen_ppm": 20, "soil_phosphorus_ppm": 10, "soil_potassium_ppm": 100, "soil_ph": 14.5},
        # Zero or negative acreage
        {"crop_type": "corn", "soil_nitrogen_ppm": 20, "soil_phosphorus_ppm": 10, "soil_potassium_ppm": 100, "field_acres": 0},
        # Unexpected extra property
        {"crop_type": "corn", "soil_nitrogen_ppm": 20, "soil_phosphorus_ppm": 10, "soil_potassium_ppm": 100, "organic": True},
    ],
)
def test_calculate_fertilizer_rejects_bad_inputs(bad_payload):
    with pytest.raises(jsonschema.ValidationError):
        execute_tool_call("calculate_fertilizer_recommendation", bad_payload)