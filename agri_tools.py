"""Weather and Soil Tool Definitions and Contracts."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import jsonschema

ZIP_CODE_PATTERN = re.compile(r"^[0-9]{5}$")

_REGIONAL_BASELINES: dict[str, tuple[str, float, int, float, float]] = {
    "0": ("Northeast", 58.0, 68, 0.18, 10.5),
    "1": ("Mid-Atlantic", 62.0, 65, 0.14, 9.2),
    "2": ("Southeast Atlantic", 72.0, 74, 0.22, 8.0),
    "3": ("Deep South", 78.0, 78, 0.25, 7.5),
    "4": ("Midwest Great Lakes", 60.0, 66, 0.15, 11.0),
    "5": ("Upper Midwest Plains", 56.0, 62, 0.12, 12.5),
    "6": ("Central Plains", 64.0, 60, 0.10, 13.0),
    "7": ("South Central", 76.0, 64, 0.11, 10.0),
    "8": ("Mountain West", 61.0, 38, 0.04, 9.5),
    "9": ("West Coast", 69.0, 52, 0.05, 8.5),
}

_CONDITIONS = ("Clear", "Partly Cloudy", "Overcast", "Light Rain", "Breezy")

_CROP_PROFILES: dict[str, dict[str, Any]] = {
    "corn": {"targets_ppm": {"N": 60.0, "P": 30.0, "K": 150.0}, "optimal_ph": (6.0, 6.8)},
    "wheat": {"targets_ppm": {"N": 45.0, "P": 25.0, "K": 120.0}, "optimal_ph": (6.0, 7.0)},
    "soybean": {"targets_ppm": {"N": 15.0, "P": 25.0, "K": 140.0}, "optimal_ph": (6.2, 7.0)},
    "tomato": {"targets_ppm": {"N": 50.0, "P": 40.0, "K": 180.0}, "optimal_ph": (6.0, 6.8)},
    "potato": {"targets_ppm": {"N": 55.0, "P": 35.0, "K": 170.0}, "optimal_ph": (5.2, 6.0)},
}

_PPM_TO_LBS_PER_ACRE = 2.0


def _convert_temp(temp_f: float, units: str) -> float:
    return round((temp_f - 32.0) * 5.0 / 9.0, 1) if units == "metric" else round(temp_f, 1)


def _convert_precip(precip_in: float, units: str) -> float:
    return round(precip_in * 25.4, 1) if units == "metric" else round(precip_in, 2)


def _convert_wind(wind_mph: float, units: str) -> float:
    return round(wind_mph * 1.60934, 1) if units == "metric" else round(wind_mph, 1)


def get_simulated_weather(zip_code: str, units: str = "imperial") -> dict[str, Any]:
    """Retrieve deterministic simulated local weather conditions for a US ZIP code."""
    if not isinstance(zip_code, str) or not ZIP_CODE_PATTERN.fullmatch(zip_code):
        raise ValueError(f"Invalid zip_code '{zip_code}': must be a 5-digit string.")
    if units not in ("imperial", "metric"):
        raise ValueError(f"Invalid units '{units}': must be 'imperial' or 'metric'.")

    region, base_temp_f, base_humidity, base_precip_in, base_wind_mph = _REGIONAL_BASELINES[zip_code[0]]
    digest = hashlib.sha256(zip_code.encode("utf-8")).digest()

    temp_f = base_temp_f + (((digest[0] / 255.0) * 10.0) - 5.0)
    humidity = max(5, min(100, base_humidity + int(((digest[1] / 255.0) * 20.0) - 10.0)))
    condition = _CONDITIONS[digest[4] % len(_CONDITIONS)]
    precip_in = 0.0 if condition in ("Clear", "Partly Cloudy") else base_precip_in * (0.5 + (digest[2] / 255.0))
    wind_mph = max(0.0, base_wind_mph + (((digest[3] / 255.0) * 6.0) - 3.0))
    soil_moisture = round(max(0.05, min(0.95, (humidity / 140.0) + (precip_in * 0.8))), 2)

    forecast_3day = []
    for day_idx in range(1, 4):
        day_delta = ((digest[4 + day_idx] / 255.0) * 8.0) - 4.0
        high_f = temp_f + 4.0 + day_delta
        low_f = temp_f - 12.0 + (day_delta * 0.5)
        precip_chance = max(0, min(100, int((humidity - 30) + ((digest[7 + day_idx] / 255.0) * 30.0))))
        forecast_3day.append(
            {
                "day": day_idx,
                "high": _convert_temp(high_f, units),
                "low": _convert_temp(low_f, units),
                "precip_chance_pct": precip_chance,
            }
        )

    return {
        "status": "success",
        "zip_code": zip_code,
        "region": region,
        "units": units,
        "current_conditions": {
            "temperature": _convert_temp(temp_f, units),
            "humidity_pct": humidity,
            "precipitation": _convert_precip(precip_in, units),
            "wind_speed": _convert_wind(wind_mph, units),
            "condition": condition,
        },
        "soil_moisture_index": soil_moisture,
        "forecast_3day": forecast_3day,
    }


def calculate_fertilizer_recommendation(
    crop_type: str,
    soil_nitrogen_ppm: float,
    soil_phosphorus_ppm: float,
    soil_potassium_ppm: float,
    soil_ph: float = 6.5,
    field_acres: float = 1.0,
) -> dict[str, Any]:
    """Calculate N-P-K fertilizer application rates from soil test values."""
    if not isinstance(crop_type, str) or crop_type.strip().lower() not in _CROP_PROFILES:
        supported = ", ".join(sorted(_CROP_PROFILES.keys()))
        raise ValueError(f"Unsupported crop_type '{crop_type}'. Supported: {supported}.")

    for name, val in (
        ("soil_nitrogen_ppm", soil_nitrogen_ppm),
        ("soil_phosphorus_ppm", soil_phosphorus_ppm),
        ("soil_potassium_ppm", soil_potassium_ppm),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
            raise ValueError(f"{name} must be a non-negative number.")

    if isinstance(soil_ph, bool) or not isinstance(soil_ph, (int, float)) or not (0.0 <= soil_ph <= 14.0):
        raise ValueError("soil_ph must be between 0.0 and 14.0.")

    if isinstance(field_acres, bool) or not isinstance(field_acres, (int, float)) or field_acres <= 0:
        raise ValueError("field_acres must be greater than 0.")

    normalized_crop = crop_type.strip().lower()
    profile = _CROP_PROFILES[normalized_crop]
    targets = profile["targets_ppm"]
    min_ph, max_ph = profile["optimal_ph"]

    deficits_ppm = {
        "N": round(max(0.0, targets["N"] - float(soil_nitrogen_ppm)), 2),
        "P": round(max(0.0, targets["P"] - float(soil_phosphorus_ppm)), 2),
        "K": round(max(0.0, targets["K"] - float(soil_potassium_ppm)), 2),
    }
    lbs_per_acre = {k: round(v * _PPM_TO_LBS_PER_ACRE, 2) for k, v in deficits_ppm.items()}
    total_lbs = {k: round(v * float(field_acres), 2) for k, v in lbs_per_acre.items()}

    notes: list[str] = []
    if soil_ph < min_ph:
        notes.append(
            f"Soil pH ({soil_ph}) is below optimal range ({min_ph}-{max_ph}) for "
            f"{normalized_crop}; consider agricultural lime application."
        )
    elif soil_ph > max_ph:
        notes.append(
            f"Soil pH ({soil_ph}) exceeds optimal range ({min_ph}-{max_ph}) for "
            f"{normalized_crop}; consider elemental sulfur to lower pH."
        )
    else:
        notes.append(f"Soil pH ({soil_ph}) is within optimal range ({min_ph}-{max_ph}).")

    if all(v == 0.0 for v in deficits_ppm.values()):
        notes.append("Soil N-P-K levels meet or exceed crop sufficiency targets; no fertilizer required.")

    return {
        "status": "success",
        "crop_type": normalized_crop,
        "field_acres": round(float(field_acres), 2),
        "soil_ph": round(float(soil_ph), 2),
        "nutrient_deficits_ppm": deficits_ppm,
        "recommended_lbs_per_acre": {
            "nitrogen_n": lbs_per_acre["N"],
            "phosphorus_p2o5": lbs_per_acre["P"],
            "potassium_k2o": lbs_per_acre["K"],
        },
        "total_field_lbs": {
            "nitrogen_n": total_lbs["N"],
            "phosphorus_p2o5": total_lbs["P"],
            "potassium_k2o": total_lbs["K"],
        },
        "agronomic_notes": notes,
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_simulated_weather",
            "description": "Fetch deterministic simulated local weather conditions, soil moisture index, and a 3-day forecast for a 5-digit US ZIP code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zip_code": {"type": "string", "pattern": "^[0-9]{5}$", "description": "5-digit US ZIP code (e.g., '90210')."},
                    "units": {"type": "string", "enum": ["imperial", "metric"], "default": "imperial"},
                },
                "required": ["zip_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_fertilizer_recommendation",
            "description": "Calculate per-acre and total field N-P-K fertilizer application rates based on crop type, soil test values (ppm), soil pH, and acreage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_type": {"type": "string", "enum": ["corn", "wheat", "soybean", "tomato", "potato"]},
                    "soil_nitrogen_ppm": {"type": "number", "minimum": 0},
                    "soil_phosphorus_ppm": {"type": "number", "minimum": 0},
                    "soil_potassium_ppm": {"type": "number", "minimum": 0},
                    "soil_ph": {"type": "number", "minimum": 0.0, "maximum": 14.0, "default": 6.5},
                    "field_acres": {"type": "number", "exclusiveMinimum": 0, "default": 1.0},
                },
                "required": ["crop_type", "soil_nitrogen_ppm", "soil_phosphorus_ppm", "soil_potassium_ppm"],
                "additionalProperties": False,
            },
        },
    },
]

_TOOL_REGISTRY = {
    "get_simulated_weather": get_simulated_weather,
    "calculate_fertilizer_recommendation": calculate_fertilizer_recommendation,
}

_INPUT_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {
    item["function"]["name"]: item["function"]["parameters"] for item in TOOL_SCHEMAS
}


def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate input arguments, execute the tool, and validate its output contract."""
    if tool_name not in _TOOL_REGISTRY:
        raise KeyError(f"Unknown tool '{tool_name}'.")
    jsonschema.validate(instance=arguments, schema=_INPUT_SCHEMA_BY_NAME[tool_name])
    result = _TOOL_REGISTRY[tool_name](**arguments)
    if not isinstance(result, dict) or not result or result.get("status") != "success":
        raise ValueError(f"Tool '{tool_name}' returned an empty or malformed payload: {result}")
    return result