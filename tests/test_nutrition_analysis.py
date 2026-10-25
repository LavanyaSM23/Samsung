"""
tests/test_nutrition_analysis.py
Comprehensive Integration & Unit Tests for Phase 7: Nutrition Analysis Module

Verifies:
1. Food Database Lookup (Master KB, Subtypes, Unified 1,556 items)
2. Portion Conversion & Defined Servings (grams, piece, katori, plate, bowl, cup)
3. Standard Formula Scaling:
     nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g
4. Complete Nutrient Spectrum (Calories, Macros, Micros, Vitamins & Minerals)
5. Missing Data Integrity (Missing fields remain None; ZERO is NEVER substituted)
6. Data Quality Flags & Source Transparency
7. Manual Food Search Fallback
8. Error Handling for Invalid Inputs (negative, zero, NaN, non-numeric, unknown units)
9. Web API Endpoints: /api/nutrition/search, /api/nutrition/servings, /api/nutrition/analyze
10. Explicit Non-Claim Disclaimer (no exact portion/calories from vision alone)
"""

import math
from pathlib import Path
import sys
import unittest

WORKSPACE_ROOT = Path("c:/Samsung_Project")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app import app
from src.nutrition_analyzer import (
    NutritionAnalyzer,
    get_analyzer,
    lookup_food,
    convert_portion_to_grams,
    calculate_nutrients,
    search_foods,
    get_available_servings,
    NUTRIENT_UNITS,
    PHASE7_DISCLAIMER,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
test_results = []


def record(name: str, condition: bool, extra: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if extra:
        msg += f" | {extra}"
    print(msg)
    test_results.append((name, condition))


def run_tests():
    print("=" * 65)
    print("RUNNING PHASE 7: NUTRITION ANALYSIS MODULE TESTS")
    print("=" * 65)

    analyzer = get_analyzer()

    # =========================================================================
    # 1. Food Database Lookup & Subtype Connection
    # =========================================================================
    print("\n--- 1. Food Lookup & Culinary Subtype Resolution ---")
    samosa = lookup_food("samosa")
    record("lookup: finds primary vision class 'samosa'", samosa is not None and samosa["class_name"] == "samosa")
    record("lookup: samosa has per_100g_nutrients baseline", "per_100g_nutrients" in samosa and samosa["per_100g_nutrients"]["calories_kcal"] > 0)
    record("lookup: samosa reference serving is 65g", samosa["reference_serving_g"] == 65.0)

    # Subtype lookup
    samosa_keema = lookup_food("samosa", subtype_id="samosa_keema")
    record("lookup: resolves specific subtype 'samosa_keema'", samosa_keema is not None and samosa_keema.get("selected_subtype") == "samosa_keema")
    record("lookup: keema samosa calories differ from default potato", samosa_keema["reference_nutrients"]["calories_kcal"] == 318.0)

    # Composite recipe lookup
    aloo_gobi = lookup_food("aloo_gobi")
    record("lookup: finds composite recipe 'aloo_gobi'", aloo_gobi is not None and aloo_gobi["match_type"] == "COMPOSITE_RECIPE")

    # Unified lookup fallback
    chai = lookup_food("hot tea (garam chai)")
    record("lookup: finds item from unified 1,556 dataset", chai is not None and "tea" in chai["display_name"].lower())

    # Unknown food returns None
    unknown = lookup_food("non_existent_dish_xyz_999")
    record("lookup: non-existent dish returns None", unknown is None)

    # =========================================================================
    # 2. Portion Conversion & Defined Serving Units
    # =========================================================================
    print("\n--- 2. Portion Conversion & Defined Serving Units ---")
    record("convert: direct grams 150g -> 150.0g", convert_portion_to_grams(150, "g") == 150.0)
    record("convert: string grams '175.5' -> 175.5g", convert_portion_to_grams("175.5", "g") == 175.5)
    record("convert: baseline '100g' unit -> 100.0g", convert_portion_to_grams(1.0, "100g") == 100.0)

    # Defined servings
    g_samosa_1pc = convert_portion_to_grams(1, "piece", food_name="samosa")
    record("convert: 1 piece of samosa = 65.0g", g_samosa_1pc == 65.0, f"g={g_samosa_1pc}")

    g_samosa_2pc = convert_portion_to_grams(1, "piece_2", food_name="samosa")
    record("convert: 2 pieces of samosa = 130.0g", g_samosa_2pc == 130.0, f"g={g_samosa_2pc}")

    g_samosa_half = convert_portion_to_grams(1, "piece_half", food_name="samosa")
    record("convert: 0.5 piece of samosa = 32.5g", g_samosa_half == 32.5, f"g={g_samosa_half}")

    g_biryani_plate = convert_portion_to_grams(1, "plate", food_name="biryani")
    record("convert: 1 plate of biryani = 250.0g", g_biryani_plate == 250.0, f"g={g_biryani_plate}")

    g_dal_katori = convert_portion_to_grams(1, "katori", food_name="dal")
    record("convert: 1 katori of dal = 150.0g", g_dal_katori == 150.0, f"g={g_dal_katori}")

    g_idli_pc = convert_portion_to_grams(1, "piece", food_name="idli")
    record("convert: 1 piece of idli = 80.0g", g_idli_pc == 80.0, f"g={g_idli_pc}")

    g_chapati_pc = convert_portion_to_grams(1, "piece", food_name="chapati")
    record("convert: 1 chapati = 35.0g", g_chapati_pc == 35.0, f"g={g_chapati_pc}")

    # Available servings list
    servings = get_available_servings("samosa")
    record("servings: get_available_servings returns >= 3 options", len(servings) >= 3)
    record("servings: contains piece option for samosa", any("piece" in s["unit"] for s in servings))

    # =========================================================================
    # 3. Invalid Inputs & Graceful Error Handling
    # =========================================================================
    print("\n--- 3. Validation & Invalid Input Error Handling ---")
    neg_caught = False
    try:
        convert_portion_to_grams(-50, "g")
    except ValueError:
        neg_caught = True
    record("error_handling: rejects negative portion value (-50g)", neg_caught)

    zero_caught = False
    try:
        convert_portion_to_grams(0, "g")
    except ValueError:
        zero_caught = True
    record("error_handling: rejects zero portion value", zero_caught)

    nan_caught = False
    try:
        convert_portion_to_grams(float("nan"), "g")
    except ValueError:
        nan_caught = True
    record("error_handling: rejects NaN portion value", nan_caught)

    str_caught = False
    try:
        convert_portion_to_grams("not_a_number", "g")
    except ValueError:
        str_caught = True
    record("error_handling: rejects non-numeric string portion", str_caught)

    unit_caught = False
    try:
        convert_portion_to_grams(100, "unsupported_unit_xyz")
    except ValueError:
        unit_caught = True
    record("error_handling: rejects unsupported portion unit", unit_caught)

    calc_unknown_caught = False
    try:
        calculate_nutrients("imaginary_food_unknown", 100.0)
    except KeyError:
        calc_unknown_caught = True
    record("error_handling: calculate_nutrients raises KeyError for unknown food", calc_unknown_caught)

    # =========================================================================
    # 4. Standard Formula Nutrient Scaling: nutrient = portion / 100 * per_100g
    # =========================================================================
    print("\n--- 4. Standard Nutrient Scaling Formula Execution ---")
    # For samosa: calculate at 100g, 200g, and 65g
    calc_100 = calculate_nutrients("samosa", 100.0)
    calc_200 = calculate_nutrients("samosa", 200.0)
    calc_65 = calculate_nutrients("samosa", 65.0)

    per_100_cal = samosa["per_100g_nutrients"]["calories_kcal"]
    val_100_cal = calc_100["nutrients"]["calories_kcal"]["value"]
    record("formula: 100g portion matches per_100g baseline", abs(val_100_cal - per_100_cal) < 0.1, f"calc={val_100_cal}, base={per_100_cal}")

    val_200_cal = calc_200["nutrients"]["calories_kcal"]["value"]
    record("formula: 200g portion is exactly 2.0x 100g baseline", abs(val_200_cal - 2.0 * per_100_cal) < 0.2, f"val={val_200_cal}")

    val_65_cal = calc_65["nutrients"]["calories_kcal"]["value"]
    expected_65 = round(0.65 * per_100_cal, 2)
    record("formula: 65g portion follows portion/100 * per_100g", abs(val_65_cal - expected_65) < 0.1, f"calc={val_65_cal}, expected={expected_65}")

    # Macros checked
    record("nutrients: protein calculated with unit 'g'", calc_100["nutrients"]["protein_g"]["unit"] == "g" and calc_100["nutrients"]["protein_g"]["value"] > 0)
    record("nutrients: carbs calculated with unit 'g'", calc_100["nutrients"]["carbs_g"]["unit"] == "g" and calc_100["nutrients"]["carbs_g"]["value"] > 0)
    record("nutrients: fat calculated with unit 'g'", calc_100["nutrients"]["fat_g"]["unit"] == "g" and calc_100["nutrients"]["fat_g"]["value"] > 0)
    record("nutrients: fibre calculated with unit 'g'", calc_100["nutrients"]["fibre_g"]["unit"] == "g" and calc_100["nutrients"]["fibre_g"]["value"] > 0)
    record("nutrients: sodium calculated with unit 'mg'", calc_100["nutrients"]["sodium_mg"]["unit"] == "mg")

    # =========================================================================
    # 5. Missing Nutrition Data Transparency (Never Zero-Filled)
    # =========================================================================
    print("\n--- 5. Missing Nutrition Data Integrity ---")
    # Synthetic item with missing micronutrients
    partial_food = {
        "food_id": "test_partial",
        "class_name": "test_partial",
        "display_name": "Test Partial Item",
        "serving_size_g": 100.0,
        "serving_unit": "100g",
        "calories_kcal": 200.0,
        "protein_g": 5.0,
        "carbs_g": 30.0,
        "fat_g": 4.0,
        "fibre_g": 2.0,
        "sodium_mg": 150.0,
        "calcium_mg": None,  # Missing
        "iron_mg": None,     # Missing
        "vitamin_c_mg": None,# Missing
        "folate_ug": None,   # Missing
        "data_source": "TEST_SOURCE"
    }

    calc_partial = calculate_nutrients(partial_food, 150.0)
    record("missing_data: missing calcium remains None, NOT 0", calc_partial["nutrients"]["calcium_mg"]["value"] is None)
    record("missing_data: missing iron remains None, NOT 0", calc_partial["nutrients"]["iron_mg"]["value"] is None)
    record("missing_data: status is DATA_NOT_AVAILABLE", calc_partial["nutrients"]["calcium_mg"]["status"] == "DATA_NOT_AVAILABLE")
    record("missing_data: reported in missing_nutrients list", any(m["nutrient"] == "calcium_mg" for m in calc_partial["missing_nutrients"]))
    record("missing_data: missing nutrients has descriptive reason", "TEST_SOURCE" in calc_partial["missing_nutrients"][0]["reason"])

    # =========================================================================
    # 6. Source Transparency & Data Quality Flags
    # =========================================================================
    print("\n--- 6. Source Transparency & Non-Claim Disclaimer ---")
    record("metadata: source is reported", bool(calc_100.get("source")))
    record("metadata: reference basis is reported", bool(calc_100.get("reference_basis")))
    record("metadata: quality flags present (sodium_flag, calorie_flag)", "sodium_flag" in calc_100.get("data_quality_flags", {}))
    record("disclaimer: disclaimer is non-empty", len(calc_100.get("disclaimer", "")) > 20)
    record("disclaimer: explicitly states portion weight must be confirmed by user", "portion weight must be confirmed" in calc_100["disclaimer"].lower())
    record("disclaimer: explicitly states category only from vision", "category only" in calc_100["disclaimer"].lower())

    # =========================================================================
    # 7. Manual Food Search Fallback
    # =========================================================================
    print("\n--- 7. Manual Food Search Fallback ---")
    search_samosa = search_foods("samosa")
    record("search: 'samosa' returns matches", len(search_samosa) > 0)
    record("search: top match is samosa", "samosa" in search_samosa[0]["food_name"].lower())

    search_chai = search_foods("chai")
    record("search: 'chai' returns matches across unified lookup", len(search_chai) > 0)

    search_roti = search_foods("roti")
    record("search: 'roti' returns matches", len(search_roti) > 0)

    search_empty = search_foods("")
    record("search: empty query returns empty list", len(search_empty) == 0)

    # =========================================================================
    # 8. Web API Endpoints: /api/nutrition/*
    # =========================================================================
    print("\n--- 8. Web API Endpoints ---")
    app.config["TESTING"] = True
    client = app.test_client()

    # GET /api/nutrition/search
    resp_s = client.get("/api/nutrition/search?q=biryani")
    record("api_search: returns 200 OK", resp_s.status_code == 200)
    s_json = resp_s.get_json()
    record("api_search: success is True", s_json.get("success") is True)
    record("api_search: returns matches", s_json.get("count", 0) > 0)

    # GET /api/nutrition/servings
    resp_serv = client.get("/api/nutrition/servings?food_name=samosa")
    record("api_servings: returns 200 OK", resp_serv.status_code == 200)
    serv_json = resp_serv.get_json()
    record("api_servings: contains servings list", len(serv_json.get("servings", [])) >= 2)

    # POST /api/nutrition/analyze (defined serving: 2 pieces of samosa)
    resp_an = client.post(
        "/api/nutrition/analyze",
        json={
            "food_name": "samosa",
            "portion_value": 2,
            "unit": "piece",
            "user_profile": "adult_male_sedentary"
        }
    )
    record("api_analyze: defined serving 2 pieces returns 200 OK", resp_an.status_code == 200)
    an_json = resp_an.get_json()
    record("api_analyze: success is True", an_json.get("success") is True)
    an_data = an_json.get("data", {})
    record("api_analyze: calculated portion is 130g", an_data.get("portion", {}).get("grams") == 130.0)
    record("api_analyze: calories calculated for 130g", an_data.get("nutrients", {}).get("calories_kcal", {}).get("value") > 0)
    record("api_analyze: health_evaluation present with suitability_score", "suitability_score" in an_data.get("health_evaluation", {}))

    # POST /api/nutrition/analyze errors
    resp_err1 = client.post("/api/nutrition/analyze", json={})
    record("api_analyze: missing food_name returns 400", resp_err1.status_code == 400)

    resp_err2 = client.post("/api/nutrition/analyze", json={"food_name": "samosa", "portion_value": -10})
    record("api_analyze: negative portion returns 400", resp_err2.status_code == 400)

    resp_err3 = client.post("/api/nutrition/analyze", json={"food_name": "non_existent_food_xyz"})
    record("api_analyze: unknown food returns 404", resp_err3.status_code == 404)

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 65)
    print(f"PHASE 7 NUTRITION ANALYSIS TESTS: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 65)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} test(s) failed in test_nutrition_analysis.py")


if __name__ == "__main__":
    run_tests()
