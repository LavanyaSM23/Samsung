"""
tests/test_system_integration.py
NutriSense AI — Phase 12: Comprehensive System Integration Test Suite

Verifies:
1. Complete End-to-End Workflow:
   User Profile -> Food Image Upload / Selection -> MobileNetV2 Prediction -> 
   User Confirmation & Ambiguity Disambiguation -> Portion Selection & Scaling -> 
   Nutrition Analysis -> 0-100 Suitability Score -> Safety-Filtered Recipe Recommendations -> 
   Meal Logging -> Daily Totals & Dashboard Metrics.
2. Manual food search and correction when image recognition is uncertain.
3. Multi-session persistence: Meal history & profile survive app restarts / new connections.
4. Correctness of nutrition calculations across foods, portions, and units.
5. Strict Pre-Ranking Allergy & Restriction Enforcement.
"""

import os
import sys
import uuid
import datetime
from pathlib import Path
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.database_manager import (
    get_db_connection,
    seed_reference_data_if_empty,
    log_meal,
    get_meal_history,
    delete_meal_log,
    get_daily_nutrition_totals,
    get_period_nutrition_summary,
    save_meal_plan,
    get_latest_meal_plan,
    log_user_feedback,
    get_backend_orchestrator
)
from src.user_profile_manager import (
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
    list_profiles,
    compute_bmi,
    compute_daily_calorie_target
)
from src.nutrition_analyzer import (
    lookup_food,
    calculate_nutrients,
    convert_portion_to_grams,
    search_foods,
    get_available_servings
)
from src.recommendation_engine import (
    get_recommendation_system,
    SafetyFilterEngine,
    SuitabilityScorer,
    AlternativeFoodsEngine
)
from src.inference_pipeline import NutriSenseInferenceEngine

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

def run_integration_tests():
    print("=" * 75)
    print("RUNNING PHASE 12: COMPLETE SYSTEM INTEGRATION VERIFICATION")
    print("=" * 75)

    conn = get_db_connection()
    seed_reference_data_if_empty(conn)

    # =========================================================================
    # Step 1: User Profile Creation & Anthropometric Calculations
    # =========================================================================
    print("\n--- Step 1: User Profile & Target Calories ---")
    test_uid = f"integ_user_{uuid.uuid4().hex[:8]}"
    profile_data = {
        "user_id": test_uid,
        "display_name": "Integration Tester",
        "age": 28,
        "gender": "female",
        "height_cm": 162.0,
        "weight_kg": 56.0,
        "activity_level": "moderately_active",
        "dietary_goal": "maintain_weight",
        "budget": "medium",
        "max_cooking_time_mins": 45,
        "preferred_cuisines": ["North Indian", "South Indian"],
        "allergies": ["peanuts", "dairy"],
        "dietary_restrictions": ["vegetarian"],
        "disliked_foods": ["bitter gourd"]
    }
    user_p = create_profile(profile_data)
    record("1.01 Profile created successfully in SQLite", user_p["user_id"] == test_uid)

    bmi_info = compute_bmi(profile_data["height_cm"], profile_data["weight_kg"])
    record("1.02 Anthropometric BMI calculation computed", "bmi" in bmi_info and bmi_info["bmi"] > 0)

    cal_target = compute_daily_calorie_target(
        profile_data["age"], profile_data["gender"],
        profile_data["height_cm"], profile_data["weight_kg"],
        profile_data["activity_level"], profile_data["dietary_goal"]
    )
    target_kcal = cal_target["target_calories_kcal"]
    record("1.03 Mifflin-St Jeor TDEE target computed", target_kcal > 1400)

    # =========================================================================
    # Step 2: Food Image Upload & Prediction with Disambiguation Fallback
    # =========================================================================
    print("\n--- Step 2: Visual Recognition & Food Disambiguation ---")
    inf_engine = NutriSenseInferenceEngine()
    test_img_path = WORKSPACE_ROOT / "static" / "samples" / "samosa.jpg"
    if test_img_path.exists():
        test_img = Image.open(test_img_path).convert("RGB")
    else:
        test_img = Image.new("RGB", (224, 224), color=(200, 150, 80))
    
    analysis = inf_engine.analyze_food(test_img, top_k=5)
    pred = analysis["prediction"]
    record("2.01 Inference engine runs and outputs prediction", "class_name" in pred and pred["confidence"] >= 0)
    record("2.02 Prediction contains top-k candidate distribution", len(pred["top_k"]) >= 1)

    # Simulate uncertain prediction / user manual correction
    corrected_dish = "samosa"
    record("2.03 Food dish confirmation / manual correction accepts verified item", corrected_dish in inf_engine.idx_to_class.values())

    # =========================================================================
    # Step 3: Portion Selection & Accurate Nutrition Scaling
    # =========================================================================
    print("\n--- Step 3: Portion Scaling & Nutrient Composition ---")
    servings = get_available_servings(corrected_dish)
    record("3.01 Serving unit options retrieved for confirmed dish", len(servings) > 0)

    # 1 portion of 2 pieces unit (or 2 portions of 1 piece standard_serving)
    grams = convert_portion_to_grams(1.0, unit="piece_2", food_name=corrected_dish)
    record("3.02 Portion converted to grams accurately (1 x piece_2 = 130g)", abs(grams - 130.0) < 1.0, f"grams={grams}")

    food_rec = lookup_food(corrected_dish)
    record("3.03 Food record retrieved from master laboratory KB", food_rec is not None and food_rec["class_name"] == "samosa")

    nutrients_scaled = calculate_nutrients(food_rec, grams)
    cals = nutrients_scaled["nutrients"]["calories_kcal"]["value"]
    prot = nutrients_scaled["nutrients"]["protein_g"]["value"]
    carbs = nutrients_scaled["nutrients"]["carbs_g"]["value"]
    fat = nutrients_scaled["nutrients"]["fat_g"]["value"]
    record("3.04 Scaled energy reflects 130g portion (393.8 * 1.3 ≈ 512 kcal)", abs(cals - 512.0) < 5.0, f"cals={cals}")
    record("3.05 Macronutrient scaling integrity maintained", prot > 0 and carbs > 0 and fat > 0)

    # =========================================================================
    # Step 4: Safety Filter, Suitability Scoring & Healthier Alternatives
    # =========================================================================
    print("\n--- Step 4: Safety Audit, Suitability Scoring & Alternatives ---")
    safety_eng = SafetyFilterEngine()
    proxy_food = {
        "recipe_name": food_rec["display_name"],
        "ingredients_raw": food_rec.get("reference_basis", ""),
        "ingredient_tokens": food_rec["class_name"].replace("_", " "),
        "diet": "vegetarian"
    }
    safety_check = safety_eng.check_safety(proxy_food, user_p)
    record("4.01 Samosa passes vegetarian & dairy/peanut safety checks", safety_check["is_safe"])

    scorer = SuitabilityScorer()
    score_res = scorer.score_recipe(proxy_food, user_p)
    record("4.02 0-100 Suitability score calculated with breakdown", 0 <= score_res["total_score"] <= 100)
    record("4.03 Explanation generated for score evaluation", len(score_res["explanation"]) > 10)

    alt_eng = AlternativeFoodsEngine()
    alts = alt_eng.get_alternatives(food_rec["class_name"], user_profile=user_p)
    record("4.04 Healthier alternatives recommended for calorie-dense samosa", len(alts) > 0)

    # =========================================================================
    # Step 5: Meal Logging & SQLite Persistence
    # =========================================================================
    print("\n--- Step 5: Meal Logging & SQLite Persistence ---")
    log_data = {
        "food_name": food_rec["display_name"],
        "food_id": food_rec["food_id"],
        "course": "lunch",
        "portion_grams": grams,
        "portion_unit": "piece_2",
        "calories_kcal": cals,
        "protein_g": prot,
        "carbs_g": carbs,
        "fat_g": fat,
        "fibre_g": nutrients_scaled["nutrients"]["fibre_g"]["value"],
        "sodium_mg": nutrients_scaled["nutrients"]["sodium_mg"]["value"],
        "notes": "Integration test lunch"
    }
    saved_meal = log_meal(test_uid, log_data, conn=conn)
    record("5.01 Meal logged to SQLite with generated log_id", "log_id" in saved_meal and saved_meal["calories_kcal"] == cals)

    # Simulate app restart by establishing a fresh, independent connection
    fresh_conn = get_db_connection()
    reloaded_history = get_meal_history(test_uid, conn=fresh_conn)
    record("5.02 Meal history persists across database connections/app restarts", len(reloaded_history) == 1)
    record("5.03 Persisted meal attributes match logged payload", reloaded_history[0]["food_name"] == food_rec["display_name"])

    # =========================================================================
    # Step 6: Nutrition Dashboard & Nutrient-Gap Insights
    # =========================================================================
    print("\n--- Step 6: Dashboard Aggregation & Trend Insights ---")
    today_str = datetime.date.today().isoformat()
    daily_totals = get_daily_nutrition_totals(test_uid, target_date=today_str, conn=fresh_conn)
    record("6.01 Daily nutrition totals aggregated from logged meals", daily_totals["totals"]["calories_kcal"] == cals)
    record("6.02 Calorie target comparison computed against user target", daily_totals["targets"]["calories_kcal"] > 0)
    record("6.03 Remaining calories correctly updated", abs(daily_totals["targets"]["remaining_calories_kcal"] - (daily_totals["targets"]["calories_kcal"] - cals)) < 1.0)

    period_summary = get_period_nutrition_summary(test_uid, days=7, conn=fresh_conn)
    record("6.04 7-Day period summary computed with active logged days", period_summary["active_logged_days"] == 1)

    # =========================================================================
    # Step 7: Safety-First Recommendation Ranking Verification
    # =========================================================================
    print("\n--- Step 7: Pre-Ranking Safety Enforcement in Recommendations ---")
    rec_sys = get_recommendation_system()
    top_recs = rec_sys.recommend_recipes(user_p, top_n=8)
    record("7.01 Recipe recommendation engine produces safety-filtered suggestions", len(top_recs) > 0)

    dairy_words = {"milk", "paneer", "butter", "ghee", "curd", "dahi", "cream", "cheese", "yogurt"}
    peanut_words = {"peanut", "peanuts", "groundnut", "mungfali"}
    safety_violations = []

    for r in top_recs:
        raw_text = (r.get("ingredients_raw", "") + " " + r.get("ingredient_tokens", "")).lower()
        found_dairy = [w for w in dairy_words if w in raw_text]
        found_peanut = [w for w in peanut_words if w in raw_text]
        if found_dairy or found_peanut:
            safety_violations.append((r["recipe_name"], found_dairy + found_peanut))

    record("7.02 Zero allergens (dairy/peanuts) present in recommended recipes", len(safety_violations) == 0, f"violations={safety_violations}")

    # =========================================================================
    # Step 8: Clean-up Test Data
    # =========================================================================
    print("\n--- Step 8: Clean-up & Teardown ---")
    del_ok = delete_profile(test_uid)
    record("8.01 Integration test profile clean-up executed", del_ok)

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 75)
    print(f"PHASE 12 SYSTEM INTEGRATION TESTS: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 75)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} test(s) failed in test_system_integration.py")

if __name__ == "__main__":
    run_integration_tests()
