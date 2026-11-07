"""
tests/test_phase13_comprehensive_validation.py
NutriSense AI — Phase 13: End-to-End System Testing & Scientific Validation Suite

Comprehensive verification across all required criteria:
1. Data loading and preprocessing pipelines.
2. Image prediction, top-k candidate distribution, and low-confidence handling.
3. Food-name mapping & subtype disambiguation.
4. Nutrition and portion calculations against laboratory benchmarks.
5. Missing and invalid values handling (preserves missing data, rejects NaNs/negatives).
6. Profile storage, updating, retrieval, and SQLite integrity.
7. Strict pre-ranking allergy and dietary restriction safety filters.
8. 0-100 Suitability scoring, transparency, and explanations.
9. Recipe recommendations, constraint compliance, and 7-day variety without consecutive repetition.
10. Meal logging, daily totals, and multi-day dashboard summaries.
11. SQLite multi-session persistence.
12. Streamlit frontend application syntax, structural integrity, and core interaction controls.
13. Model evaluation on held-out test split (Top-1 and Top-5 accuracy).
"""

import datetime
import json
import math
import os
import sys
import uuid
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Backend imports
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
    compute_daily_calorie_target,
    get_breakfast_suggestion,
    validate_profile_data
)
from src.nutrition_analyzer import (
    lookup_food,
    calculate_nutrients,
    convert_portion_to_grams,
    search_foods,
    get_available_servings,
    NUTRIENT_LABELS
)
from src.recommendation_engine import (
    get_recommendation_system,
    SafetyFilterEngine,
    SuitabilityScorer,
    AlternativeFoodsEngine,
    IngredientSubstitutionEngine
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

def run_phase13_tests():
    print("=" * 80)
    print("NUTRISENSE AI — PHASE 13: END-TO-END SYSTEM TESTING & VALIDATION")
    print("=" * 80)

    conn = get_db_connection()
    seed_reference_data_if_empty(conn)

    # =========================================================================
    # 1. Data Loading and Preprocessing
    # =========================================================================
    print("\n--- 1. Data Loading and Preprocessing ---")
    kb_path = WORKSPACE_ROOT / "data" / "processed" / "feature_store" / "master_food_nutrition_kb.json"
    recipe_csv_path = WORKSPACE_ROOT / "data" / "processed" / "feature_store" / "recipe_feature_store.csv"
    
    record("1.01 Master food nutrition KB file exists", kb_path.exists())
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    record("1.02 Master food nutrition KB contains validated records", len(kb_data) >= 43)

    record("1.03 Curated recipe feature store exists", recipe_csv_path.exists())
    df_recipes = pd.read_csv(recipe_csv_path)
    record("1.04 Recipe feature store contains indexed recipes", len(df_recipes) >= 5000)
    record("1.05 Recipe dataset contains necessary dietary metadata", 
           all(col in df_recipes.columns for col in ["recipe_name", "diet", "cuisine", "course", "total_time_mins", "ingredients_raw"]))

    # =========================================================================
    # 2. Image Prediction & Low-Confidence Handling
    # =========================================================================
    print("\n--- 2. Image Prediction & Low-Confidence Handling ---")
    inf_engine = NutriSenseInferenceEngine()
    record("2.01 Inference engine loaded with trained model", inf_engine.model is not None)
    record("2.02 Model covers 43 distinct authentic food classes", len(inf_engine.idx_to_class) == 43)

    # Blank / uniform image to test low-confidence behavior
    blank_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    blank_res = inf_engine.analyze_food(blank_img, top_k=5)
    blank_conf = blank_res["prediction"]["confidence"]
    record("2.03 Low-confidence/uncertain image produces low probability (<0.35)", blank_conf < 0.35, f"conf={blank_conf:.3f}")
    record("2.04 Candidate distribution returned even when uncertain", len(blank_res["prediction"]["top_k"]) == 5)

    # Real sample image prediction
    sample_path = WORKSPACE_ROOT / "static" / "samples" / "samosa.jpg"
    if sample_path.exists():
        real_img = Image.open(sample_path).convert("RGB")
        real_res = inf_engine.analyze_food(real_img, top_k=3)
        record("2.05 Valid food image produces prediction", real_res["prediction"]["confidence"] > 0)
        record("2.06 Samosa sample recognized among top candidates", 
               any(c["class_name"] == "samosa" for c in real_res["prediction"]["top_k"]))

    # =========================================================================
    # 3. Food-Name Mapping & Subtype Disambiguation
    # =========================================================================
    print("\n--- 3. Food-Name Mapping & Subtype Disambiguation ---")
    search_res = search_foods("samosa", limit=5)
    record("3.01 Search for 'samosa' resolves to catalog entries", len(search_res) > 0)
    record("3.02 Ambiguous class flagged with subtype options", 
           any(r["match_type"] == "AMBIGUOUS_SUBTYPE" for r in search_res))

    food_entry = lookup_food("samosa")
    record("3.03 lookup_food resolves ambiguous food class to canonical entry", food_entry is not None)
    record("3.04 Canonical entry carries standard serving weights", food_entry["reference_serving_g"] > 0)

    # =========================================================================
    # 4. Nutrition & Portion Calculations Against Benchmarks
    # =========================================================================
    print("\n--- 4. Nutrition & Portion Calculations Against Reference Records ---")
    servings = get_available_servings("samosa")
    unit_map = {s["unit"]: s["grams"] for s in servings}
    record("4.01 Standard servings include 100g reference and piece units", "100g" in unit_map and "standard_serving" in unit_map)

    # Portion conversion test
    g_100 = convert_portion_to_grams(1.0, "100g", "samosa")
    record("4.02 1 portion of 100g converts to 100.0g", abs(g_100 - 100.0) < 0.01)

    nut_100 = calculate_nutrients(food_entry, 100.0)
    cal_100 = nut_100["nutrients"]["calories_kcal"]["value"]
    # Verified baseline samosa per 100g is ~393.8 kcal
    record("4.03 Baseline samosa energy per 100g validated (~393.8 kcal)", abs(cal_100 - 393.8) < 1.0, f"cal={cal_100}")

    nut_200 = calculate_nutrients(food_entry, 200.0)
    cal_200 = nut_200["nutrients"]["calories_kcal"]["value"]
    record("4.04 Energy doubles linearly with portion doubling", abs(cal_200 - (cal_100 * 2.0)) < 0.1)

    # Check Atwater factor consistency: 4*protein + 4*carbs + 9*fat ≈ total calories
    prot = nut_100["nutrients"]["protein_g"]["value"]
    carbs = nut_100["nutrients"]["carbs_g"]["value"]
    fat = nut_100["nutrients"]["fat_g"]["value"]
    calc_cals = (prot * 4.0) + (carbs * 4.0) + (fat * 9.0)
    record("4.05 Atwater macronutrient consistency verified (within 10%)", abs(calc_cals - cal_100) / cal_100 < 0.10, f"atwater={calc_cals:.1f}")

    # =========================================================================
    # 5. Missing & Invalid Values Handling
    # =========================================================================
    print("\n--- 5. Missing & Invalid Values Handling ---")
    # Verify scientific data integrity: missing laboratory values are None, not 0
    missing_reported = False
    for k, v in nut_100["nutrients"].items():
        if v["value"] is None:
            missing_reported = True
            break
    record("5.01 Unmeasured laboratory micronutrients preserved as None (not zero-fabricated)", missing_reported)

    # Input validation checks for calculate_nutrients & convert_portion_to_grams
    try:
        calculate_nutrients(food_entry, -50.0)
        invalid_weight_handled = False
    except ValueError:
        invalid_weight_handled = True
    record("5.02 Negative portion weight raises ValueError", invalid_weight_handled)

    # Database meal logging validation checks
    bad_log = {"food_name": "Test", "portion_grams": float("nan"), "calories_kcal": -100}
    try:
        log_meal("dummy_user", bad_log, conn=conn)
        db_rejects_bad = False
    except (ValueError, Exception):
        db_rejects_bad = True
    record("5.03 Database rejects NaN and negative values on meal logging", db_rejects_bad)

    # =========================================================================
    # 6. Profile Storage, Updating & Retrieval
    # =========================================================================
    print("\n--- 6. Profile Storage, Updating & Retrieval ---")
    uid = f"p13_val_user_{uuid.uuid4().hex[:6]}"
    p_data = {
        "user_id": uid,
        "display_name": "Validation User",
        "age": 32,
        "gender": "male",
        "height_cm": 175.0,
        "weight_kg": 72.0,
        "activity_level": "sedentary",
        "dietary_goal": "weight_loss",
        "budget": "low",
        "max_cooking_time_mins": 30,
        "preferred_cuisines": ["North Indian"],
        "allergies": ["nuts", "dairy"],
        "dietary_restrictions": ["vegetarian"],
        "disliked_foods": ["karela"]
    }
    p_created = create_profile(p_data)
    record("6.01 Profile successfully stored in SQLite", p_created["user_id"] == uid)

    p_fetched = get_profile(uid)
    record("6.02 Profile accurately retrieved with all metadata", 
           p_fetched is not None and p_fetched["display_name"] == "Validation User")

    # Update profile
    p_updated = update_profile(uid, {"weight_kg": 70.5, "budget": "medium"})
    record("6.03 Profile updates persisted in SQLite", 
           p_updated["weight_kg"] == 70.5 and p_updated["budget"] == "medium")

    # =========================================================================
    # 7. Allergy & Dietary Safety Filters
    # =========================================================================
    print("\n--- 7. Allergy & Dietary Safety Filters ---")
    safety_engine = SafetyFilterEngine()

    safe_recipe = {
        "recipe_name": "Yellow Moong Dal Tadka",
        "ingredients_raw": "moong dal, cumin, turmeric, tomato, coriander",
        "ingredient_tokens": "moong dal cumin turmeric tomato coriander",
        "diet": "vegetarian"
    }
    dairy_recipe = {
        "recipe_name": "Paneer Tikka",
        "ingredients_raw": "paneer, yogurt, dahi, cream, spices",
        "ingredient_tokens": "paneer yogurt dahi cream spices",
        "diet": "vegetarian"
    }
    nut_recipe = {
        "recipe_name": "Badam Halwa",
        "ingredients_raw": "almonds, badam, cashews, sugar",
        "ingredient_tokens": "almonds badam cashews sugar",
        "diet": "vegetarian"
    }
    nonveg_recipe = {
        "recipe_name": "Mutton Rogan Josh",
        "ingredients_raw": "mutton, oil, spices",
        "ingredient_tokens": "mutton oil spices",
        "diet": "non_vegetarian"
    }

    chk_safe = safety_engine.check_safety(safe_recipe, p_updated)
    chk_dairy = safety_engine.check_safety(dairy_recipe, p_updated)
    chk_nut = safety_engine.check_safety(nut_recipe, p_updated)
    chk_nonveg = safety_engine.check_safety(nonveg_recipe, p_updated)

    record("7.01 Safe vegetarian dish without declared allergens passes check", chk_safe["is_safe"])
    record("7.02 Declared dairy allergen correctly caught and blocked", not chk_dairy["is_safe"])
    record("7.03 Declared nut allergen correctly caught and blocked", not chk_nut["is_safe"])
    record("7.04 Non-vegetarian recipe blocked for vegetarian profile", not chk_nonveg["is_safe"])

    # =========================================================================
    # 8. Suitability Scoring & Transparency
    # =========================================================================
    print("\n--- 8. Suitability Scoring & Factor Transparency ---")
    scorer = SuitabilityScorer()
    score_out = scorer.score_recipe(safe_recipe, p_updated)
    record("8.01 Suitability score is within valid bounds (0–100)", 0 <= score_out["total_score"] <= 100)
    record("8.02 Transparent component score breakdown present", 
           all(k in score_out["component_scores"] for k in ["nutrient_alignment", "preference_alignment", "time_feasibility", "budget_complexity", "goal_synergy"]))
    record("8.03 Non-clinical educational disclaimer attached", len(score_out["disclaimer"]) > 20)

    # =========================================================================
    # 9. Recipe Recommendations & Meal Planning Variety
    # =========================================================================
    print("\n--- 9. Recipe Recommendations & Variety Compliance ---")
    rec_sys = get_recommendation_system()
    recs = rec_sys.recommend_recipes(p_updated, top_n=6)
    record("9.01 Recommendations generated without runtime exceptions", len(recs) > 0)
    
    # Verify no dairy or nuts leaked into recommendations
    forbidden = {"paneer", "milk", "butter", "ghee", "curd", "dahi", "cream", "cheese", "yogurt", "almond", "badam", "cashew", "kaju", "walnut"}
    leaked = []
    for r in recs:
        text = (r.get("ingredients_raw", "") + " " + r.get("ingredient_tokens", "")).lower()
        found = [f for f in forbidden if f in text]
        if found:
            leaked.append((r["recipe_name"], found))
    record("9.02 Zero safety filter breaches across top recommendations", len(leaked) == 0, f"leaked={leaked}")

    # Generate 7-day plan and check variety (no identical meal on consecutive days)
    w_plan = rec_sys.generate_weekly_meal_plan(p_updated)
    record("9.03 7-Day weekly meal plan generated with full schedule", len(w_plan["weekly_schedule"]) == 7)

    days = list(w_plan["weekly_schedule"].keys())
    consecutive_reps = 0
    for i in range(len(days) - 1):
        d1 = w_plan["weekly_schedule"][days[i]]
        d2 = w_plan["weekly_schedule"][days[i + 1]]
        for meal in ["breakfast", "lunch", "dinner"]:
            if d1[meal]["recipe_name"] == d2[meal]["recipe_name"]:
                consecutive_reps += 1
    record("9.04 Variety constraint: zero consecutive-day meal repetitions", consecutive_reps == 0, f"reps={consecutive_reps}")

    # =========================================================================
    # 10. Meal Logging, Daily Totals & Multi-Session Persistence
    # =========================================================================
    print("\n--- 10. Meal Logging, Daily Totals & Multi-Session Persistence ---")
    meal1 = {
        "food_name": "Vegetable Poha",
        "food_id": "poha_test",
        "course": "breakfast",
        "portion_grams": 150.0,
        "portion_unit": "standard_serving",
        "calories_kcal": 260.0,
        "protein_g": 5.0,
        "carbs_g": 45.0,
        "fat_g": 6.0,
        "fibre_g": 3.0,
        "sodium_mg": 320.0,
        "notes": "Healthy breakfast"
    }
    meal2 = {
        "food_name": "Dal Tadka with Roti",
        "food_id": "dal_roti_test",
        "course": "lunch",
        "portion_grams": 250.0,
        "portion_unit": "standard_serving",
        "calories_kcal": 420.0,
        "protein_g": 14.0,
        "carbs_g": 68.0,
        "fat_g": 9.0,
        "fibre_g": 8.0,
        "sodium_mg": 480.0,
        "notes": "Home cooked"
    }

    log1 = log_meal(uid, meal1, conn=conn)
    log2 = log_meal(uid, meal2, conn=conn)
    record("10.01 Multiple meals logged successfully", "log_id" in log1 and "log_id" in log2)

    # Independent connection simulation for app restart persistence
    fresh_conn = get_db_connection()
    today_iso = datetime.date.today().isoformat()
    d_totals = get_daily_nutrition_totals(uid, target_date=today_iso, conn=fresh_conn)

    expected_cals = meal1["calories_kcal"] + meal2["calories_kcal"]
    actual_cals = d_totals["totals"]["calories_kcal"]
    record("10.02 Daily calorie total matches sum of logged meals (680 kcal)", abs(actual_cals - expected_cals) < 0.1, f"actual={actual_cals}")
    record("10.03 Logged meals persist in history across fresh database sessions", len(d_totals["meals"]) == 2)

    # =========================================================================
    # 11. Streamlit Application Core Component Verification
    # =========================================================================
    print("\n--- 11. Streamlit Application Code & Interaction Checks ---")
    app_py_path = WORKSPACE_ROOT / "streamlit_app.py"
    record("11.01 streamlit_app.py exists and is readable", app_py_path.exists())
    
    with open(app_py_path, "r", encoding="utf-8") as f:
        app_code = f.read()

    expected_sections = [
        "Home",
        "User Profile",
        "Log a Meal",
        "Nutrition Analysis",
        "Recommendations & Meal Planning",
        "Nutrition Dashboard"
    ]
    all_sections_present = all(sec in app_code for sec in expected_sections)
    record("11.02 All 6 primary application sections defined in frontend", all_sections_present)

    # Core interaction widgets
    widgets = ["st.file_uploader", "st.number_input", "st.selectbox", "st.radio", "st.bar_chart", "st.form"]
    record("11.03 Essential Streamlit interactive input & display widgets implemented", 
           all(w in app_code for w in widgets))

    # =========================================================================
    # 12. Model Evaluation on Held-Out Test Set
    # =========================================================================
    print("\n--- 12. Model Evaluation on Held-Out Test Split ---")
    test_split_path = WORKSPACE_ROOT / "data" / "processed" / "vision_splits" / "enhanced_test_split.csv"
    cache_dir = WORKSPACE_ROOT / "data" / "processed" / "images" / "enhanced_cache"

    if test_split_path.exists() and cache_dir.exists():
        df_test = pd.read_csv(test_split_path)
        # Evaluate on a representative held-out sample of 100 test images
        sample_eval = df_test.sample(n=min(100, len(df_test)), random_state=42)
        top1_corr = 0
        top5_corr = 0
        total_eval = 0

        for _, row in sample_eval.iterrows():
            img_file = cache_dir / f"{row['md5_hash']}.jpg"
            if not img_file.exists():
                continue
            true_label = row["class_name"]
            img = Image.open(img_file).convert("RGB")
            res = inf_engine.analyze_food(img, top_k=5)
            pred_label = res["prediction"]["class_name"]
            top_k_labels = [c["class_name"] for c in res["prediction"]["top_k"]]

            if pred_label == true_label:
                top1_corr += 1
            if true_label in top_k_labels:
                top5_corr += 1
            total_eval += 1

        top1_acc = top1_corr / total_eval if total_eval > 0 else 0
        top5_acc = top5_corr / total_eval if total_eval > 0 else 0
        record("12.01 Held-out test evaluation completed", total_eval >= 50, f"evaluated={total_eval}")
        record("12.02 Held-out Top-1 Accuracy benchmark (>70%)", top1_acc >= 0.70, f"Top-1={top1_acc:.4f} ({top1_corr}/{total_eval})")
        record("12.03 Held-out Top-5 Accuracy benchmark (>90%)", top5_acc >= 0.90, f"Top-5={top5_acc:.4f} ({top5_corr}/{total_eval})")
    else:
        record("12.01 Held-out test evaluation files accessible", False)

    # =========================================================================
    # Teardown
    # =========================================================================
    delete_profile(uid)

    # Scorecard
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 80)
    print(f"PHASE 13 TESTING & VALIDATION SCORECARD: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 80)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} validation checks failed.")

if __name__ == "__main__":
    run_phase13_tests()

