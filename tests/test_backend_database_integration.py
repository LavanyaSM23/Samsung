"""
tests/test_backend_database_integration.py
NutriSense AI — Phase 10: Backend and Database Integration Comprehensive Test Suite

Verifies:
1. Database Schema & Relational Integrity (user_profiles, foods, nutrients, recipes, allergens, meal_logs, meal_plans, feedback)
2. Parameterized SQL Queries & Injection Immunity
3. Input Validation, Integrity Constraints, and Graceful Error Handling
4. Baseline Reference Data Seeding (Foods, Nutrients, Recipes, Allergens)
5. Meal Logging & History Querying (filtering by date, course, user)
6. Daily Totals & Period Nutrition Summaries (target comparison, remaining calories, sodium alerts)
7. Meal Plan Persistence & Retrieval (daily and weekly plans)
8. User Feedback System (1-5 star validation, qualitative comments)
9. End-to-End Integrated Orchestrator Flow (vision -> nutrition -> safety -> scoring -> persistence)
10. REST Web API Endpoints Integration (/api/meals/*, /api/meal-plans/*, /api/feedback/*, /api/flow/process)
11. Preservation of Existing Working Features (Phases 1-9)
"""

from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app import app
from src.database_manager import (
    get_db_connection,
    init_database_schema,
    seed_reference_data_if_empty,
    log_meal,
    get_meal_history,
    delete_meal_log,
    get_daily_nutrition_totals,
    get_period_nutrition_summary,
    save_meal_plan,
    get_latest_meal_plan,
    log_user_feedback,
    get_user_feedback,
    get_backend_orchestrator,
    BackendOrchestrator,
)
from src.user_profile_manager import create_profile, delete_profile, get_profile

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
    print("=" * 70)
    print("RUNNING PHASE 10: BACKEND & DATABASE INTEGRATION VERIFICATION TESTS")
    print("=" * 70)

    conn = get_db_connection()

    # =========================================================================
    # Section 1: Relational Database Schema & Table Verification
    # =========================================================================
    print("\n--- Section 1: Relational Schema & Table Verification ---")

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {r[0] for r in cur.fetchall()}

    expected_tables = {
        "user_profiles", "foods", "nutrients", "recipes",
        "allergens", "meal_logs", "suitability_evaluations",
        "meal_plans", "user_feedback"
    }
    for t in expected_tables:
        record(f"1. Table '{t}' exists in SQLite database", t in tables)

    # Check index existence
    cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indexes = {r[0] for r in cur.fetchall()}
    record("1.10 Index idx_meal_logs_user_date exists", "idx_meal_logs_user_date" in indexes)
    record("1.11 Index idx_nutrients_food_id exists", "idx_nutrients_food_id" in indexes)

    # =========================================================================
    # Section 2: Reference Data Seeding
    # =========================================================================
    print("\n--- Section 2: Reference Data Seeding ---")

    seed_res = seed_reference_data_if_empty(conn)
    record("2.01 Seeding execution succeeds without error", isinstance(seed_res, dict))

    cur.execute("SELECT COUNT(*) FROM foods;")
    food_count = cur.fetchone()[0]
    record("2.02 Foods table contains seeded reference foods", food_count >= 43)

    cur.execute("SELECT COUNT(*) FROM nutrients;")
    nut_count = cur.fetchone()[0]
    record("2.03 Nutrients table contains seeded records", nut_count >= 43)

    cur.execute("SELECT COUNT(*) FROM recipes;")
    recipe_count = cur.fetchone()[0]
    record("2.04 Recipes table contains seeded 5,381 curated recipes", recipe_count >= 5000)

    cur.execute("SELECT COUNT(*) FROM allergens;")
    allergen_count = cur.fetchone()[0]
    record("2.05 Allergens table contains 8 standard allergen classes", allergen_count == 8)

    # =========================================================================
    # Section 3: Parameterized Queries & SQL Injection Immunity
    # =========================================================================
    print("\n--- Section 3: Parameterized Queries & Injection Immunity ---")

    # Attempt classic SQL injection in user_id query
    injection_user_id = "' OR '1'='1"
    history_injection = get_meal_history(injection_user_id, conn=conn)
    record("3.01 SQL injection string in user_id returns zero rows safely", len(history_injection) == 0)

    # Attempt injection in date filter
    history_inj_date = get_meal_history("valid_user", start_date="2026-01-01' OR '1'='1", conn=conn)
    record("3.02 SQL injection string in start_date handled safely", isinstance(history_inj_date, list))

    # =========================================================================
    # Section 4: Input Validation & Error Handling
    # =========================================================================
    print("\n--- Section 4: Input Validation & Error Handling ---")

    # Rejection of empty user_id
    err_user = False
    try:
        log_meal("", {"food_name": "Roti", "portion_grams": 50, "calories_kcal": 100}, conn=conn)
    except ValueError:
        err_user = True
    record("4.01 log_meal rejects empty user_id", err_user)

    # Rejection of empty food_name
    err_food = False
    try:
        log_meal("test_user_val", {"food_name": "", "portion_grams": 50, "calories_kcal": 100}, conn=conn)
    except ValueError:
        err_food = True
    record("4.02 log_meal rejects empty food_name", err_food)

    # Rejection of negative portion grams
    err_portion = False
    try:
        log_meal("test_user_val", {"food_name": "Roti", "portion_grams": -20.0, "calories_kcal": 100}, conn=conn)
    except ValueError:
        err_portion = True
    record("4.03 log_meal rejects negative portion_grams", err_portion)

    # Rejection of NaN portion grams
    err_nan = False
    try:
        log_meal("test_user_val", {"food_name": "Roti", "portion_grams": float("nan"), "calories_kcal": 100}, conn=conn)
    except ValueError:
        err_nan = True
    record("4.04 log_meal rejects NaN portion_grams", err_nan)

    # Rejection of negative calories
    err_cal = False
    try:
        log_meal("test_user_val", {"food_name": "Roti", "portion_grams": 50.0, "calories_kcal": -150.0}, conn=conn)
    except ValueError:
        err_cal = True
    record("4.05 log_meal rejects negative calories_kcal", err_cal)

    # =========================================================================
    # Section 5: Meal Logging, History & Deletion
    # =========================================================================
    print("\n--- Section 5: Meal Logging, History & Deletion ---")

    # Create test user profile
    test_u = create_profile({
        "display_name": "Integration Tester",
        "age": 28,
        "gender": "female",
        "height_cm": 162.0,
        "weight_kg": 56.0,
        "activity_level": "moderately_active",
        "dietary_goal": "maintain_weight"
    })
    uid = test_u["user_id"]

    # Log 3 distinct meals
    today_str = date.today().isoformat()
    m1 = log_meal(uid, {
        "food_name": "Idli with Sambar",
        "course": "breakfast",
        "portion_grams": 160.0,
        "calories_kcal": 240.0,
        "protein_g": 8.0,
        "carbs_g": 45.0,
        "fat_g": 2.5,
        "fibre_g": 5.0,
        "sodium_mg": 420.0,
        "meal_date": today_str
    }, conn=conn)
    record("5.01 Meal 1 (Breakfast) logged successfully", bool(m1.get("log_id")))

    m2 = log_meal(uid, {
        "food_name": "Dal Tadka with Steamed Rice",
        "course": "lunch",
        "portion_grams": 250.0,
        "calories_kcal": 420.0,
        "protein_g": 14.0,
        "carbs_g": 72.0,
        "fat_g": 8.0,
        "fibre_g": 7.5,
        "sodium_mg": 680.0,
        "meal_date": today_str
    }, conn=conn)
    record("5.02 Meal 2 (Lunch) logged successfully", bool(m2.get("log_id")))

    m3 = log_meal(uid, {
        "food_name": "Palak Paneer with 2 Rotis",
        "course": "dinner",
        "portion_grams": 280.0,
        "calories_kcal": 510.0,
        "protein_g": 22.0,
        "carbs_g": 48.0,
        "fat_g": 24.0,
        "fibre_g": 9.0,
        "sodium_mg": 750.0,
        "meal_date": today_str
    }, conn=conn)
    record("5.03 Meal 3 (Dinner) logged successfully", bool(m3.get("log_id")))

    # Retrieve history
    history_all = get_meal_history(uid, conn=conn)
    record("5.04 Meal history contains all 3 logged meals", len(history_all) >= 3)

    # Filter history by course
    hist_lunch = get_meal_history(uid, course="lunch", conn=conn)
    record("5.05 Course filter retrieves only lunch meals", all(m["course"].lower() == "lunch" for m in hist_lunch))

    # Filter history by date range
    hist_date = get_meal_history(uid, start_date=today_str, end_date=today_str, conn=conn)
    record("5.06 Date range filter retrieves today's meals", len(hist_date) >= 3)

    # Delete meal log
    del_ok = delete_meal_log(m1["log_id"], uid, conn=conn)
    record("5.07 delete_meal_log returns True for valid log", del_ok)

    hist_after_del = get_meal_history(uid, conn=conn)
    record("5.08 Deleted meal no longer in history", not any(m["log_id"] == m1["log_id"] for m in hist_after_del))

    # Unauthorized delete (different user)
    del_unauth = delete_meal_log(m2["log_id"], "unauthorized_user_xyz", conn=conn)
    record("5.09 Unauthorized delete returns False", not del_unauth)

    # =========================================================================
    # Section 6: Daily Totals & Period Nutrition Summaries
    # =========================================================================
    print("\n--- Section 6: Daily Totals & Nutrition Summaries ---")

    daily_totals = get_daily_nutrition_totals(uid, target_date=today_str, conn=conn)
    record("6.01 Daily totals returns dict for active user", isinstance(daily_totals, dict))
    record("6.02 Daily totals aggregated calories match sum of m2 + m3", abs(daily_totals["totals"]["calories_kcal"] - (420.0 + 510.0)) < 1.0)
    record("6.03 Daily totals aggregated protein calculated", daily_totals["totals"]["protein_g"] == (14.0 + 22.0))
    record("6.04 Targets present with personalized Mifflin-St Jeor source", "targets" in daily_totals)
    record("6.05 Remaining calories calculated", "remaining_calories_kcal" in daily_totals["targets"])
    record("6.06 Percent target achieved calculated", daily_totals["targets"]["percent_target_achieved"] > 0)
    record("6.07 Meals list embedded in daily totals", len(daily_totals["meals"]) == 2)

    # Sodium overage alert test
    m_high_salt = log_meal(uid, {
        "food_name": "High Sodium Pickled Curry",
        "course": "snack",
        "portion_grams": 100.0,
        "calories_kcal": 150.0,
        "sodium_mg": 1500.0,
        "meal_date": today_str
    }, conn=conn)
    totals_alert = get_daily_nutrition_totals(uid, target_date=today_str, conn=conn)
    record("6.08 Sodium alert triggered when daily sodium > 2000 mg", any("sodium" in a["message"].lower() for a in totals_alert["alerts"]))

    # 7-day period summary
    summary_7d = get_period_nutrition_summary(uid, days=7, conn=conn)
    record("6.09 Period summary active_logged_days > 0", summary_7d["active_logged_days"] > 0)
    record("6.10 Period summary daily calories average calculated", summary_7d["averages"]["calories_kcal"] > 0)

    # Clean up m_high_salt
    delete_meal_log(m_high_salt["log_id"], uid, conn=conn)

    # =========================================================================
    # Section 7: Meal Plan Persistence & Retrieval
    # =========================================================================
    print("\n--- Section 7: Meal Plan Persistence & Retrieval ---")

    sample_plan = {
        "daily_target_calories_kcal": 1850.0,
        "meals": {
            "breakfast": {"recipe_name": "Oats Upma", "target_calories_kcal": 450.0},
            "lunch": {"recipe_name": "Rajma Chawal with Salad", "target_calories_kcal": 650.0},
            "snack": {"recipe_name": "Roasted Makhana", "target_calories_kcal": 150.0},
            "dinner": {"recipe_name": "Vegetable Pulao with Raita", "target_calories_kcal": 600.0},
        }
    }
    saved_p = save_meal_plan(uid, sample_plan, plan_type="daily", conn=conn)
    record("7.01 save_meal_plan returns plan_id", bool(saved_p.get("plan_id")))

    fetched_p = get_latest_meal_plan(uid, plan_type="daily", conn=conn)
    record("7.02 get_latest_meal_plan retrieves saved plan", fetched_p is not None)
    record("7.03 Deserialized plan_data matches input", fetched_p["plan_data"]["daily_target_calories_kcal"] == 1850.0)
    record("7.04 Fetched plan has breakfast meal", "breakfast" in fetched_p["plan_data"]["meals"])

    # Nonexistent user plan returns None
    p_none = get_latest_meal_plan("nonexistent_user_999", conn=conn)
    record("7.05 get_latest_meal_plan for nonexistent user returns None", p_none is None)

    # =========================================================================
    # Section 8: User Feedback Operations
    # =========================================================================
    print("\n--- Section 8: User Feedback Operations ---")

    fb1 = log_user_feedback(uid, {
        "item_id": "rec_palak_paneer",
        "item_type": "recommendation",
        "rating": 5,
        "feedback_type": "liked",
        "comments": "Loved the portion sizing and accurate calorie estimation!"
    }, conn=conn)
    record("8.01 log_user_feedback returns feedback_id", bool(fb1.get("feedback_id")))
    record("8.02 Rating 5 recorded", fb1["rating"] == 5)

    # Rating validation (out of bounds)
    err_fb_high = False
    try:
        log_user_feedback(uid, {"rating": 6}, conn=conn)
    except ValueError:
        err_fb_high = True
    record("8.03 Rating 6 rejected (out of bounds > 5)", err_fb_high)

    err_fb_low = False
    try:
        log_user_feedback(uid, {"rating": 0}, conn=conn)
    except ValueError:
        err_fb_low = True
    record("8.04 Rating 0 rejected (out of bounds < 1)", err_fb_low)

    # Fetch feedback history
    fb_list = get_user_feedback(uid, conn=conn)
    record("8.05 get_user_feedback returns list containing fb1", len(fb_list) > 0 and fb_list[0]["feedback_id"] == fb1["feedback_id"])

    # =========================================================================
    # Section 9: Integrated End-to-End Orchestrator Flow
    # =========================================================================
    print("\n--- Section 9: Integrated Orchestrator Flow ---")

    orch = get_backend_orchestrator()

    # Flow 1: Vision prediction to meal logging
    flow1 = orch.process_food_recognition_and_logging(
        food_name_or_prediction="samosa",
        portion_value=2,
        unit="piece",
        user_id=uid,
        course="snack",
        save_to_log=True,
        notes="Evening snack after workout"
    )
    record("9.01 Flow completes with success=True", flow1["success"] is True)
    record("9.02 Food identification resolved display_name", "Samosa" in flow1["food_record"]["display_name"])
    record("9.03 Portion converted to grams (2 pieces = 130g)", flow1["portion"]["grams"] == 130.0)
    record("9.04 Scaled calories calculated (> 400 kcal)", flow1["nutrients"]["calories_kcal"]["value"] > 400)
    record("9.05 Safety check evaluated against user profile", "is_safe" in flow1["safety"])
    record("9.06 Suitability score evaluated", "total_score" in flow1["suitability_evaluation"])
    record("9.07 Healthier alternatives provided for samosa", len(flow1["healthier_alternatives"]) > 0)
    record("9.08 Meal logged to SQLite with log_id", bool(flow1["logged_meal"]["log_id"]))
    record("9.09 Updated daily totals recalculated", flow1["updated_daily_totals"]["totals"]["calories_kcal"] > 0)

    # Flow 2: Unknown food raises KeyError
    err_flow_food = False
    try:
        orch.process_food_recognition_and_logging("unknown_extraterrestrial_dish", 100.0, user_id=uid)
    except KeyError:
        err_flow_food = True
    record("9.10 Unknown food item raises KeyError in orchestrator", err_flow_food)

    # Clean up logged samosa
    if flow1.get("logged_meal"):
        delete_meal_log(flow1["logged_meal"]["log_id"], uid, conn=conn)

    # =========================================================================
    # Section 10: Web REST API Endpoints Integration
    # =========================================================================
    print("\n--- Section 10: Web REST API Endpoints Integration ---")

    app.config["TESTING"] = True
    client = app.test_client()

    # POST /api/meals/log
    resp_log = client.post("/api/meals/log", json={
        "user_id": uid,
        "food_name": "API Test Dish",
        "portion_grams": 200.0,
        "calories_kcal": 300.0,
        "protein_g": 10.0,
        "course": "lunch"
    })
    record("10.01 POST /api/meals/log returns 201 Created", resp_log.status_code == 201)
    log_json = resp_log.get_json()
    record("10.02 API returns logged meal payload", log_json["success"] is True and bool(log_json["data"]["log_id"]))
    api_log_id = log_json["data"]["log_id"]

    # GET /api/meals/history
    resp_hist = client.get(f"/api/meals/history?user_id={uid}")
    record("10.03 GET /api/meals/history returns 200", resp_hist.status_code == 200)
    hist_json = resp_hist.get_json()
    record("10.04 History count > 0", hist_json["count"] > 0)

    # GET /api/meals/daily-totals
    resp_tot = client.get(f"/api/meals/daily-totals?user_id={uid}")
    record("10.05 GET /api/meals/daily-totals returns 200", resp_tot.status_code == 200)
    tot_json = resp_tot.get_json()
    record("10.06 Daily totals response contains totals and targets", "totals" in tot_json["data"] and "targets" in tot_json["data"])

    # GET /api/meals/summary
    resp_sum = client.get(f"/api/meals/summary?user_id={uid}&days=7")
    record("10.07 GET /api/meals/summary returns 200", resp_sum.status_code == 200)
    sum_json = resp_sum.get_json()
    record("10.08 Summary response contains averages", "averages" in sum_json["data"])

    # POST /api/meal-plans/save
    resp_mp_save = client.post("/api/meal-plans/save", json={
        "user_id": uid,
        "plan_type": "daily",
        "plan_data": sample_plan
    })
    record("10.09 POST /api/meal-plans/save returns 201", resp_mp_save.status_code == 201)

    # GET /api/meal-plans/latest
    resp_mp_lat = client.get(f"/api/meal-plans/latest?user_id={uid}&plan_type=daily")
    record("10.10 GET /api/meal-plans/latest returns 200", resp_mp_lat.status_code == 200)

    # POST /api/feedback/log
    resp_fb = client.post("/api/feedback/log", json={
        "user_id": uid,
        "rating": 5,
        "item_type": "recommendation",
        "comments": "Excellent integration!"
    })
    record("10.11 POST /api/feedback/log returns 201", resp_fb.status_code == 201)

    # GET /api/feedback
    resp_fb_get = client.get(f"/api/feedback?user_id={uid}")
    record("10.12 GET /api/feedback returns 200", resp_fb_get.status_code == 200)

    # POST /api/flow/process (Full end-to-end endpoint)
    resp_flow = client.post("/api/flow/process", json={
        "user_id": uid,
        "food_name": "idli",
        "portion_value": 2,
        "unit": "piece",
        "course": "breakfast",
        "save_to_log": True
    })
    record("10.13 POST /api/flow/process returns 200", resp_flow.status_code == 200)
    flow_json = resp_flow.get_json()
    record("10.14 Flow endpoint returns food_record, nutrients, safety, evaluation", all(k in flow_json for k in ["food_record", "nutrients", "safety", "suitability_evaluation"]))

    # DELETE /api/meals/<log_id>
    resp_del = client.delete(f"/api/meals/{api_log_id}?user_id={uid}")
    record("10.15 DELETE /api/meals/<log_id> returns 200", resp_del.status_code == 200)

    # DELETE nonexistent meal returns 404
    resp_del_404 = client.delete(f"/api/meals/nonexistent_id_999?user_id={uid}")
    record("10.16 DELETE nonexistent meal returns 404", resp_del_404.status_code == 404)

    # =========================================================================
    # Section 11: Preservation of Existing Working Features
    # =========================================================================
    print("\n--- Section 11: Preservation of Existing Features ---")

    # Verify Phase 8 user profile retrieval works
    p_check = get_profile(uid)
    record("11.01 Phase 8 user profile persists and is retrievable", p_check is not None and p_check["display_name"] == "Integration Tester")

    # Verify Phase 7 nutrition search endpoint works
    resp_p7 = client.get("/api/nutrition/search?q=biryani")
    record("11.02 Phase 7 /api/nutrition/search continues to return 200", resp_p7.status_code == 200 and resp_p7.get_json()["count"] > 0)

    # Verify Phase 9 recommendation endpoint works
    resp_p9 = client.post("/api/recommendations/recipes", json={"user_id": uid, "top_n": 3})
    record("11.03 Phase 9 /api/recommendations/recipes continues to return 200", resp_p9.status_code == 200 and len(resp_p9.get_json()["data"]) == 3)

    # Clean up test user
    delete_profile(uid)

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 70)
    print(f"PHASE 10 BACKEND & DATABASE TESTS: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 70)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} test(s) failed in test_backend_database_integration.py")


if __name__ == "__main__":
    run_tests()

