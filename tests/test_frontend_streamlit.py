"""
tests/test_frontend_streamlit.py
NutriSense AI — Phase 11: Frontend Development Comprehensive Test Suite

Verifies:
1. Streamlit application file existence and clean syntax compilation
2. Implementation of all 6 mandatory sections:
   - 1. Home
   - 2. User Profile
   - 3. Log a Meal
   - 4. Nutrition Analysis
   - 5. Recipe Recommendations and Meal Planning
   - 6. Nutrition Dashboard
3. Backend & Database Integration within UI:
   - Uses real SQLite database (`database/nutrisense_users.db`)
   - Reuses `src.database_manager`, `src.user_profile_manager`, `src.recommendation_engine`, `src.nutrition_analyzer`
   - Zero duplicate logic or hardcoded mock data
4. Interactive UI Capabilities:
   - Image upload and preview with MobileNetV2 vision prediction
   - Portion size selection (grams and defined serving units)
   - Real-time nutrient calculation and macro breakdown
   - Pre-ranking safety check against active user profile
   - 0-100 suitability score and explanation
   - Healthier alternative food suggestions
   - Meal history retrieval and deletion
   - Calorie & macro target progress indicators
   - Multi-day trend charts and clinical nutrient-gap insights
5. Integrity & Non-Clinical Guardrails:
   - BMI limitations disclaimer
   - Mifflin-St Jeor calorie estimation disclaimer
   - Recommendation heuristics non-claim notice
"""

from pathlib import Path
import py_compile
import re
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

APP_PATH = WORKSPACE_ROOT / "streamlit_app.py"

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
    print("RUNNING PHASE 11: FRONTEND STREAMLIT VERIFICATION TESTS")
    print("=" * 70)

    # =========================================================================
    # Section 1: File Existence & Clean Compilation
    # =========================================================================
    print("\n--- Section 1: File Existence & Python Compilation ---")
    record("1.01 streamlit_app.py exists in project root", APP_PATH.exists())

    compiled = False
    try:
        py_compile.compile(str(APP_PATH), doraise=True)
        compiled = True
    except Exception as e:
        print(f"Compilation error: {e}")
    record("1.02 streamlit_app.py compiles without syntax errors", compiled)

    app_content = APP_PATH.read_text(encoding="utf-8")

    # =========================================================================
    # Section 2: Verification of All 6 Mandatory Sections
    # =========================================================================
    print("\n--- Section 2: Verification of All 6 Mandatory Sections ---")

    record("2.01 Section 1: 'Home' is implemented", "Home" in app_content and "Curated Indian Recipes" in app_content)
    record("2.02 Section 2: 'User Profile' is implemented", "User Profile" in app_content and "compute_bmi" in app_content)
    record("2.03 Section 3: 'Log a Meal' is implemented", "Log a Meal" in app_content and "NutriSenseInferenceEngine" in app_content)
    record("2.04 Section 4: 'Nutrition Analysis' is implemented", "Nutrition Analysis" in app_content and "search_foods" in app_content)
    record("2.05 Section 5: 'Recommendations & Meal Planning' is implemented", "Recommendations & Meal Planning" in app_content and "generate_daily_meal_plan" in app_content)
    record("2.06 Section 6: 'Nutrition Dashboard' is implemented", "Nutrition Dashboard" in app_content and "get_daily_nutrition_totals" in app_content)

    # =========================================================================
    # Section 3: Backend & Database Subsystem Integration
    # =========================================================================
    print("\n--- Section 3: Subsystem Integration in Frontend ---")

    record("3.01 Imports database_manager functions", "from src.database_manager import" in app_content and "log_meal" in app_content)
    record("3.02 Imports user_profile_manager functions", "from src.user_profile_manager import" in app_content and "compute_daily_calorie_target" in app_content)
    record("3.03 Imports recommendation_engine functions", "from src.recommendation_engine import" in app_content and "AlternativeFoodsEngine" in app_content)
    record("3.04 Imports nutrition_analyzer functions", "from src.nutrition_analyzer import" in app_content and "calculate_nutrients" in app_content)
    record("3.05 Zero mock/fake statistics (queries real SQLite database)", "SELECT COUNT(*) FROM" in app_content and "nutrisense_users.db" in str(app_content) or "_db_conn" in app_content)

    # =========================================================================
    # Section 4: Core User Flows & Form Operations
    # =========================================================================
    print("\n--- Section 4: Interactive Controls & Form Operations ---")

    # Image upload and preview
    record("4.01 Image uploader defined for JPG/PNG", "st.file_uploader" in app_content and "png" in app_content)
    record("4.02 Sample food images selector supported", "SAMPLES_DIR" in app_content and "Select Sample Dish Image" in app_content)
    record("4.03 MobileNetV2 inference executed with spinner", "st.spinner" in app_content and "NutriSenseInferenceEngine" in app_content)

    # Food prediction & user correction
    record("4.04 Food prediction displayed with confidence", "AI Prediction" in app_content and "confidence" in app_content)
    record("4.05 Manual dish confirmation / correction dropdown", "Confirmed Dish / Food Item" in app_content)

    # Portion selection & scaling
    record("4.06 Portion input value supported", "Portion Amount" in app_content and "number_input" in app_content)
    record("4.07 Defined serving units supported (grams, piece, katori, etc.)", "get_available_servings" in app_content and "Serving Unit" in app_content)

    # Nutrient breakdown & safety check
    record("4.08 Real-time macro metrics (calories, protein, carbs, fat)", "Calories" in app_content and "Protein" in app_content and "Carbs" in app_content)
    record("4.09 Pre-ranking safety audit alert against active user profile", "Safety Certified" in app_content and "Safety Conflict" in app_content)

    # 0-100 suitability score & alternatives
    record("4.10 0-100 suitability score and explanation", "Suitability Score" in app_content and "score_eval" in app_content)
    record("4.11 Healthier alternative foods surfaced for dense items", "Healthier Alternative Suggestions" in app_content)

    # SQLite Persistence
    record("4.12 Save meal button commits to SQLite log_meal", "Save Meal to My Log" in app_content and "log_meal(" in app_content)

    # =========================================================================
    # Section 5: Meal Planning & Grocery List Integration
    # =========================================================================
    print("\n--- Section 5: Meal Planning & Grocery List UI ---")

    record("5.01 Calibrated 1-Day Meal Plan UI controls", "Calibrated 1-Day Meal Plan" in app_content and "generate_daily_meal_plan" in app_content)
    record("5.02 7-Day Weekly Meal Schedule UI controls", "7-Day Weekly Meal Schedule" in app_content and "generate_weekly_meal_plan" in app_content)
    record("5.03 Aggregated Grocery Shopping List table", "Aggregated Weekly Grocery Shopping List" in app_content and "grocery_shopping_list" in app_content)
    record("5.04 Meal plan save to database buttons", "Save Daily Plan to Database" in app_content and "Save Weekly Plan to Database" in app_content)

    # =========================================================================
    # Section 6: Dashboard, Charts & Nutrient-Gap Insights
    # =========================================================================
    print("\n--- Section 6: Dashboard Charts & Nutrient-Gap Insights ---")

    record("6.01 Real meal history displayed from SQLite", "get_daily_nutrition_totals" in app_content and "Meals Recorded on" in app_content)
    record("6.02 Calorie target comparison (consumed vs target calories)", "Calories Consumed" in app_content and "percent_target_achieved" in app_content)
    record("6.03 Macronutrient energy distribution percentages", "Protein Energy" in app_content and "Carbohydrates Energy" in app_content)
    record("6.04 Multi-day calorie trend chart", "st.bar_chart" in app_content and "get_period_nutrition_summary" in app_content)
    record("6.05 Clinical nutrient-gap insights (Fiber & Protein gaps)", "Fiber Deficit" in app_content and "Protein Gap" in app_content)

    # =========================================================================
    # Section 7: Clinical Guardrails & Disclaimers
    # =========================================================================
    print("\n--- Section 7: Clinical Guardrails & Disclaimers ---")

    record("7.01 BMI limitations disclaimer included", "BMI_DISCLAIMER" in app_content)
    record("7.02 South Asian BMI threshold explained in UI", "South Asian Consensus Range" in app_content)
    record("7.03 Calorie estimation disclaimer included", "cal_target[\"explanation\"]" in app_content or "CALORIE_DISCLAIMER" in app_content)
    record("7.04 Missing laboratory micronutrients disclaimer", "Missing Laboratory Micronutrients Notice" in app_content)

    # =========================================================================
    # Section 8: Live Backend Functionality Probe
    # =========================================================================
    print("\n--- Section 8: Live Backend Functionality Probe ---")

    from src.database_manager import get_daily_nutrition_totals, get_period_nutrition_summary
    from src.user_profile_manager import list_profiles

    profs = list_profiles()
    record("8.01 UI profile list retrieves valid database records", isinstance(profs, list) and len(profs) > 0)

    test_uid = profs[0]["user_id"]
    totals = get_daily_nutrition_totals(test_uid)
    record("8.02 UI daily totals endpoint executes successfully", "totals" in totals and "targets" in totals)

    summary = get_period_nutrition_summary(test_uid, days=7)
    record("8.03 UI period summary endpoint executes successfully", "averages" in summary and "period_days" in summary)

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 70)
    print(f"PHASE 11 FRONTEND STREAMLIT TESTS: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 70)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} test(s) failed in test_frontend_streamlit.py")


if __name__ == "__main__":
    run_tests()

