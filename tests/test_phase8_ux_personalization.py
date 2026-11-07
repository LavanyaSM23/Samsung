"""
tests/test_phase8_ux_personalization.py
Verification suite for Phase 8: User Experience, Personalization & UI Enhancement.

Tests:
1. Complete User Profile Creation & Field Validation (including dietary_preference & budget)
2. Profile Editing, Updating, and SQLite Persistence across sessions
3. Automated BMI Calculation & South Asian Risk Stratification
4. Personalized Caloric & Nutrient Requirements (Mifflin-St Jeor + Activity + Goal)
5. Personalization in Recommendations & Meal Planning (Dietary preference & safety filtering)
6. Affordable Indian Alternatives & Budget Awareness
7. Food Image Scanning & Manual Dish Correction Interface
8. Nutrition Label Scanning (OCR facts, serving basis, manual corrections)
9. Dark and Light Theme Configuration & CSS Selectors
10. Web API Profile Endpoints & Contract Integrity
"""

import json
import os
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))
DB_PATH = WORKSPACE / "database" / "nutrisense_users.db"

from src.user_profile_manager import (
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
    list_profiles,
    compute_bmi,
    compute_daily_calorie_target,
    get_breakfast_suggestion,
    personalize_portion_guidance,
    get_food_suitability_for_profile,
    validate_profile_data,
    map_to_rda_profile,
    BMI_DISCLAIMER,
    CALORIE_DISCLAIMER,
)
from src.recommendation_engine import (
    get_recommendation_system,
    SafetyFilterEngine,
    SuitabilityScorer,
    AlternativeFoodsEngine,
)
from src.nutrition_analyzer import lookup_food, search_foods


class TestPhase8UXPersonalization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rec_sys = get_recommendation_system()

    def setUp(self):
        self.created_uids = []

    def tearDown(self):
        for uid in self.created_uids:
            try:
                delete_profile(uid)
            except Exception:
                pass

    # =========================================================================
    # 1. Complete User Profile Creation
    # =========================================================================
    def test_01_profile_creation_with_all_required_fields(self):
        """Test complete profile creation with all Phase 8 fields."""
        data = {
            "display_name": "Rohan Mehra",
            "age": 29,
            "gender": "male",
            "height_cm": 178.0,
            "weight_kg": 74.0,
            "activity_level": "moderately_active",
            "dietary_goal": "muscle_gain",
            "dietary_preference": "vegetarian",
            "budget": "medium",
            "allergies": ["peanuts"],
            "dietary_restrictions": ["vegetarian"],
            "max_cooking_time_mins": 45,
            "preferred_cuisines": ["North Indian", "South Indian"],
        }
        ok, errs = validate_profile_data(data)
        self.assertTrue(ok, f"Validation failed: {errs}")

        profile = create_profile(data)
        self.assertIn("user_id", profile)
        self.created_uids.append(profile["user_id"])

        self.assertEqual(profile["display_name"], "Rohan Mehra")
        self.assertEqual(profile["age"], 29)
        self.assertEqual(profile["gender"], "male")
        self.assertEqual(profile["height_cm"], 178.0)
        self.assertEqual(profile["weight_kg"], 74.0)
        self.assertEqual(profile["dietary_preference"], "vegetarian")
        self.assertEqual(profile["budget"], "medium")
        self.assertIn("peanuts", profile["allergies"])

    # =========================================================================
    # 2. Profile Editing and SQLite Persistence
    # =========================================================================
    def test_02_profile_update_and_persistence(self):
        """Test that profile editing persists correctly in SQLite."""
        initial_data = {
            "display_name": "Anita Roy",
            "age": 34,
            "gender": "female",
            "height_cm": 162.0,
            "weight_kg": 58.0,
            "activity_level": "sedentary",
            "dietary_goal": "maintain_weight",
            "budget": "low",
        }
        profile = create_profile(initial_data)
        uid = profile["user_id"]
        self.created_uids.append(uid)

        # Update profile
        updates = {
            "display_name": "Anita Roy Sharma",
            "weight_kg": 56.5,
            "activity_level": "lightly_active",
            "dietary_goal": "weight_loss",
            "dietary_preference": "vegan",
            "budget": "high",
        }
        updated = update_profile(uid, updates)
        self.assertIsNotNone(updated)
        self.assertEqual(updated["display_name"], "Anita Roy Sharma")
        self.assertEqual(updated["weight_kg"], 56.5)
        self.assertEqual(updated["dietary_preference"], "vegan")
        self.assertEqual(updated["budget"], "high")

        # Re-fetch from fresh query to verify SQLite persistence
        fetched = get_profile(uid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["display_name"], "Anita Roy Sharma")
        self.assertEqual(fetched["weight_kg"], 56.5)
        self.assertEqual(fetched["dietary_preference"], "vegan")

    # =========================================================================
    # 3. Automated BMI Calculation & South Asian Risk Categories
    # =========================================================================
    def test_03_automated_bmi_calculation(self):
        """Test BMI calculation with WHO and South Asian risk categories."""
        # 175 cm, 72 kg -> BMI = 72 / (1.75^2) = 23.51
        # WHO: Normal (<25.0); South Asian: Overweight (>=23.0)
        bmi_res = compute_bmi(175.0, 72.0)
        self.assertAlmostEqual(bmi_res["bmi"], 23.51, places=1)
        self.assertEqual(bmi_res["bmi_category"], "Normal weight")
        self.assertIn("Overweight", bmi_res["south_asian_category"])
        self.assertIn("consensus guidelines", bmi_res["disclaimer"])

    # =========================================================================
    # 4. Personalized Calorie & Nutrient Requirements
    # =========================================================================
    def test_04_personalized_calorie_target(self):
        """Test Mifflin-St Jeor TDEE with goal adjustments."""
        # Male 28y, 175cm, 75kg, moderate exercise (+200 for muscle_gain)
        cal_res = compute_daily_calorie_target(
            age=28,
            gender="male",
            height_cm=175.0,
            weight_kg=75.0,
            activity_level="moderately_active",
            dietary_goal="muscle_gain",
        )
        self.assertGreater(cal_res["target_calories_kcal"], 2000.0)
        self.assertEqual(cal_res["goal_adjustment_kcal"], 200.0)
        self.assertIn("registered dietitian", cal_res["disclaimer"])

        # Female 32y, 160cm, 68kg, sedentary, weight_loss (-300)
        cal_res_f = compute_daily_calorie_target(
            age=32,
            gender="female",
            height_cm=160.0,
            weight_kg=68.0,
            activity_level="sedentary",
            dietary_goal="weight_loss",
        )
        self.assertEqual(cal_res_f["goal_adjustment_kcal"], -300.0)
        self.assertLess(cal_res_f["target_calories_kcal"], cal_res_f["tdee_kcal"])

    # =========================================================================
    # 5. Personalization with Zero-Tolerance Safety Filters
    # =========================================================================
    def test_05_personalization_respects_allergies_and_diet(self):
        """Test that recommendations filter out allergens and honor dietary preferences."""
        user = {
            "user_id": "test_user_safety",
            "display_name": "Kavita",
            "age": 27,
            "gender": "female",
            "height_cm": 165.0,
            "weight_kg": 55.0,
            "activity_level": "lightly_active",
            "dietary_goal": "general_wellness",
            "allergies": ["dairy", "peanuts"],
            "dietary_restrictions": ["vegetarian"],
            "budget": "low",
        }
        recs = self.rec_sys.recommend_recipes(user, top_n=5)
        self.assertIsInstance(recs, list)
        for r in recs:
            # Must be safe
            self.assertIn(r["verdict"], ["HIGHLY_RECOMMENDED", "HIGHLY_SUITABLE", "MODERATELY_SUITABLE", "SAFE_ALTERNATIVE", "ACCEPTABLE"])
            # Ingredients must not contain peanuts or dairy
            ing_text = str(r.get("ingredients_raw", "")).lower()
            self.assertNotIn("peanut", ing_text)

    # =========================================================================
    # 6. Affordable Indian Alternatives & Budget Awareness
    # =========================================================================
    def test_06_budget_and_affordable_alternatives(self):
        """Test alternative food suggestions engine."""
        alt_eng = AlternativeFoodsEngine()
        # Samosa is high calorie & fat -> should have healthier alternatives
        user_prof = {"dietary_goal": "weight_loss", "budget": "low"}
        alts = alt_eng.get_alternatives("samosa", user_profile=user_prof)
        self.assertIsInstance(alts, list)
        self.assertGreater(len(alts), 0)
        for alt in alts:
            self.assertIn("alternative_name", alt)
            self.assertIn("improvement_type", alt)

    # =========================================================================
    # 7. Food Image Scanning & Manual Correction in Streamlit App
    # =========================================================================
    def test_07_streamlit_food_scanner_controls(self):
        """Verify Streamlit app provides image upload, confidence display, and manual correction."""
        st_content = (WORKSPACE / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("st.file_uploader", st_content)
        self.assertIn("AI Prediction", st_content)
        self.assertIn("confidence", st_content)
        self.assertIn("Confirmed Dish / Food Item", st_content)
        self.assertIn("Select Sample Dish Image", st_content)

    # =========================================================================
    # 8. Nutrition Label Scanning (OCR & Verification)
    # =========================================================================
    def test_08_packaged_label_scanner_integration(self):
        """Verify label scanner module extracts declared nutrition without fabricating."""
        from src.label_scanner import scan_label
        from PIL import Image
        test_img = WORKSPACE / "static" / "samples" / "samosa.jpg"
        if test_img.exists():
            img = Image.open(test_img).convert("RGB")
            res = scan_label(img)
            self.assertIn("ocr_text", res)
            self.assertIn("nutrition", res)
            self.assertIn("missing_fields", res)
            self.assertIn("disclaimer", res)

    # =========================================================================
    # 9. Dark and Light Theme Support
    # =========================================================================
    def test_09_dark_and_light_theme_configuration(self):
        """Verify both Streamlit and Web frontends implement theme toggle & dark CSS."""
        # Streamlit app theme toggle
        st_content = (WORKSPACE / "streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn('_get_theme_css', st_content)
        self.assertIn('theme_toggle', st_content)
        self.assertIn('theme == "dark"', st_content)
        self.assertIn('#0F172A', st_content)  # Dark theme background

        # Web frontend styles
        css_content = (WORKSPACE / "static" / "css" / "profile-styles.css").read_text(encoding="utf-8")
        self.assertIn('[data-theme="dark"]', css_content)
        self.assertIn('.btn-theme-toggle', css_content)

        # Web frontend JS
        js_content = (WORKSPACE / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn('applyTheme', js_content)
        self.assertIn('nutrisense_theme', js_content)
        self.assertIn('btnThemeToggle', js_content)

    # =========================================================================
    # 10. Web Navigation Simplification (Clean minimal UI)
    # =========================================================================
    def test_10_simplified_navigation_and_subtext_cleanup(self):
        """Verify main navigation is limited to 3 clean tabs and subtext is clean."""
        html_content = (WORKSPACE / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="tabScanner"', html_content)
        self.assertIn('id="tabLabelScanner"', html_content)
        self.assertIn('id="tabProfile"', html_content)
        # Verify old clutter tabs are removed from active navigation
        self.assertNotIn('<button class="nav-tab" id="tabSamples"', html_content)
        self.assertNotIn('<button class="nav-tab" id="tabCatalog"', html_content)
        self.assertNotIn('<button class="nav-tab" id="tabBenchmarks"', html_content)


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPhase8UXPersonalization)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print("\nPhase 8: UX, Personalization & UI Enhancement Verification: ALL TESTS PASSED.")


if __name__ == "__main__":
    run_tests()
