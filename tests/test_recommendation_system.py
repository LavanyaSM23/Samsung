"""
tests/test_recommendation_system.py
NutriSense AI — Phase 9: Recommendation System Comprehensive Test Suite

Tests:
1. Safety Filter Engine:
   - Allergen detection (dairy, peanuts, nuts, gluten, eggs, shellfish, soy, sesame).
   - Dietary restriction filtering (vegetarian, vegan, eggetarian, jain, gluten-free).
   - Flagging incomplete ingredient or allergen data.
   - MANDATORY SAFETY OVERRIDE: High score NEVER overrides safety filters.
2. Transparent 0 to 100 Suitability Scorer:
   - Component scores: Nutrient alignment (35%), Preferences (25%), Feasibility (15%), Budget (15%), Synergy (10%).
   - Total score bounded [0, 100].
   - Plain-language explanation generation.
   - Non-clinical recommendation disclaimer attached.
3. Content-Based Filtering:
   - TF-IDF vectorization & Cosine similarity across recipe corpus.
   - Query similarity ranking.
4. Alternative Foods Engine:
   - Healthy swaps for fried/dense foods (samosa, butter naan, jalebi, chole bhature, biryani).
   - Justification and calorie saving explanations.
5. Ingredient Substitution Engine:
   - Ingredient-level swaps (milk, paneer, ghee, maida, sugar, salt, cashew, egg).
   - Recipe-level conflict scanning & culinary swap suggestions.
6. Daily Meal Planning:
   - 4-meal plan (Breakfast, Lunch, Snack, Dinner).
   - Caloric distribution calibration (~25%, ~35%, ~10%, ~30%).
   - All meals pass safety filter.
7. Weekly Meal Planning:
   - 7-day schedule (Monday to Sunday) with variety control (no consecutive repeats).
   - Aggregated grocery shopping list.
8. REST Web API Endpoints:
   - /api/recommendations/recipes
   - /api/recommendations/alternatives
   - /api/recommendations/substitutions
   - /api/recommendations/score-recipe
   - /api/recommendations/daily-plan
   - /api/recommendations/weekly-plan
   - Error handling (400, 404, invalid inputs).
"""

from pathlib import Path
import sys
import unittest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app import app
from src.recommendation_engine import (
    get_recommendation_system,
    SafetyFilterEngine,
    SuitabilityScorer,
    ContentBasedRecommender,
    AlternativeFoodsEngine,
    IngredientSubstitutionEngine,
    RecommendationSystem,
    RECOMMENDATION_DISCLAIMER,
    SAFETY_OVERRIDE_NOTICE,
    ALLERGEN_LEXICON,
)
from src.user_profile_manager import create_profile, delete_profile

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
    print("RUNNING PHASE 9: RECOMMENDATION SYSTEM VERIFICATION TESTS")
    print("=" * 70)

    engine = get_recommendation_system()
    safety = SafetyFilterEngine()
    scorer = SuitabilityScorer()
    alt_engine = AlternativeFoodsEngine()
    sub_engine = IngredientSubstitutionEngine()

    # =========================================================================
    # Section 1: Safety Filter — Allergen Detection & Strict Exclusion
    # =========================================================================
    print("\n--- Section 1: Allergen Detection & Safety Filtering ---")

    rec_dairy = {
        "recipe_id": "t_dairy_01",
        "recipe_name": "Shahi Paneer",
        "ingredients_raw": "500g paneer, 2 tbsp butter, 1/2 cup fresh cream, 1 cup milk, spices",
        "ingredient_tokens": "paneer butter cream milk spices",
        "diet": "vegetarian",
    }
    p_dairy_allergy = {"allergies": ["dairy"], "dietary_restrictions": []}
    s_dairy = safety.check_safety(rec_dairy, p_dairy_allergy)
    record("1.01 Dairy allergy detected in Shahi Paneer", not s_dairy["is_safe"])
    record("1.02 Dairy conflict reported in conflicts list", any("DAIRY" in c for c in s_dairy["allergen_conflicts"]))

    rec_peanut = {
        "recipe_id": "t_peanut_01",
        "recipe_name": "Peanut Poha",
        "ingredients_raw": "poha, 1/2 cup roasted peanuts, mustard seeds, green chilli",
        "ingredient_tokens": "poha roasted peanuts mustard seeds green chilli",
        "diet": "vegetarian",
    }
    p_peanut_allergy = {"allergies": ["peanuts"], "dietary_restrictions": []}
    s_peanut = safety.check_safety(rec_peanut, p_peanut_allergy)
    record("1.03 Peanut allergy detected in Peanut Poha", not s_peanut["is_safe"])

    rec_nuts = {
        "recipe_id": "t_nuts_01",
        "recipe_name": "Kaju Curry",
        "ingredients_raw": "cashews, almond paste, tomatoes, garam masala",
        "ingredient_tokens": "cashews almond paste tomatoes garam masala",
        "diet": "vegetarian",
    }
    p_nuts_allergy = {"allergies": ["nuts"], "dietary_restrictions": []}
    s_nuts = safety.check_safety(rec_nuts, p_nuts_allergy)
    record("1.04 Tree nut allergy detected in Kaju Curry", not s_nuts["is_safe"])

    rec_gluten = {
        "recipe_id": "t_gluten_01",
        "recipe_name": "Butter Naan",
        "ingredients_raw": "2 cups maida (all purpose flour), baking powder, yogurt, butter",
        "ingredient_tokens": "maida all purpose flour baking powder yogurt butter",
        "diet": "vegetarian",
    }
    p_gluten_allergy = {"allergies": ["gluten"], "dietary_restrictions": []}
    s_gluten = safety.check_safety(rec_gluten, p_gluten_allergy)
    record("1.05 Gluten allergy detected in Butter Naan (maida)", not s_gluten["is_safe"])

    rec_egg = {
        "recipe_id": "t_egg_01",
        "recipe_name": "Egg Bhurji",
        "ingredients_raw": "4 whole eggs, onions, tomatoes, green chillies",
        "ingredient_tokens": "whole eggs onions tomatoes green chillies",
        "diet": "eggetarian",
    }
    p_egg_allergy = {"allergies": ["eggs"], "dietary_restrictions": []}
    s_egg = safety.check_safety(rec_egg, p_egg_allergy)
    record("1.06 Egg allergy detected in Egg Bhurji", not s_egg["is_safe"])

    rec_shellfish = {
        "recipe_id": "t_shellfish_01",
        "recipe_name": "Goan Prawn Curry",
        "ingredients_raw": "300g fresh prawns, coconut milk, tamarind, red chillies",
        "ingredient_tokens": "fresh prawns coconut milk tamarind red chillies",
        "diet": "non_vegetarian",
    }
    p_shellfish_allergy = {"allergies": ["shellfish"], "dietary_restrictions": []}
    s_shellfish = safety.check_safety(rec_shellfish, p_shellfish_allergy)
    record("1.07 Shellfish allergy detected in Goan Prawn Curry", not s_shellfish["is_safe"])

    rec_soy = {
        "recipe_id": "t_soy_01",
        "recipe_name": "Soya Chunks Curry",
        "ingredients_raw": "1 cup soya chunks, onion tomato gravy, garam masala",
        "ingredient_tokens": "soya chunks onion tomato gravy garam masala",
        "diet": "vegetarian_high_protein",
    }
    p_soy_allergy = {"allergies": ["soy"], "dietary_restrictions": []}
    s_soy = safety.check_safety(rec_soy, p_soy_allergy)
    record("1.08 Soy allergy detected in Soya Chunks Curry", not s_soy["is_safe"])

    rec_sesame = {
        "recipe_id": "t_sesame_01",
        "recipe_name": "Til Ladoo",
        "ingredients_raw": "white sesame seeds (til), jaggery, cardamom",
        "ingredient_tokens": "white sesame seeds til jaggery cardamom",
        "diet": "vegetarian",
    }
    p_sesame_allergy = {"allergies": ["sesame"], "dietary_restrictions": []}
    s_sesame = safety.check_safety(rec_sesame, p_sesame_allergy)
    record("1.09 Sesame allergy detected in Til Ladoo", not s_sesame["is_safe"])

    # Safe recipe passes
    rec_safe = {
        "recipe_id": "t_safe_01",
        "recipe_name": "Steamed Rice with Moong Dal",
        "ingredients_raw": "rice, moong dal, turmeric, cumin, salt, water",
        "ingredient_tokens": "rice moong dal turmeric cumin salt water",
        "diet": "vegetarian",
    }
    s_safe = safety.check_safety(rec_safe, {"allergies": ["peanuts", "nuts", "dairy"], "dietary_restrictions": ["vegetarian"]})
    record("1.10 Safe recipe with no conflicts passes safety check", s_safe["is_safe"])

    # =========================================================================
    # Section 2: Safety Filter — Dietary Restrictions
    # =========================================================================
    print("\n--- Section 2: Dietary Restrictions Filtering ---")

    rec_chicken = {
        "recipe_id": "t_chicken_01",
        "recipe_name": "Chicken Tikka Masala",
        "ingredients_raw": "boneless chicken, yogurt, spices, ginger garlic paste",
        "ingredient_tokens": "boneless chicken yogurt spices ginger garlic paste",
        "diet": "non_vegetarian",
    }
    p_veg = {"allergies": [], "dietary_restrictions": ["vegetarian"]}
    s_veg_fail = safety.check_safety(rec_chicken, p_veg)
    record("2.01 Vegetarian restriction rejects chicken dish", not s_veg_fail["is_safe"])
    record("2.02 Conflict mentions VEGETARIAN violation", any("VEGETARIAN" in c for c in s_veg_fail["restriction_conflicts"]))

    # Vegan rejects dairy
    p_vegan = {"allergies": [], "dietary_restrictions": ["vegan"]}
    s_vegan_fail = safety.check_safety(rec_dairy, p_vegan)
    record("2.03 Vegan restriction rejects dairy recipe (Shahi Paneer)", not s_vegan_fail["is_safe"])

    # Vegan rejects eggs
    s_vegan_egg = safety.check_safety(rec_egg, p_vegan)
    record("2.04 Vegan restriction rejects egg recipe", not s_vegan_egg["is_safe"])

    # Jain restriction rejects onions and garlic
    rec_onion_garlic = {
        "recipe_id": "t_jain_fail",
        "recipe_name": "Aloo Pyaz Sabzi",
        "ingredients_raw": "potatoes (aloo), 2 sliced onions (pyaz), 5 cloves garlic (lehsun), cumin",
        "ingredient_tokens": "potatoes aloo sliced onions pyaz cloves garlic lehsun cumin",
        "diet": "vegetarian",
    }
    p_jain = {"allergies": [], "dietary_restrictions": ["jain"]}
    s_jain = safety.check_safety(rec_onion_garlic, p_jain)
    record("2.05 Jain restriction rejects root vegetables (aloo, pyaz, garlic)", not s_jain["is_safe"])

    # Gluten-free restriction rejects wheat/maida
    p_gf = {"allergies": [], "dietary_restrictions": ["gluten_free"]}
    s_gf = safety.check_safety(rec_gluten, p_gf)
    record("2.06 Gluten-free restriction rejects maida naan", not s_gf["is_safe"])

    # =========================================================================
    # Section 3: Incomplete Data Flagging & Mandatory Safety Override Rule
    # =========================================================================
    print("\n--- Section 3: Incomplete Data Flagging & Safety Override Guarantee ---")

    rec_incomplete = {
        "recipe_id": "t_incomp_01",
        "recipe_name": "Mystery Dish",
        "ingredients_raw": None,
        "ingredient_tokens": "",
        "diet": "vegetarian",
    }
    s_incomp = safety.check_safety(rec_incomplete, {"allergies": ["dairy"], "dietary_restrictions": []})
    record("3.01 Missing ingredient text correctly flagged as incomplete", s_incomp["has_incomplete_ingredients"])
    record("3.02 Incomplete ingredients with declared allergy fails safe check", not s_incomp["is_safe"])

    # MANDATORY SAFETY OVERRIDE TEST:
    # Even if recipe has favorable attributes (e.g. perfect cuisine, perfect time, high protein),
    # an allergen conflict MUST yield score = 0.0 and is_safe = False. High score NEVER overrides safety!
    perfect_but_dairy = {
        "recipe_id": "t_override_test",
        "recipe_name": "Delicious Malai Kofta",
        "cuisine": "North Indian",
        "course": "Main Course",
        "diet": "vegetarian_high_protein",
        "total_time_mins": 25.0,
        "complexity_level": "EASY",
        "ingredients_raw": "paneer, heavy malai, cashew paste, rich butter gravy",
        "ingredient_tokens": "paneer heavy malai cashew paste rich butter gravy",
    }
    p_dairy_strict = {
        "preferred_cuisines": ["North Indian"],
        "max_cooking_time_mins": 60,
        "dietary_goal": "muscle_gain",
        "budget": "high",
        "allergies": ["dairy"],  # Active allergy
        "dietary_restrictions": ["vegetarian"],
    }
    score_override = scorer.score_recipe(perfect_but_dairy, p_dairy_strict)
    record("3.03 Unsafe recipe score is STRICTLY 0.0", score_override["total_score"] == 0.0)
    record("3.04 Unsafe recipe is_safe is False", score_override["is_safe"] is False)
    record("3.05 Verdict is SAFETY_CONFLICT_EXCLUDED", score_override["verdict"] == "SAFETY_CONFLICT_EXCLUDED")
    record("3.06 Explanation explicitly states safety override rule", "Safety filters strictly override" in score_override["explanation"])
    record("3.07 All component scores are 0.0 for unsafe recipe", all(v == 0.0 for v in score_override["component_scores"].values()))

    # =========================================================================
    # Section 4: Transparent 0–100 Suitability Scorer
    # =========================================================================
    print("\n--- Section 4: Transparent 0-100 Suitability Scoring ---")

    rec_palak_dal = {
        "recipe_id": "t_palak_01",
        "recipe_name": "Palak Moong Dal",
        "cuisine": "North Indian",
        "course": "Lunch",
        "diet": "vegetarian_high_protein",
        "total_time_mins": 25.0,
        "complexity_level": "EASY",
        "ingredients_raw": "1 cup spinach leaves (palak), 1/2 cup yellow moong dal, cumin, turmeric, garlic, salt",
        "ingredient_tokens": "spinach leaves palak yellow moong dal cumin turmeric garlic salt",
    }
    p_healthy = {
        "preferred_cuisines": ["North Indian"],
        "max_cooking_time_mins": 30,
        "dietary_goal": "weight_loss",
        "budget": "low",
        "allergies": [],
        "dietary_restrictions": ["vegetarian"],
        "disliked_foods": [],
    }
    score_res = scorer.score_recipe(rec_palak_dal, p_healthy, content_sim=0.3)
    record("4.01 Safe recipe receives positive score", score_res["total_score"] > 50.0)
    record("4.02 Total score does not exceed 100", score_res["total_score"] <= 100.0)
    record("4.03 Verdict is HIGHLY_RECOMMENDED or RECOMMENDED", score_res["verdict"] in ("HIGHLY_RECOMMENDED", "RECOMMENDED"))
    record("4.04 Nutrient alignment component is <= 35.0", score_res["component_scores"]["nutrient_alignment"] <= 35.0)
    record("4.05 Preference alignment component is <= 25.0", score_res["component_scores"]["preference_alignment"] <= 25.0)
    record("4.06 Time feasibility component is <= 15.0", score_res["component_scores"]["time_feasibility"] <= 15.0)
    record("4.07 Budget & complexity component is <= 15.0", score_res["component_scores"]["budget_complexity"] <= 15.0)
    record("4.08 Goal synergy bonus is <= 10.0", score_res["component_scores"]["goal_synergy"] <= 10.0)
    record("4.09 Sum of components equals total score", abs(sum(score_res["component_scores"].values()) - score_res["total_score"]) < 0.2)
    record("4.10 Natural language explanation is provided", len(score_res["explanation"]) > 30)
    record("4.11 Non-clinical recommendation disclaimer attached", "not as medically validated" in score_res["disclaimer"])

    # Disliked food penalty
    p_dislikes_spinach = {**p_healthy, "disliked_foods": ["spinach"]}
    score_disliked = scorer.score_recipe(rec_palak_dal, p_dislikes_spinach)
    record("4.12 Disliked ingredient incurs score penalty", score_disliked["total_score"] < score_res["total_score"])

    # Cooking time overage penalty
    rec_long_cook = {**rec_palak_dal, "total_time_mins": 90.0}
    score_long = scorer.score_recipe(rec_long_cook, p_healthy)
    record("4.13 Excessive cooking time incurs penalty", score_long["component_scores"]["time_feasibility"] < score_res["component_scores"]["time_feasibility"])

    # =========================================================================
    # Section 5: Content-Based Filtering (TF-IDF & Cosine Similarity)
    # =========================================================================
    print("\n--- Section 5: Content-Based Recommender (TF-IDF) ---")

    c_recommender = ContentBasedRecommender()
    record("5.01 TF-IDF vectorizer initialized", c_recommender.vectorizer is not None)
    record("5.02 TF-IDF matrix populated", c_recommender.tfidf_matrix is not None and c_recommender.tfidf_matrix.shape[0] > 1000)

    sims = c_recommender.get_similar_recipes("biryani rice aromatic", top_k=5)
    record("5.03 Similar recipes returned for query", len(sims) > 0)
    record("5.04 Cosine similarity scores between 0 and 1", all(0.0 <= s <= 1.0 for _, s in sims))

    # Test top ranked recipe query
    rec_results = engine.recommend_recipes(p_healthy, course="lunch", top_n=5)
    record("5.05 Recommendation engine returns top 5 recipes", len(rec_results) == 5)
    record("5.06 Recommended recipes are sorted descending by score", rec_results[0]["suitability_score"] >= rec_results[-1]["suitability_score"])
    record("5.07 Recommended recipes contain suitability_score and explanation", "explanation" in rec_results[0] and "suitability_score" in rec_results[0])

    # =========================================================================
    # Section 6: Alternative Foods Engine
    # =========================================================================
    print("\n--- Section 6: Alternative Foods Engine ---")

    alt_samosa = alt_engine.get_alternatives("samosa")
    record("6.01 Returns alternatives for samosa", len(alt_samosa) >= 2)
    record("6.02 Samosa alternatives contain baked/steamed options", any("baked" in a["alternative_name"].lower() or "dhokla" in a["alternative_name"].lower() for a in alt_samosa))
    record("6.03 Benefit explanation provided for alternative", len(alt_samosa[0]["benefit_explanation"]) > 20)
    record("6.04 Estimated calories provided where available", alt_samosa[0]["estimated_calories"] is not None)

    alt_naan = alt_engine.get_alternatives("butter naan")
    record("6.05 Returns alternatives for butter naan", len(alt_naan) >= 1)
    record("6.06 Naan alternative is whole wheat / roti", any("roti" in a["alternative_name"].lower() or "phulka" in a["alternative_name"].lower() for a in alt_naan))

    alt_jalebi = alt_engine.get_alternatives("jalebi")
    record("6.07 Returns alternatives for jalebi", len(alt_jalebi) >= 1)

    alt_biryani = alt_engine.get_alternatives("biryani")
    record("6.08 Returns alternatives for biryani", len(alt_biryani) >= 1)

    alt_unknown = alt_engine.get_alternatives("unknown_fried_fritter")
    record("6.09 Graceful fallback for unlisted food", len(alt_unknown) > 0 and "Steamed" in alt_unknown[0]["alternative_name"])

    # =========================================================================
    # Section 7: Ingredient Substitution Engine
    # =========================================================================
    print("\n--- Section 7: Ingredient Substitution Engine ---")

    subs_milk = sub_engine.get_substitutions_for_ingredient("milk")
    record("7.01 Returns milk substitutions", len(subs_milk) >= 2)
    record("7.02 Milk substitutes include almond or soy milk", any("almond" in s["substitute"].lower() or "soy" in s["substitute"].lower() for s in subs_milk))
    record("7.03 Substitution ratio provided (e.g. 1:1)", bool(subs_milk[0]["ratio"]))

    subs_paneer = sub_engine.get_substitutions_for_ingredient("paneer")
    record("7.04 Paneer substitutes include tofu", any("tofu" in s["substitute"].lower() for s in subs_paneer))

    subs_maida = sub_engine.get_substitutions_for_ingredient("maida")
    record("7.05 Maida substitutes include atta or ragi", any("atta" in s["substitute"].lower() or "ragi" in s["substitute"].lower() for s in subs_maida))

    subs_sugar = sub_engine.get_substitutions_for_ingredient("sugar")
    record("7.06 Sugar substitutes include jaggery or stevia", any("jaggery" in s["substitute"].lower() or "stevia" in s["substitute"].lower() for s in subs_sugar))

    subs_cashew = sub_engine.get_substitutions_for_ingredient("cashew")
    record("7.07 Cashew substitutes include melon or sunflower seeds", any("seed" in s["substitute"].lower() for s in subs_cashew))

    # Recipe-level substitution scan
    rec_sweet_chai = {
        "recipe_id": "t_chai_01",
        "recipe_name": "Masala Chai with Milk and Sugar",
        "ingredients_raw": "water, tea leaves, 1 cup buffalo milk, 2 tbsp white sugar, cardamom",
    }
    p_vegan_diabetic = {"allergies": [], "dietary_restrictions": ["vegan", "manage_diabetes"]}
    recipe_swaps = sub_engine.suggest_recipe_substitutions(rec_sweet_chai, p_vegan_diabetic)
    record("7.08 Recipe scan identifies milk swap for vegan user", any(s["ingredient_found"] == "milk" for s in recipe_swaps["swaps"]))
    record("7.09 Recipe scan identifies sugar swap for diabetic user", any(s["ingredient_found"] == "sugar" for s in recipe_swaps["swaps"]))
    record("7.10 Total identified swaps count matches list length", recipe_swaps["total_swaps_identified"] == len(recipe_swaps["swaps"]))

    # =========================================================================
    # Section 8: Daily Meal Planning
    # =========================================================================
    print("\n--- Section 8: Daily Meal Planning ---")

    daily_plan = engine.generate_daily_meal_plan(p_healthy, target_calories_kcal=1900.0)
    record("8.01 Daily plan has daily_target_calories_kcal", daily_plan["daily_target_calories_kcal"] == 1900.0)
    record("8.02 Daily plan contains exactly 4 meals", set(daily_plan["meals"].keys()) == {"breakfast", "lunch", "snack", "dinner"})
    record("8.03 Breakfast meal has target calories (~25% = 475)", abs(daily_plan["meals"]["breakfast"]["target_calories_kcal"] - 475.0) < 5.0)
    record("8.04 Lunch meal has target calories (~35% = 665)", abs(daily_plan["meals"]["lunch"]["target_calories_kcal"] - 665.0) < 5.0)
    record("8.05 Snack meal has target calories (~10% = 190)", abs(daily_plan["meals"]["snack"]["target_calories_kcal"] - 190.0) < 5.0)
    record("8.06 Dinner meal has target calories (~30% = 570)", abs(daily_plan["meals"]["dinner"]["target_calories_kcal"] - 570.0) < 5.0)
    record("8.07 Average suitability score present and > 50", daily_plan["average_suitability_score"] > 50.0)
    record("8.08 Total prep/cook time calculated", daily_plan["total_prep_cook_time_mins"] > 0)
    record("8.09 Safety notice included in daily plan", "Safety Rule Enforced" in daily_plan["safety_notice"])

    # Strict allergy in daily plan
    p_peanut_vegan = {
        "allergies": ["peanuts", "dairy"],
        "dietary_restrictions": ["vegan"],
        "preferred_cuisines": ["South Indian"],
        "max_cooking_time_mins": 45,
    }
    daily_allergy_plan = engine.generate_daily_meal_plan(p_peanut_vegan, target_calories_kcal=1700.0)
    for m_name, meal in daily_allergy_plan["meals"].items():
        # Verify no allergens
        m_safety = safety.check_safety(meal, p_peanut_vegan)
        record(f"8.10 Daily plan {m_name} complies with strict allergy/vegan filter", m_safety["is_safe"])

    # =========================================================================
    # Section 9: Weekly Meal Planning
    # =========================================================================
    print("\n--- Section 9: Weekly Meal Planning ---")

    weekly_plan = engine.generate_weekly_meal_plan(p_healthy, target_calories_kcal=2000.0)
    record("9.01 Weekly plan type is 7_DAY_WEEKLY_PLAN", weekly_plan["plan_type"] == "7_DAY_WEEKLY_PLAN")
    record("9.02 Weekly schedule contains 7 days", len(weekly_plan["weekly_schedule"]) == 7)
    days_expected = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    record("9.03 All 7 day names present", all(d in weekly_plan["weekly_schedule"] for d in days_expected))

    mon = weekly_plan["weekly_schedule"]["Monday"]
    tue = weekly_plan["weekly_schedule"]["Tuesday"]
    record("9.04 Monday has 4 meals", set(mon.keys()).issuperset({"breakfast", "lunch", "snack", "dinner"}))
    record("9.05 Variety control: Monday and Tuesday lunches differ", mon["lunch"]["recipe_id"] != tue["lunch"]["recipe_id"])
    record("9.06 Variety control: Monday and Tuesday breakfasts differ", mon["breakfast"]["recipe_id"] != tue["breakfast"]["recipe_id"])
    record("9.07 Grocery shopping list generated", len(weekly_plan["grocery_shopping_list"]) > 0)
    record("9.08 Grocery item has ingredient and frequency", "ingredient" in weekly_plan["grocery_shopping_list"][0] and "frequency_in_week" in weekly_plan["grocery_shopping_list"][0])
    record("9.09 Overall weekly suitability score is > 50", weekly_plan["overall_weekly_suitability_score"] > 50.0)

    # =========================================================================
    # Section 10: Web REST API Endpoints
    # =========================================================================
    print("\n--- Section 10: Web REST API Endpoints ---")

    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Create a temporary profile for API tests
    test_user = create_profile({
        "display_name": "API Rec Test User",
        "age": 30,
        "gender": "male",
        "height_cm": 175.0,
        "weight_kg": 72.0,
        "activity_level": "moderately_active",
        "dietary_goal": "muscle_gain",
        "preferred_cuisines": ["North Indian", "South Indian"],
        "allergies": ["peanuts"],
        "dietary_restrictions": ["vegetarian"],
        "max_cooking_time_mins": 45,
        "budget": "medium"
    })
    uid = test_user["user_id"]

    # POST /api/recommendations/recipes with user_id
    resp_rec = client.post("/api/recommendations/recipes", json={"user_id": uid, "course": "lunch", "top_n": 4})
    record("10.01 /api/recommendations/recipes returns 200", resp_rec.status_code == 200)
    rec_json = resp_rec.get_json()
    record("10.02 API returns success=True and data list", rec_json["success"] is True and len(rec_json["data"]) == 4)
    record("10.03 API response contains disclaimer and safety notice", "disclaimer" in rec_json and "safety_notice" in rec_json)

    # POST /api/recommendations/recipes with embedded profile
    resp_emb = client.post("/api/recommendations/recipes", json={"user_profile": {"dietary_restrictions": ["vegetarian"]}, "top_n": 3})
    record("10.04 Recipes endpoint accepts embedded user_profile dict", resp_emb.status_code == 200 and len(resp_emb.get_json()["data"]) == 3)

    # POST /api/recommendations/alternatives
    resp_alt = client.post("/api/recommendations/alternatives", json={"food_name": "samosa", "user_id": uid})
    record("10.05 /api/recommendations/alternatives returns 200", resp_alt.status_code == 200)
    alt_json = resp_alt.get_json()
    record("10.06 Alternatives data count > 0", alt_json["count"] > 0)

    # POST /api/recommendations/alternatives error (missing food_name)
    resp_alt_err = client.post("/api/recommendations/alternatives", json={})
    record("10.07 Alternatives returns 400 when food_name missing", resp_alt_err.status_code == 400)

    # POST /api/recommendations/substitutions (ingredient)
    resp_sub1 = client.post("/api/recommendations/substitutions", json={"ingredient": "milk", "reason": "dairy"})
    record("10.08 /api/recommendations/substitutions (ingredient) returns 200", resp_sub1.status_code == 200)
    sub1_json = resp_sub1.get_json()
    record("10.09 Substitutions data non-empty", len(sub1_json["data"]) > 0)

    # POST /api/recommendations/substitutions (recipe)
    resp_sub2 = client.post("/api/recommendations/substitutions", json={"recipe": rec_dairy, "user_id": uid})
    record("10.10 /api/recommendations/substitutions (recipe) returns 200", resp_sub2.status_code == 200)

    # POST /api/recommendations/substitutions error
    resp_sub_err = client.post("/api/recommendations/substitutions", json={})
    record("10.11 Substitutions returns 400 when neither ingredient nor recipe provided", resp_sub_err.status_code == 400)

    # POST /api/recommendations/score-recipe
    resp_score = client.post("/api/recommendations/score-recipe", json={"recipe": rec_palak_dal, "user_id": uid})
    record("10.12 /api/recommendations/score-recipe returns 200", resp_score.status_code == 200)
    score_json = resp_score.get_json()
    record("10.13 Recipe score has total_score and component_scores", "total_score" in score_json["data"] and "component_scores" in score_json["data"])

    # POST /api/recommendations/daily-plan
    resp_daily = client.post("/api/recommendations/daily-plan", json={"user_id": uid, "target_calories_kcal": 2100.0})
    record("10.14 /api/recommendations/daily-plan returns 200", resp_daily.status_code == 200)
    daily_json = resp_daily.get_json()
    record("10.15 Daily plan data has 4 meals and average score", len(daily_json["data"]["meals"]) == 4 and "average_suitability_score" in daily_json["data"])

    # POST /api/recommendations/weekly-plan
    resp_weekly = client.post("/api/recommendations/weekly-plan", json={"user_id": uid})
    record("10.16 /api/recommendations/weekly-plan returns 200", resp_weekly.status_code == 200)
    weekly_json = resp_weekly.get_json()
    record("10.17 Weekly plan has 7-day schedule and grocery shopping list", len(weekly_json["data"]["weekly_schedule"]) == 7 and "grocery_shopping_list" in weekly_json["data"])

    # Nonexistent user_id returns 404
    resp_404 = client.post("/api/recommendations/daily-plan", json={"user_id": "nonexistent_uid_12345"})
    record("10.18 Nonexistent user_id returns 404", resp_404.status_code == 404)

    # Clean up test profile
    delete_profile(uid)

    # =========================================================================
    # Section 11: Edge Cases & Integrity
    # =========================================================================
    print("\n--- Section 11: Edge Cases & Integrity ---")

    # Empty user profile
    rec_empty_prof = engine.recommend_recipes({})
    record("11.01 Recommend recipes works with empty profile dictionary", len(rec_empty_prof) > 0)

    # Unknown dietary goal
    score_unknown_goal = scorer.score_recipe(rec_palak_dal, {"dietary_goal": "unknown_goal_xyz"})
    record("11.02 Scorer handles unknown dietary goal gracefully", score_unknown_goal["is_safe"] is True and score_unknown_goal["total_score"] > 0)

    # Score bounds strictly within 0-100
    all_scores = [r["suitability_score"] for r in rec_empty_prof]
    record("11.03 All candidate scores strictly bounded in [0, 100]", all(0.0 <= s <= 100.0 for s in all_scores))

    # Weekly plan total prep time positive
    record("11.04 Weekly plan overall suitability score > 0", weekly_plan["overall_weekly_suitability_score"] > 0)

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 70)
    print(f"PHASE 9 RECOMMENDATION SYSTEM TESTS: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 70)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} test(s) failed in test_recommendation_system.py")


if __name__ == "__main__":
    run_tests()

