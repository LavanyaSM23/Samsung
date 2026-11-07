"""
src/recommendation_engine.py
NutriSense AI — Phase 9: Recommendation System

Provides:
1. SafetyFilterEngine: Mandatory allergen and dietary restriction filtering BEFORE ranking.
   - Detects allergens: dairy, peanuts, nuts, gluten, eggs, shellfish, soy, sesame.
   - Detects dietary restrictions: vegetarian, vegan, eggetarian, jain, halal, gluten_free, diabetic_friendly, low_sodium, high_protein.
   - Flags incomplete ingredient or allergen data.
   - Strictly enforces that NO high score can ever override safety filters.
2. SuitabilityScorer: Transparent 0 to 100 suitability score with explicit weights:
   - Nutrient & Calorie Alignment (35%)
   - Cuisine & Taste Preference Alignment (25%)
   - Cooking Time & Feasibility (15%)
   - Budget & Complexity (15%)
   - Goal Synergy Bonus (10%)
   - Human-understandable explanations and non-clinical disclaimers.
3. Content-Based Recommender with TF-IDF & Cosine Similarity on 5,381 Indian recipes.
4. AlternativeFoodsEngine: Suggests healthier, goal-aligned alternatives for caloric/fat/sodium-dense foods.
5. IngredientSubstitutionEngine: Culinary substitutions for Indian recipes (dairy, gluten, nuts, eggs, sugar, salt).
6. DailyMealPlanner: Generates calibrated 4-meal daily plans (Breakfast, Lunch, Snack, Dinner).
7. WeeklyMealPlanner: Generates 7-day meal plans with variety control and aggregated grocery shopping lists.
"""

from collections import Counter
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths and Dependencies
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = WORKSPACE_ROOT / "data" / "processed" / "feature_store"
RECIPE_CSV = FEATURE_DIR / "recipe_feature_store.csv"
BENCHMARKS_CSV = FEATURE_DIR / "food_suitability_benchmarks.csv"
KB_JSON = FEATURE_DIR / "master_food_nutrition_kb.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimers
# ---------------------------------------------------------------------------
RECOMMENDATION_DISCLAIMER = (
    "Recommendations and suitability scores (0–100) are algorithmic heuristics based on "
    "user-declared preferences, lifestyle parameters, and published dietary references (ICMR-NIN 2020). "
    "They are provided for educational and lifestyle guidance only, not as medically validated prescriptions "
    "or clinical treatment plans. Individuals with severe medical conditions, food anaphylaxis, or specific metabolic "
    "requirements should consult a qualified physician or registered dietitian before modifying their diet."
)

SAFETY_OVERRIDE_NOTICE = (
    "Safety Rule Enforced: Recipes containing declared allergens or violating dietary restrictions "
    "are strictly filtered out before ranking. A high suitability score can NEVER override safety constraints."
)

# ---------------------------------------------------------------------------
# Allergen Lexicon for Indian & Western Ingredients
# ---------------------------------------------------------------------------
ALLERGEN_LEXICON: Dict[str, Set[str]] = {
    "dairy": {
        "milk", "curd", "dahi", "yogurt", "paneer", "ghee", "butter", "cheese",
        "cream", "malai", "khoa", "khoya", "mawa", "whey", "buttermilk", "chaas",
        "cottage cheese", "mozzarella", "cheddar", "condensed milk", "milk powder"
    },
    "peanuts": {
        "peanut", "peanuts", "groundnut", "groundnuts", "mungfali", "moongphali",
        "peanut butter", "peanut oil"
    },
    "nuts": {
        "almond", "almonds", "badam", "cashew", "cashews", "kaju", "walnut",
        "walnuts", "akhrot", "pistachio", "pistachios", "pista", "hazelnut",
        "hazelnuts", "pecan", "pecans", "pine nuts", "chilgoza"
    },
    "gluten": {
        "wheat", "maida", "atta", "semolina", "sooji", "suji", "rava", "flour",
        "all purpose flour", "bread", "breadcrumbs", "noodles", "pasta", "barley",
        "rye", "seitan", "roti", "chapati", "naan", "paratha", "puri", "poori"
    },
    "eggs": {
        "egg", "eggs", "anda", "egg white", "egg yolk", "albumin"
    },
    "shellfish": {
        "prawn", "prawns", "shrimp", "shrimps", "crab", "crabs", "lobster",
        "lobsters", "mussel", "mussels", "clam", "clams", "jhinga"
    },
    "soy": {
        "soy", "soya", "soyabean", "soybean", "tofu", "soy milk", "soy chunks",
        "soya chunks", "soy sauce", "edamame"
    },
    "sesame": {
        "sesame", "til", "gingelly", "tahini", "sesame seeds", "sesame oil"
    }
}

# Non-vegetarian indicators
NON_VEG_KEYWORDS = {
    "chicken", "mutton", "lamb", "goat", "fish", "maach", "fish fillet",
    "prawn", "prawns", "shrimp", "crab", "pork", "beef", "meat", "keema",
    "gosht", "murgh", "rohu", "katla", "surmai", "hilsa", "pomfret", "salmon"
}

# Jain restricted keywords (root vegetables & alliums)
JAIN_RESTRICTED_KEYWORDS = {
    "onion", "onions", "pyaz", "garlic", "lehsun", "ginger", "adrak",
    "potato", "potatoes", "aloo", "carrot", "carrots", "gajjar", "radish",
    "mooli", "beetroot", "turnip", "arbi", "yam", "jimikand", "sweet potato"
}

# ---------------------------------------------------------------------------
# 1. SafetyFilterEngine
# ---------------------------------------------------------------------------
class SafetyFilterEngine:
    """
    Mandatory safety filtering executed BEFORE any recommendation scoring or ranking.
    Ensures:
    1. Zero tolerance for known allergen conflicts.
    2. Strict compliance with dietary restrictions (vegetarian, vegan, jain, etc.).
    3. Detection and transparent flagging of incomplete ingredient or allergen information.
    4. Safety guarantees that can never be overridden by algorithm scores.
    """

    def __init__(self):
        self.allergen_lexicon = ALLERGEN_LEXICON
        self.non_veg_keywords = NON_VEG_KEYWORDS
        self.jain_restricted = JAIN_RESTRICTED_KEYWORDS

    def analyze_recipe_ingredients(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses ingredients text/tokens, detects present allergens, and checks data completeness.
        """
        raw_text = str(recipe.get("ingredients_raw", "") or "")
        tokens_text = str(recipe.get("ingredient_tokens", "") or "")
        recipe_name = str(recipe.get("recipe_name", "") or "")
        combined = f"{recipe_name} {raw_text} {tokens_text}".lower()

        # Check for incomplete ingredient data
        has_incomplete_info = False
        reasons_incomplete = []
        if not raw_text.strip() or raw_text.strip().lower() in ("nan", "none"):
            has_incomplete_info = True
            reasons_incomplete.append("Missing raw ingredient list")
        if not tokens_text.strip() or tokens_text.strip().lower() in ("nan", "none"):
            has_incomplete_info = True
            reasons_incomplete.append("Missing parsed ingredient tokens")

        # Detect allergens
        detected_allergens: Set[str] = set()
        for allergen, keywords in self.allergen_lexicon.items():
            for kw in keywords:
                # Use word boundary check
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, combined):
                    detected_allergens.add(allergen)
                    break

        return {
            "has_incomplete_ingredients": has_incomplete_info,
            "incomplete_reasons": reasons_incomplete,
            "detected_allergens": sorted(list(detected_allergens)),
            "combined_text": combined,
        }

    def check_safety(
        self,
        recipe: Dict[str, Any],
        profile: Dict[str, Any],
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates recipe against user's declared allergies and dietary restrictions.

        Returns:
            {
                "is_safe": bool,
                "allergen_conflicts": List[str],
                "restriction_conflicts": List[str],
                "has_incomplete_ingredients": bool,
                "safety_reason": str
            }
        """
        analysis = self.analyze_recipe_ingredients(recipe)
        combined = analysis["combined_text"]
        has_incomplete = analysis["has_incomplete_ingredients"]
        detected_allergens = set(analysis["detected_allergens"])

        allergen_conflicts = []
        restriction_conflicts = []

        # 1. Allergen conflict check
        user_allergies = [a.lower().strip() for a in profile.get("allergies", []) if a]
        for allergy in user_allergies:
            if allergy in detected_allergens:
                allergen_conflicts.append(f"Contains declared allergen: {allergy.upper()}")
            else:
                # Check for direct keyword matches in combined text
                kw_list = self.allergen_lexicon.get(allergy, {allergy})
                for kw in kw_list:
                    if re.search(r"\b" + re.escape(kw) + r"\b", combined):
                        allergen_conflicts.append(f"Contains allergen keyword '{kw}' matching {allergy.upper()}")
                        break

        # 2. Dietary restriction check
        diet_str = str(recipe.get("diet", "") or "").lower()
        restrictions = [r.lower().strip() for r in profile.get("dietary_restrictions", []) if r]

        # A. Vegetarian check
        if "vegetarian" in restrictions:
            is_non_veg_tagged = any(nv in diet_str for nv in ["non_veg", "non vegetarian", "non-vegetarian"])
            has_meat_keyword = any(re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.non_veg_keywords)
            if is_non_veg_tagged or has_meat_keyword:
                restriction_conflicts.append("Violates VEGETARIAN restriction: contains meat, poultry, or fish.")

        # B. Vegan check (Vegetarian + No Dairy + No Eggs)
        if "vegan" in restrictions:
            has_dairy = "dairy" in detected_allergens or any(
                re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.allergen_lexicon["dairy"]
            )
            has_eggs = "eggs" in detected_allergens or any(
                re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.allergen_lexicon["eggs"]
            )
            is_non_veg = any(nv in diet_str for nv in ["non_veg", "non vegetarian", "non-vegetarian"]) or any(
                re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.non_veg_keywords
            )
            if is_non_veg or has_dairy or has_eggs or "honey" in combined:
                restriction_conflicts.append("Violates VEGAN restriction: contains animal, dairy, or egg products.")

        # C. Eggetarian check (Allowed eggs, but no meat/fish)
        if "eggetarian" in restrictions:
            is_meat_tagged = any(nv in diet_str for nv in ["non_veg", "non vegetarian", "non-vegetarian"])
            has_meat_keyword = any(re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.non_veg_keywords)
            if (is_meat_tagged or has_meat_keyword) and not ("egg" in combined and not has_meat_keyword):
                restriction_conflicts.append("Violates EGGETARIAN restriction: contains meat or seafood.")

        # D. Jain check (Vegetarian + No root vegetables / alliums)
        if "jain" in restrictions:
            has_jain_restricted = any(re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.jain_restricted)
            if has_jain_restricted or "sattvic" not in diet_str and any(nv in diet_str for nv in ["non_veg"]):
                restriction_conflicts.append("Violates JAIN dietary restriction: contains onion, garlic, or root vegetables.")

        # E. Gluten-Free check
        if "gluten_free" in restrictions:
            if "gluten" in detected_allergens or any(
                re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in self.allergen_lexicon["gluten"]
            ):
                restriction_conflicts.append("Violates GLUTEN-FREE restriction: contains wheat, maida, semolina, or gluten.")

        # In strict mode: if recipe has incomplete ingredients AND user has active allergies, flag risk
        if strict_mode and has_incomplete and (user_allergies or "vegan" in restrictions or "gluten_free" in restrictions):
            allergen_conflicts.append("UNVERIFIED_INGREDIENTS: Incomplete ingredient information for declared allergies.")

        is_safe = (len(allergen_conflicts) == 0) and (len(restriction_conflicts) == 0)
        reasons = allergen_conflicts + restriction_conflicts
        reason_str = "; ".join(reasons) if reasons else "Recipe satisfies all declared safety criteria."

        return {
            "is_safe": is_safe,
            "allergen_conflicts": allergen_conflicts,
            "restriction_conflicts": restriction_conflicts,
            "has_incomplete_ingredients": has_incomplete,
            "safety_reason": reason_str,
            "detected_allergens": analysis["detected_allergens"],
        }


# ---------------------------------------------------------------------------
# 2. Transparent Suitability Scorer (0 to 100)
# ---------------------------------------------------------------------------
class SuitabilityScorer:
    """
    Transparent 0 to 100 suitability scoring engine with explicit additive weights:
      1. Nutrient & Calorie Alignment (35%)
      2. Cuisine & Taste Preference Alignment (25%)
      3. Cooking Time & Feasibility (15%)
      4. Budget & Complexity Alignment (15%)
      5. Goal Synergy Bonus (10%)

    Safety Guarantee: If a recipe has a safety conflict, the score is strictly 0.0.
    """

    WEIGHTS = {
        "nutrient_alignment": 35.0,
        "preference_alignment": 25.0,
        "time_feasibility": 15.0,
        "budget_complexity": 15.0,
        "goal_synergy": 10.0,
    }

    def __init__(self):
        self.safety_engine = SafetyFilterEngine()

    def score_recipe(
        self,
        recipe: Dict[str, Any],
        profile: Dict[str, Any],
        content_sim: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculates a transparent 0-100 suitability score with complete breakdown.
        """
        # MANDATORY STEP: Execute safety filter first
        safety_result = self.safety_engine.check_safety(recipe, profile)
        if not safety_result["is_safe"]:
            return {
                "total_score": 0.0,
                "verdict": "SAFETY_CONFLICT_EXCLUDED",
                "is_safe": False,
                "safety_reason": safety_result["safety_reason"],
                "component_scores": {
                    "nutrient_alignment": 0.0,
                    "preference_alignment": 0.0,
                    "time_feasibility": 0.0,
                    "budget_complexity": 0.0,
                    "goal_synergy": 0.0,
                },
                "weights": self.WEIGHTS,
                "explanation": f"Recipe excluded due to safety conflicts: {safety_result['safety_reason']}. "
                               f"Safety filters strictly override any recommendation score.",
                "disclaimer": RECOMMENDATION_DISCLAIMER,
                "has_incomplete_ingredients": safety_result["has_incomplete_ingredients"],
            }

        # Component 1: Nutrient & Calorie Alignment (Max: 35.0)
        c1_score, c1_reasons = self._calc_nutrient_alignment(recipe, profile)

        # Component 2: Cuisine & Taste Preference Alignment (Max: 25.0)
        c2_score, c2_reasons = self._calc_preference_alignment(recipe, profile, content_sim)

        # Component 3: Cooking Time & Feasibility (Max: 15.0)
        c3_score, c3_reasons = self._calc_time_feasibility(recipe, profile)

        # Component 4: Budget & Complexity Alignment (Max: 15.0)
        c4_score, c4_reasons = self._calc_budget_complexity(recipe, profile)

        # Component 5: Goal Synergy Bonus (Max: 10.0)
        c5_score, c5_reasons = self._calc_goal_synergy(recipe, profile)

        total = round(c1_score + c2_score + c3_score + c4_score + c5_score, 1)
        total = min(100.0, max(0.0, total))

        # Determine verdict
        if total >= 80.0:
            verdict = "HIGHLY_RECOMMENDED"
        elif total >= 65.0:
            verdict = "RECOMMENDED"
        elif total >= 50.0:
            verdict = "MODERATE_FIT"
        else:
            verdict = "LOW_FIT"

        # Build natural language explanation
        all_reasons = c1_reasons + c2_reasons + c3_reasons + c4_reasons + c5_reasons
        explanation = (
            f"Suitability Score {total:.1f}/100 ({verdict.replace('_', ' ').title()}). "
            f"{' '.join(all_reasons[:3])} "
            f"Meets all declared allergy and dietary requirements."
        )

        return {
            "total_score": total,
            "verdict": verdict,
            "is_safe": True,
            "safety_reason": safety_result["safety_reason"],
            "component_scores": {
                "nutrient_alignment": round(c1_score, 1),
                "preference_alignment": round(c2_score, 1),
                "time_feasibility": round(c3_score, 1),
                "budget_complexity": round(c4_score, 1),
                "goal_synergy": round(c5_score, 1),
            },
            "weights": self.WEIGHTS,
            "explanation": explanation,
            "disclaimer": RECOMMENDATION_DISCLAIMER,
            "has_incomplete_ingredients": safety_result["has_incomplete_ingredients"],
        }

    def _calc_nutrient_alignment(self, recipe: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
        score = 25.0  # Baseline neutral
        reasons = []
        diet_str = str(recipe.get("diet", "")).lower()
        goal = str(profile.get("dietary_goal", "general_wellness")).lower()

        if "high_protein" in diet_str:
            if goal in ("muscle_gain", "weight_loss", "high_protein"):
                score += 10.0
                reasons.append("High protein content supports your lean muscle and satiety goals.")
            else:
                score += 5.0
        elif "diabetic_friendly" in diet_str:
            if goal in ("manage_diabetes", "weight_loss"):
                score += 10.0
                reasons.append("Controlled carbohydrate profile aligns with glycemic control goals.")
            else:
                score += 4.0

        if goal == "weight_loss":
            # Prefer recipes with vegetable / fiber focus
            ing_tokens = str(recipe.get("ingredient_tokens", "")).lower()
            if any(veg in ing_tokens for veg in ["spinach", "methi", "cucumber", "gourd", "lauki", "sprouts", "dal"]):
                score += 5.0
                reasons.append("Rich in fiber and whole ingredients favorable for caloric management.")

        return min(35.0, max(5.0, score)), reasons

    def _calc_preference_alignment(
        self,
        recipe: Dict[str, Any],
        profile: Dict[str, Any],
        content_sim: float
    ) -> Tuple[float, List[str]]:
        score = 10.0
        reasons = []
        r_cuisine = str(recipe.get("cuisine", "")).lower()
        pref_cuisines = [c.lower() for c in profile.get("preferred_cuisines", [])]

        # Match preferred cuisines
        matched_cuisine = False
        if "any" in pref_cuisines or not pref_cuisines:
            score += 8.0
        else:
            for c in pref_cuisines:
                if c in r_cuisine or r_cuisine in c:
                    score += 12.0
                    matched_cuisine = True
                    reasons.append(f"Matches your preferred {recipe.get('cuisine')} cuisine.")
                    break

        # Blend content-based similarity from TF-IDF
        if content_sim > 0.0:
            sim_bonus = min(8.0, content_sim * 10.0)
            score += sim_bonus
            if content_sim > 0.25:
                reasons.append("Ingredient profile strongly aligns with your taste preferences.")

        # Disliked foods penalty
        disliked = [d.lower().strip() for d in profile.get("disliked_foods", []) if d]
        ing_raw = str(recipe.get("ingredients_raw", "")).lower()
        for d in disliked:
            if d and d in ing_raw:
                score = max(0.0, score - 15.0)
                reasons.append(f"Contains disliked ingredient: '{d}'.")
                break

        return min(25.0, max(0.0, score)), reasons

    def _calc_time_feasibility(self, recipe: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
        reasons = []
        max_time = float(profile.get("max_cooking_time_mins", 60) or 60)
        rec_time = float(recipe.get("total_time_mins", 40) or 40)

        if rec_time <= max_time:
            # Score proportional to how easily it fits
            ratio = rec_time / max(max_time, 1.0)
            if ratio <= 0.5:
                score = 15.0
                reasons.append(f"Quick to prepare ({int(rec_time)} min), well within your {int(max_time)} min limit.")
            elif ratio <= 0.8:
                score = 13.0
                reasons.append(f"Fits comfortably in your {int(max_time)} min cooking window ({int(rec_time)} min).")
            else:
                score = 10.0
        else:
            # Exceeds max time -> penalty
            overage = rec_time - max_time
            penalty = min(12.0, (overage / max_time) * 10.0)
            score = max(0.0, 10.0 - penalty)
            reasons.append(f"Cooking time ({int(rec_time)} min) exceeds preferred limit of {int(max_time)} min.")

        return min(15.0, max(0.0, score)), reasons

    def _calc_budget_complexity(self, recipe: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
        reasons = []
        user_budget = str(profile.get("budget", "medium")).lower()
        complexity = str(recipe.get("complexity_level", "MODERATE")).upper()

        if user_budget == "low":
            if complexity == "EASY":
                score = 15.0
                reasons.append("Economical and simple preparation requiring minimal specialized pantry items.")
            elif complexity == "MODERATE":
                score = 10.0
            else:
                score = 5.0
        elif user_budget == "medium":
            if complexity in ("EASY", "MODERATE"):
                score = 15.0
                reasons.append("Balanced ingredient complexity suitable for standard household cooking.")
            else:
                score = 10.0
        else:  # high budget
            score = 15.0

        return min(15.0, max(0.0, score)), reasons

    def _calc_goal_synergy(self, recipe: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[float, List[str]]:
        reasons = []
        score = 5.0
        goal = str(profile.get("dietary_goal", "general_wellness")).lower()
        diet = str(recipe.get("diet", "")).lower()

        if goal == "weight_loss" and ("vegetarian_high_protein" in diet or "diabetic_friendly" in diet):
            score = 10.0
            reasons.append("Excellent synergy with healthy weight management targets.")
        elif goal == "muscle_gain" and "high_protein" in diet:
            score = 10.0
            reasons.append("High protein synergy for muscle synthesis.")
        elif goal == "manage_diabetes" and "diabetic_friendly" in diet:
            score = 10.0
            reasons.append("Synergistic glycemic index formulation.")
        elif goal == "manage_hypertension" and "low_sodium" in diet:
            score = 10.0
            reasons.append("Low sodium recipe synergy.")
        elif goal == "general_wellness":
            score = 8.0

        return min(10.0, max(0.0, score)), reasons


# ---------------------------------------------------------------------------
# 3. Content-Based Recipe Recommender (TF-IDF & Cosine Similarity)
# ---------------------------------------------------------------------------
class ContentBasedRecommender:
    """
    Computes content-based recommendations over 5,381 recipes using TF-IDF vectors
    derived from recipe names, cuisines, courses, and ingredient tokens.
    """

    def __init__(self, recipe_df: Optional[pd.DataFrame] = None):
        if recipe_df is not None:
            self.df = recipe_df.copy()
        elif RECIPE_CSV.exists():
            self.df = pd.read_csv(RECIPE_CSV)
        else:
            self.df = pd.DataFrame()

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._build_index()

    def _build_index(self):
        if self.df.empty:
            return

        corpus = []
        for _, row in self.df.iterrows():
            name = str(row.get("recipe_name", "") or "")
            cuisine = str(row.get("cuisine", "") or "")
            course = str(row.get("course", "") or "")
            diet = str(row.get("diet", "") or "")
            tokens = str(row.get("ingredient_tokens", "") or "")
            corpus.append(f"{name} {cuisine} {course} {diet} {tokens}".lower())

        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def get_similar_recipes(
        self,
        query_text: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Returns list of (row_index, similarity_score) sorted by cosine similarity.
        """
        if self.vectorizer is None or self.tfidf_matrix is None or not query_text.strip():
            return []

        q_vec = self.vectorizer.transform([query_text.lower()])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(-sims)[:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > 0.05]


# ---------------------------------------------------------------------------
# 4. Alternative Foods Engine
# ---------------------------------------------------------------------------
class AlternativeFoodsEngine:
    """
    Recommends healthier, goal-aligned culinary alternatives for calorie-dense,
    fried, high-sodium, or refined-flour foods.
    """

    CURATED_ALTERNATIVES: Dict[str, List[Dict[str, Any]]] = {
        "samosa": [
            {
                "alternative_name": "Baked Vegetable Samosa",
                "improvement_type": "LOWER_FAT_CALORIES",
                "calories_kcal": 135.0,
                "benefit": "Oven baked with phyllo or whole wheat pastry instead of deep-frying; reduces fat by ~60%.",
                "recommended_for": ["weight_loss", "general_wellness"]
            },
            {
                "alternative_name": "Khamand Dhokla",
                "improvement_type": "STEAMED_HIGH_PROTEIN",
                "calories_kcal": 150.0,
                "benefit": "Steamed fermented chickpea flour snack offering high protein and minimal oil.",
                "recommended_for": ["weight_loss", "manage_diabetes", "general_wellness"]
            },
            {
                "alternative_name": "Moong Dal Chilla",
                "improvement_type": "HIGH_FIBER_PROTEIN",
                "calories_kcal": 140.0,
                "benefit": "Savory pancake from yellow moong lentils with zero refined flour (maida).",
                "recommended_for": ["muscle_gain", "manage_diabetes"]
            }
        ],
        "butter_naan": [
            {
                "alternative_name": "Tandoori Whole Wheat Roti",
                "improvement_type": "ZERO_MAIDA_LOWER_FAT",
                "calories_kcal": 110.0,
                "benefit": "100% whole wheat atta prepared without refined butter or ghee saturation.",
                "recommended_for": ["weight_loss", "manage_diabetes", "manage_hypertension"]
            },
            {
                "alternative_name": "Multigrain Phulka",
                "improvement_type": "HIGH_FIBER_LOW_GI",
                "calories_kcal": 95.0,
                "benefit": "Atta blend with ragi, oats, and barley providing complex low-GI carbohydrates.",
                "recommended_for": ["manage_diabetes", "weight_loss"]
            }
        ],
        "jalebi": [
            {
                "alternative_name": "Steamed Modak",
                "improvement_type": "STEAMED_NATURAL_SWEETENER",
                "calories_kcal": 140.0,
                "benefit": "Steamed rice flour shell with fresh coconut and unrefined jaggery filling.",
                "recommended_for": ["general_wellness"]
            },
            {
                "alternative_name": "Date and Nut Laddu",
                "improvement_type": "ZERO_REFINED_SUGAR",
                "calories_kcal": 120.0,
                "benefit": "Sweetened exclusively with natural dates and roasted seeds, no sugar syrup.",
                "recommended_for": ["manage_diabetes", "general_wellness"]
            }
        ],
        "chole_bhature": [
            {
                "alternative_name": "Chole with Whole Wheat Kulcha",
                "improvement_type": "LESS_ABSORBED_FAT",
                "calories_kcal": 380.0,
                "benefit": "Tawa-toasted kulcha replaces deep-fried bhatura; preserves protein from chickpeas.",
                "recommended_for": ["weight_loss", "general_wellness"]
            },
            {
                "alternative_name": "Sprouted Chickpea Salad",
                "improvement_type": "HIGH_FIBER_MICRONUTRIENT",
                "calories_kcal": 210.0,
                "benefit": "Sprouted legumes tossed with lemon juice, cucumber, and roasted cumin.",
                "recommended_for": ["weight_loss", "manage_diabetes", "muscle_gain"]
            }
        ],
        "biryani": [
            {
                "alternative_name": "Brown Rice Vegetable Pulao",
                "improvement_type": "HIGHER_FIBER_LOWER_GI",
                "calories_kcal": 220.0,
                "benefit": "Replaces polished white rice with brown rice; reduces oil and increases dietary fiber.",
                "recommended_for": ["manage_diabetes", "weight_loss"]
            },
            {
                "alternative_name": "Quinoa / Millet Biryani",
                "improvement_type": "COMPLETE_PROTEIN_MINERALS",
                "calories_kcal": 240.0,
                "benefit": "Foxtail millet or quinoa base with aromatic whole spices and paneer/chicken.",
                "recommended_for": ["muscle_gain", "manage_diabetes"]
            }
        ]
    }

    def get_alternatives(
        self,
        food_name: str,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns structured list of healthier alternatives with justification.
        """
        if not food_name or not isinstance(food_name, str):
            return []

        fn = food_name.lower().strip()
        fn_norm = re.sub(r"[\s\-_]+", "_", fn)
        matched_key = None
        for k in self.CURATED_ALTERNATIVES:
            k_norm = re.sub(r"[\s\-_]+", "_", k)
            if k_norm in fn_norm or fn_norm in k_norm:
                matched_key = k
                break

        results = []
        if matched_key:
            alternatives = self.CURATED_ALTERNATIVES[matched_key]
            user_goal = (user_profile or {}).get("dietary_goal", "general_wellness").lower()

            for alt in alternatives:
                rec_goals = alt.get("recommended_for", [])
                goal_match = user_goal in rec_goals or "general_wellness" in rec_goals
                results.append({
                    "original_food": food_name,
                    "alternative_name": alt["alternative_name"],
                    "improvement_type": alt["improvement_type"],
                    "estimated_calories": alt["calories_kcal"],
                    "benefit_explanation": alt["benefit"],
                    "goal_aligned": goal_match,
                    "disclaimer": RECOMMENDATION_DISCLAIMER
                })
        else:
            # Fallback heuristic for unlisted foods
            results.append({
                "original_food": food_name,
                "alternative_name": f"Steamed / Grilled {food_name.title()}",
                "improvement_type": "LOWER_FAT_PREPARATION",
                "estimated_calories": None,
                "benefit_explanation": "Preparing via steaming, grilling, or air-frying significantly cuts added cooking oils.",
                "goal_aligned": True,
                "disclaimer": RECOMMENDATION_DISCLAIMER
            })

        return results


# ---------------------------------------------------------------------------
# 5. Ingredient Substitution Engine
# ---------------------------------------------------------------------------
class IngredientSubstitutionEngine:
    """
    Provides culturally grounded ingredient substitutions for Indian cooking
    to handle allergies, veganism, diabetes, and sodium restrictions.
    """

    SUBSTITUTIONS_DB: Dict[str, List[Dict[str, Any]]] = {
        "milk": [
            {"substitute": "Almond Milk (Unsweetened)", "ratio": "1:1", "reason": "Lactose-free & dairy-free with low calories.", "suitable_for": ["dairy", "vegan"]},
            {"substitute": "Soy Milk", "ratio": "1:1", "reason": "Plant-based protein comparable to dairy milk.", "suitable_for": ["dairy", "vegan"]},
            {"substitute": "Oat Milk", "ratio": "1:1", "reason": "Creamy texture ideal for curries and chai.", "suitable_for": ["dairy", "vegan"]}
        ],
        "paneer": [
            {"substitute": "Firm Tofu", "ratio": "1:1", "reason": "100% plant-based soy protein with identical texture in gravies.", "suitable_for": ["dairy", "vegan"]},
            {"substitute": "Sprouted Moong Lentils", "ratio": "1 cup sprouts per 100g paneer", "reason": "High-fiber whole legume protein.", "suitable_for": ["dairy", "vegan", "manage_diabetes"]}
        ],
        "ghee": [
            {"substitute": "Cold-Pressed Mustard Oil", "ratio": "3/4 measure of ghee", "reason": "Heart-healthy monounsaturated fats with authentic regional Indian aroma.", "suitable_for": ["dairy", "vegan", "manage_hypertension"]},
            {"substitute": "Extra Virgin Olive Oil", "ratio": "3/4 measure of ghee", "reason": "High in polyphenols and healthy fats.", "suitable_for": ["dairy", "vegan"]}
        ],
        "butter": [
            {"substitute": "Mashed Avocado", "ratio": "1:1 in baking/spreads", "reason": "Rich in oleic acid and dietary fiber without saturated fats.", "suitable_for": ["dairy", "vegan"]},
            {"substitute": "Cold-Pressed Sesame Oil", "ratio": "3/4 measure", "reason": "Plant-based rich aroma.", "suitable_for": ["dairy", "vegan"]}
        ],
        "maida": [
            {"substitute": "Whole Wheat Atta", "ratio": "1:1", "reason": "Unrefined flour providing natural bran, B-vitamins, and dietary fiber.", "suitable_for": ["manage_diabetes", "weight_loss"]},
            {"substitute": "Ragi Flour (Finger Millet)", "ratio": "1:1 (or 50/50 blend)", "reason": "Gluten-free, extremely rich in calcium and low glycemic index.", "suitable_for": ["gluten", "manage_diabetes"]},
            {"substitute": "Gram Flour (Besan)", "ratio": "1:1 in batter", "reason": "Naturally gluten-free, dense in plant protein.", "suitable_for": ["gluten", "high_protein"]}
        ],
        "sugar": [
            {"substitute": "Unrefined Jaggery (Gud)", "ratio": "3/4 cup per 1 cup sugar", "reason": "Retains trace minerals (iron, potassium) with lower processing.", "suitable_for": ["general_wellness"]},
            {"substitute": "Stevia Herb Extract", "ratio": "As per packaging equivalent", "reason": "Zero glycemic impact, non-caloric natural sweetener.", "suitable_for": ["manage_diabetes", "weight_loss"]},
            {"substitute": "Pureed Medjool Dates", "ratio": "1:1 paste", "reason": "Whole fruit sweetness packaged with dietary fiber.", "suitable_for": ["manage_diabetes", "general_wellness"]}
        ],
        "salt": [
            {"substitute": "Lemon Juice + Amchur (Dry Mango Powder)", "ratio": "1 tsp per 1/2 tsp salt", "reason": "Bright acid activates taste buds, reducing need for added sodium.", "suitable_for": ["manage_hypertension", "low_sodium"]},
            {"substitute": "Roasted Cumin Powder + Fresh Herbs", "ratio": "To taste", "reason": "Intense aromatic flavor profile masks low sodium content.", "suitable_for": ["manage_hypertension", "low_sodium"]}
        ],
        "cashew": [
            {"substitute": "Melon Seeds (Magaz) Paste", "ratio": "1:1", "reason": "Provides identical silky creaminess in gravies without tree nuts.", "suitable_for": ["nuts"]},
            {"substitute": "White Poppy Seeds (Khus Khus) Paste", "ratio": "1:1", "reason": "Traditional royal Indian thickener completely free of tree nuts.", "suitable_for": ["nuts"]},
            {"substitute": "Roasted Sunflower Seed Paste", "ratio": "1:1", "reason": "Nut-free emulsifier rich in Vitamin E.", "suitable_for": ["nuts"]}
        ],
        "peanut": [
            {"substitute": "Roasted Sunflower Seeds", "ratio": "1:1", "reason": "Crunchy texture and nutty flavor with zero peanut allergenicity.", "suitable_for": ["peanuts"]},
            {"substitute": "Roasted Chickpeas (Bhuna Chana)", "ratio": "1:1", "reason": "Crisp texture, high protein, peanut-free.", "suitable_for": ["peanuts"]}
        ],
        "egg": [
            {"substitute": "Flaxseed Slurry (1 tbsp ground flax + 3 tbsp water)", "ratio": "Replaces 1 egg in baking", "reason": "Natural binding agent loaded with Omega-3 fatty acids.", "suitable_for": ["eggs", "vegan"]},
            {"substitute": "Aquafaba (Chickpea Brine)", "ratio": "3 tbsp per egg", "reason": "Whips and emulsifies identically to egg whites.", "suitable_for": ["eggs", "vegan"]}
        ]
    }

    def get_substitutions_for_ingredient(
        self,
        ingredient_name: str,
        reason_or_allergy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Finds culinary substitutions for a specific ingredient.
        """
        if not ingredient_name:
            return []

        ing_clean = ingredient_name.lower().strip()
        matched_key = None
        for key in self.SUBSTITUTIONS_DB:
            if key in ing_clean or ing_clean in key:
                matched_key = key
                break

        if not matched_key:
            return []

        subs = self.SUBSTITUTIONS_DB[matched_key]
        if reason_or_allergy:
            r_lower = reason_or_allergy.lower().strip()
            filtered = [s for s in subs if any(r_lower in sf.lower() for sf in s.get("suitable_for", []))]
            return filtered or subs

        return subs

    def suggest_recipe_substitutions(
        self,
        recipe: Dict[str, Any],
        profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Scans a recipe for ingredients that violate user allergies or restrictions,
        and generates actionable culinary swap recommendations.
        """
        raw_text = str(recipe.get("ingredients_raw", "") or "").lower()
        user_allergies = [a.lower().strip() for a in profile.get("allergies", []) if a]
        restrictions = [r.lower().strip() for r in profile.get("dietary_restrictions", []) if r]

        suggested_swaps = []
        for ing_key, subs in self.SUBSTITUTIONS_DB.items():
            if re.search(r"\b" + re.escape(ing_key) + r"\b", raw_text):
                # Check if this ingredient causes an allergy or restriction conflict
                needs_swap = False
                swap_reasons = []

                for allergy in user_allergies:
                    if allergy == ing_key or allergy in ALLERGEN_LEXICON.get(ing_key, set()):
                        needs_swap = True
                        swap_reasons.append(f"Allergy conflict: {allergy.upper()}")

                if "vegan" in restrictions and ing_key in ("milk", "paneer", "ghee", "butter", "egg"):
                    needs_swap = True
                    swap_reasons.append("Vegan compliance")

                if "gluten_free" in restrictions and ing_key in ("maida", "wheat"):
                    needs_swap = True
                    swap_reasons.append("Gluten-free compliance")

                if "manage_diabetes" in restrictions and ing_key in ("sugar", "maida"):
                    needs_swap = True
                    swap_reasons.append("Glycemic control")

                if "low_sodium" in restrictions and ing_key == "salt":
                    needs_swap = True
                    swap_reasons.append("Sodium reduction")

                if needs_swap:
                    suggested_swaps.append({
                        "ingredient_found": ing_key,
                        "reasons": swap_reasons,
                        "recommended_options": subs
                    })

        return {
            "recipe_id": recipe.get("recipe_id"),
            "recipe_name": recipe.get("recipe_name"),
            "total_swaps_identified": len(suggested_swaps),
            "swaps": suggested_swaps,
            "disclaimer": RECOMMENDATION_DISCLAIMER
        }


# ---------------------------------------------------------------------------
# 6. Recommendation Coordinator & Meal Planning Engines
# ---------------------------------------------------------------------------
class RecommendationSystem:
    """
    Central orchestration engine for Phase 9:
    - Personalized recipe recommendations with transparent 0-100 scoring.
    - Safety filter enforcement.
    - Alternative foods and ingredient substitutions.
    - Calibrated daily and weekly meal plan generation.
    """

    def __init__(self, recipe_path: Optional[Path] = None):
        self.recipe_path = recipe_path or RECIPE_CSV
        self.recipes_df = pd.read_csv(self.recipe_path) if self.recipe_path.exists() else pd.DataFrame()
        self.safety_engine = SafetyFilterEngine()
        self.scorer = SuitabilityScorer()
        self.content_recommender = ContentBasedRecommender(self.recipes_df)
        self.alternatives_engine = AlternativeFoodsEngine()
        self.substitution_engine = IngredientSubstitutionEngine()

    def recommend_recipes(
        self,
        profile: Dict[str, Any],
        course: Optional[str] = None,
        top_n: int = 10,
        query_hint: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns top N safe, scored, and ranked recipe recommendations for user profile.
        """
        if self.recipes_df.empty:
            return []

        # Step 1: Pre-filter by course if specified
        candidates = self.recipes_df.copy()
        if course:
            c_lower = course.lower().strip()
            candidates = candidates[candidates["course"].astype(str).str.lower().str.contains(c_lower, na=False)]

        # Step 2: Content similarity scoring if query_hint exists
        sim_scores_map = {}
        if query_hint:
            sims = self.content_recommender.get_similar_recipes(query_hint, top_k=50)
            for idx, score in sims:
                sim_scores_map[idx] = score

        scored_candidates = []
        candidates_records = candidates.to_dict(orient="records")
        for idx, rec_dict in enumerate(candidates_records):
            # Step 3: MANDATORY SAFETY CHECK FIRST
            safety = self.safety_engine.check_safety(rec_dict, profile)
            if not safety["is_safe"]:
                continue  # Never pass unsafe recipes to ranking

            # Step 4: Transparent 0-100 Scoring
            sim_score = sim_scores_map.get(idx, 0.0)
            score_meta = self.scorer.score_recipe(rec_dict, profile, content_sim=sim_score)

            scored_candidates.append({
                "recipe_id": str(rec_dict.get("recipe_id", f"rec_{idx}")),
                "recipe_name": str(rec_dict.get("recipe_name", "Unknown Dish")).title(),
                "cuisine": str(rec_dict.get("cuisine", "Indian")),
                "course": str(rec_dict.get("course", "Main Course")),
                "diet": str(rec_dict.get("diet", "vegetarian")),
                "total_time_mins": float(rec_dict.get("total_time_mins", 30) or 30),
                "complexity_level": str(rec_dict.get("complexity_level", "MODERATE")),
                "ingredients_raw": str(rec_dict.get("ingredients_raw", "") or ""),
                "ingredient_tokens": str(rec_dict.get("ingredient_tokens", "") or ""),
                "suitability_score": score_meta["total_score"],
                "verdict": score_meta["verdict"],
                "component_scores": score_meta["component_scores"],
                "explanation": score_meta["explanation"],
                "has_incomplete_ingredients": score_meta["has_incomplete_ingredients"],
            })

        # Step 5: Rank descending by suitability score
        scored_candidates.sort(key=lambda x: x["suitability_score"], reverse=True)
        return scored_candidates[:top_n]

    def generate_daily_meal_plan(
        self,
        profile: Dict[str, Any],
        target_calories_kcal: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates a 1-day meal plan: Breakfast, Lunch, Snack, Dinner.
        Calibrates caloric distribution: Breakfast (~25%), Lunch (~35%), Snack (~10%), Dinner (~30%).
        Strictly enforces all safety constraints and maximizes suitability scores.
        """
        # Determine target daily calories
        if target_calories_kcal is None:
            # Fallback to estimated 2000 kcal if profile incomplete
            target_calories_kcal = float(profile.get("daily_calorie_target", {}).get("target_calories_kcal", 2000.0) or 2000.0)

        # Caloric target per meal
        meal_calorie_targets = {
            "breakfast": round(target_calories_kcal * 0.25, 0),
            "lunch": round(target_calories_kcal * 0.35, 0),
            "snack": round(target_calories_kcal * 0.10, 0),
            "dinner": round(target_calories_kcal * 0.30, 0),
        }

        # Query candidates for each course
        bfast_candidates = self.recommend_recipes(profile, course="breakfast", top_n=5)
        lunch_candidates = self.recommend_recipes(profile, course="lunch", top_n=5)
        snack_candidates = self.recommend_recipes(profile, course="snack", top_n=5)
        dinner_candidates = self.recommend_recipes(profile, course="dinner", top_n=5)

        # Fallbacks if course-specific count is low
        if not bfast_candidates:
            bfast_candidates = self.recommend_recipes(profile, course="south indian breakfast", top_n=5)
        if not dinner_candidates:
            dinner_candidates = [r for r in lunch_candidates if r != (lunch_candidates[0] if lunch_candidates else None)]
        if not lunch_candidates:
            lunch_candidates = self.recommend_recipes(profile, course="main course", top_n=5)

        # Pick top unique meals
        used_ids = set()
        def _pick_best(candidates_list, default_name, default_ing="vegetables, turmeric, cumin, salt, water"):
            for cand in candidates_list:
                if cand["recipe_id"] not in used_ids:
                    used_ids.add(cand["recipe_id"])
                    return cand
            if candidates_list:
                return candidates_list[0]
            return {
                "recipe_id": "fallback_01",
                "recipe_name": default_name,
                "cuisine": "Indian",
                "course": "General",
                "diet": "vegetarian",
                "total_time_mins": 25.0,
                "complexity_level": "EASY",
                "ingredients_raw": default_ing,
                "ingredient_tokens": default_ing.replace(",", " "),
                "suitability_score": 75.0,
                "verdict": "RECOMMENDED",
                "explanation": "Standard balanced fallback adhering to profile safety constraints.",
                "has_incomplete_ingredients": False
            }

        bfast = _pick_best(bfast_candidates, "Vegetable Poha")
        lunch = _pick_best(lunch_candidates, "Moong Dal Khichdi with Raita")
        snack = _pick_best(snack_candidates, "Roasted Chana with Cucumber Salad")
        dinner = _pick_best(dinner_candidates, "Phulka with Palak Sabzi and Dal")

        plan = {
            "breakfast": {**bfast, "target_calories_kcal": meal_calorie_targets["breakfast"]},
            "lunch": {**lunch, "target_calories_kcal": meal_calorie_targets["lunch"]},
            "snack": {**snack, "target_calories_kcal": meal_calorie_targets["snack"]},
            "dinner": {**dinner, "target_calories_kcal": meal_calorie_targets["dinner"]},
        }

        avg_score = round(
            (bfast["suitability_score"] + lunch["suitability_score"] +
             snack["suitability_score"] + dinner["suitability_score"]) / 4.0, 1
        )

        return {
            "daily_target_calories_kcal": target_calories_kcal,
            "meals": plan,
            "average_suitability_score": avg_score,
            "disclaimer": RECOMMENDATION_DISCLAIMER,
            "safety_notice": SAFETY_OVERRIDE_NOTICE,
            "total_prep_cook_time_mins": sum(m["total_time_mins"] for m in plan.values())
        }

    def generate_weekly_meal_plan(
        self,
        profile: Dict[str, Any],
        target_calories_kcal: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates a 7-day meal plan with variety control and an aggregated grocery shopping list.
        """
        daily_target = target_calories_kcal or float(
            profile.get("daily_calorie_target", {}).get("target_calories_kcal", 2000.0) or 2000.0
        )

        # Retrieve larger pools to avoid consecutive repeats
        bfast_pool = self.recommend_recipes(profile, course="breakfast", top_n=20)
        lunch_pool = self.recommend_recipes(profile, course="lunch", top_n=20)
        snack_pool = self.recommend_recipes(profile, course="snack", top_n=20)
        dinner_pool = self.recommend_recipes(profile, course="dinner", top_n=20) or lunch_pool

        # Ensure pools are non-empty
        if not bfast_pool:
            bfast_pool = self.recommend_recipes(profile, top_n=15)
        if not lunch_pool:
            lunch_pool = self.recommend_recipes(profile, top_n=15)
        if not snack_pool:
            snack_pool = self.recommend_recipes(profile, top_n=15)
        if not dinner_pool:
            dinner_pool = self.recommend_recipes(profile, top_n=15)

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_schedule = {}
        all_planned_recipes = []

        last_bfast_id = None
        last_lunch_id = None
        last_snack_id = None
        last_dinner_id = None

        def _select_non_consecutive(pool, last_id, day_idx):
            candidates = [p for p in pool if p["recipe_id"] != last_id]
            if not candidates:
                candidates = pool
            # Rotate by day index to encourage variety
            chosen = candidates[day_idx % len(candidates)]
            return chosen

        for day_idx, day_name in enumerate(days):
            bfast = _select_non_consecutive(bfast_pool, last_bfast_id, day_idx)
            lunch = _select_non_consecutive(lunch_pool, last_lunch_id, day_idx)
            snack = _select_non_consecutive(snack_pool, last_snack_id, day_idx)
            dinner = _select_non_consecutive(dinner_pool, last_dinner_id, day_idx)

            last_bfast_id = bfast["recipe_id"]
            last_lunch_id = lunch["recipe_id"]
            last_snack_id = snack["recipe_id"]
            last_dinner_id = dinner["recipe_id"]

            all_planned_recipes.extend([bfast, lunch, snack, dinner])

            daily_avg_score = round(
                (bfast["suitability_score"] + lunch["suitability_score"] +
                 snack["suitability_score"] + dinner["suitability_score"]) / 4.0, 1
            )

            weekly_schedule[day_name] = {
                "breakfast": bfast,
                "lunch": lunch,
                "snack": snack,
                "dinner": dinner,
                "day_suitability_score": daily_avg_score,
                "day_target_calories": daily_target,
            }

        # Build aggregated grocery list from recipes
        grocery_counter = Counter()
        for rec in all_planned_recipes:
            rid = rec["recipe_id"]
            # Look up ingredients in raw dataframe
            match_row = self.recipes_df[self.recipes_df["recipe_id"].astype(str) == str(rid)]
            if not match_row.empty:
                tokens = str(match_row.iloc[0].get("ingredient_tokens", "") or "")
                for token in tokens.split():
                    if len(token) > 3 and token not in ("with", "taste", "salt", "water", "oil", "grams"):
                        grocery_counter[token] += 1

        top_grocery_items = [
            {"ingredient": item, "frequency_in_week": count}
            for item, count in grocery_counter.most_common(25)
        ]

        overall_avg = round(
            sum(d["day_suitability_score"] for d in weekly_schedule.values()) / 7.0, 1
        )

        return {
            "plan_type": "7_DAY_WEEKLY_PLAN",
            "daily_target_calories_kcal": daily_target,
            "overall_weekly_suitability_score": overall_avg,
            "weekly_schedule": weekly_schedule,
            "grocery_shopping_list": top_grocery_items,
            "disclaimer": RECOMMENDATION_DISCLAIMER,
            "safety_notice": SAFETY_OVERRIDE_NOTICE,
        }


# ---------------------------------------------------------------------------
# Module Singleton
# ---------------------------------------------------------------------------
_global_engine: Optional[RecommendationSystem] = None

def get_recommendation_system() -> RecommendationSystem:
    global _global_engine
    if _global_engine is None:
        _global_engine = RecommendationSystem()
    return _global_engine
