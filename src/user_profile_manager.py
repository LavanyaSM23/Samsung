"""
src/user_profile_manager.py
NutriSense AI — Phase 8: Personalization and User Profiling

Provides:
1. User profile creation, validation, persistence (SQLite)
2. BMI computation with WHO categories and South Asian thresholds disclaimer
3. Daily calorie target via Mifflin-St Jeor equation with activity + goal adjustments
4. ICMR-NIN RDA profile mapping for suitability scoring
5. Personalised breakfast suggestions from recipe feature store
6. Personalised portion guidance per food
7. Food suitability scoring against user profile

No passwords or clinical medical advice. All BMI/calorie outputs include mandatory disclaimers.
"""

import json
import logging
import math
import os
import random
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = WORKSPACE_ROOT / "database"
DB_PATH = DB_DIR / "nutrisense_users.db"
FEATURE_DIR = WORKSPACE_ROOT / "data" / "processed" / "feature_store"
RECIPE_CSV = FEATURE_DIR / "recipe_feature_store.csv"
HEALTH_PROFILES_JSON = FEATURE_DIR / "user_health_profiles.json"
SUITABILITY_CSV = FEATURE_DIR / "food_suitability_benchmarks.csv"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disclaimer constants (verbatim per spec)
# ---------------------------------------------------------------------------
BMI_DISCLAIMER = (
    "BMI is a population-level screening index, not a diagnostic tool. "
    "It does not account for muscle mass, bone density, age, sex, ethnicity, or fat distribution. "
    "For individuals of South Asian descent, health risk thresholds may differ from WHO universal standards "
    "(overweight ≥23.0, obese ≥27.5 per consensus guidelines). "
    "BMI values are provided for general awareness only and must not be used for clinical decisions "
    "without consulting a qualified healthcare professional."
)

CALORIE_DISCLAIMER = (
    "This is an estimated target based on population-level equations. "
    "Consult a registered dietitian for personalised dietary advice."
)

BREAKFAST_DISCLAIMER = (
    "Suggestions are based on your profile preferences and are not prescriptive meal plans."
)

# ---------------------------------------------------------------------------
# Valid option sets
# ---------------------------------------------------------------------------
VALID_GENDERS = {"male", "female", "other", "prefer_not_to_say"}
VALID_ACTIVITY_LEVELS = {
    "sedentary", "lightly_active", "moderately_active", "very_active", "extra_active"
}
VALID_DIETARY_GOALS = {
    "weight_loss", "weight_gain", "maintain_weight", "muscle_gain",
    "improve_nutrition", "manage_diabetes", "manage_hypertension", "general_wellness"
}
VALID_CUISINES = {
    "North Indian", "South Indian", "Punjabi", "Bengali", "Gujarati",
    "Maharashtrian", "Chettinad", "Rajasthani", "Any"
}
VALID_ALLERGIES = {
    "nuts", "peanuts", "dairy", "gluten", "eggs", "shellfish", "soy", "sesame"
}
VALID_DIETARY_RESTRICTIONS = {
    "vegetarian", "vegan", "eggetarian", "jain", "halal", "kosher",
    "diabetic_friendly", "low_sodium", "high_protein", "gluten_free"
}
VALID_BUDGETS = {"low", "medium", "high"}
VALID_BREAKFAST_PREFERENCES = {"light", "medium", "hearty"}

ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extra_active": 1.9,
}

GOAL_ADJUSTMENTS = {
    "weight_loss": -300,
    "weight_gain": 300,
    "muscle_gain": 200,
}

# ICMR-NIN profile keys
RDA_PROFILE_KEYS = [
    "adult_male_sedentary",
    "adult_female_sedentary",
    "diabetic_adult",
    "hypertensive_adult",
    "weight_management",
]

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB directory and table if needed."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create user_profiles table if it does not already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id      TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            gender       TEXT,
            age          INTEGER,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            profile_blob TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_up_display_name ON user_profiles(display_name)"
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_profile_data(data: dict) -> Tuple[bool, List[str]]:
    """
    Validate all user profile fields.

    Returns (True, []) on success, or (False, [error_messages]) on failure.
    """
    errors: List[str] = []

    # display_name — required; accept 'name' fallback if provided
    raw_name = data.get("display_name") or data.get("name")
    if not raw_name or not str(raw_name).strip():
        errors.append("display_name is required and cannot be empty.")

    # age
    if "age" in data and data["age"] is not None:
        try:
            age = int(data["age"])
            if not (5 <= age <= 120):
                errors.append("age must be an integer between 5 and 120.")
        except (TypeError, ValueError):
            errors.append("age must be a valid integer.")

    # gender (also accept 'sex' as alias for frontend compatibility)
    gender_val = data.get("gender") or data.get("sex")
    if gender_val is not None and gender_val != "":
        if gender_val not in VALID_GENDERS:
            errors.append(
                f"gender must be one of: {', '.join(sorted(VALID_GENDERS))}."
            )

    # height_cm
    if "height_cm" in data and data["height_cm"] is not None:
        try:
            h = float(data["height_cm"])
            if not (50.0 <= h <= 280.0):
                errors.append("height_cm must be between 50 and 280.")
        except (TypeError, ValueError):
            errors.append("height_cm must be a valid number.")

    # weight_kg
    if "weight_kg" in data and data["weight_kg"] is not None:
        try:
            w = float(data["weight_kg"])
            if not (10.0 <= w <= 300.0):
                errors.append("weight_kg must be between 10 and 300.")
        except (TypeError, ValueError):
            errors.append("weight_kg must be a valid number.")

    # activity_level
    if "activity_level" in data and data["activity_level"] is not None:
        if data["activity_level"] not in VALID_ACTIVITY_LEVELS:
            errors.append(
                f"activity_level must be one of: {', '.join(sorted(VALID_ACTIVITY_LEVELS))}."
            )

    # dietary_goal (also check nutrition_goal alias; normalize legacy frontend values)
    _goal_legacy_map = {'maintenance': 'maintain_weight', 'general_health': 'general_wellness'}
    goal_val = data.get("dietary_goal") or data.get("nutrition_goal")
    if goal_val is not None and goal_val != "":
        goal_val_normalized = _goal_legacy_map.get(goal_val, goal_val)
        if goal_val_normalized not in VALID_DIETARY_GOALS:
            errors.append(
                f"dietary_goal must be one of: {', '.join(sorted(VALID_DIETARY_GOALS))}."
            )

    # preferred_cuisines
    if "preferred_cuisines" in data and data["preferred_cuisines"] is not None:
        if not isinstance(data["preferred_cuisines"], list):
            errors.append("preferred_cuisines must be a list.")
        else:
            invalid = [c for c in data["preferred_cuisines"] if c not in VALID_CUISINES]
            if invalid:
                errors.append(
                    f"Invalid cuisine(s): {invalid}. Valid: {sorted(VALID_CUISINES)}."
                )

    # allergies
    if "allergies" in data and data["allergies"] is not None:
        if not isinstance(data["allergies"], list):
            errors.append("allergies must be a list.")
        else:
            # Allow both standard items and free-text strings (non-empty)
            std = [a for a in data["allergies"] if a not in VALID_ALLERGIES]
            # free-text entries are allowed; just validate they are non-empty strings
            invalid_free = [a for a in std if not isinstance(a, str) or not a.strip()]
            if invalid_free:
                errors.append("All allergy entries must be non-empty strings.")

    # dietary_restrictions
    if "dietary_restrictions" in data and data["dietary_restrictions"] is not None:
        if not isinstance(data["dietary_restrictions"], list):
            errors.append("dietary_restrictions must be a list.")
        else:
            invalid = [
                r for r in data["dietary_restrictions"]
                if r not in VALID_DIETARY_RESTRICTIONS
            ]
            if invalid:
                errors.append(
                    f"Invalid dietary restriction(s): {invalid}. "
                    f"Valid: {sorted(VALID_DIETARY_RESTRICTIONS)}."
                )

    # budget
    if "budget" in data and data["budget"] is not None:
        if data["budget"] not in VALID_BUDGETS:
            errors.append(
                f"budget must be one of: {', '.join(sorted(VALID_BUDGETS))}."
            )

    # max_cooking_time_mins
    if "max_cooking_time_mins" in data and data["max_cooking_time_mins"] is not None:
        try:
            t = int(data["max_cooking_time_mins"])
            if not (5 <= t <= 300):
                errors.append("max_cooking_time_mins must be between 5 and 300.")
        except (TypeError, ValueError):
            errors.append("max_cooking_time_mins must be a valid integer.")

    # dietary_considerations (free text, max 500)
    if "dietary_considerations" in data and data["dietary_considerations"] is not None:
        if len(str(data["dietary_considerations"])) > 500:
            errors.append(
                "dietary_considerations must not exceed 500 characters."
            )

    # wake_time / breakfast_time  HH:MM format
    for field in ("wake_time", "breakfast_time"):
        if field in data and data[field] is not None:
            val = str(data[field])
            parts = val.split(":")
            if len(parts) != 2:
                errors.append(f"{field} must be in HH:MM format.")
            else:
                try:
                    hh, mm = int(parts[0]), int(parts[1])
                    if not (0 <= hh <= 23 and 0 <= mm <= 59):
                        errors.append(f"{field} must be a valid time (HH:MM).")
                except ValueError:
                    errors.append(f"{field} must be in HH:MM format.")

    # breakfast_preference
    if "breakfast_preference" in data and data["breakfast_preference"] is not None:
        if data["breakfast_preference"] not in VALID_BREAKFAST_PREFERENCES:
            errors.append(
                f"breakfast_preference must be one of: {', '.join(sorted(VALID_BREAKFAST_PREFERENCES))}."
            )

    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# BMI
# ---------------------------------------------------------------------------

def compute_bmi(height_cm, weight_kg) -> dict:
    """
    Compute BMI, WHO category, South Asian thresholds, and disclaimer.

    Returns a dict with bmi, bmi_category, who_ranges, south_asian_note, disclaimer.
    Raises ValueError for missing/invalid inputs.
    """
    if height_cm is None or weight_kg is None:
        raise ValueError("Both height_cm and weight_kg are required to compute BMI.")
    h = float(height_cm)
    w = float(weight_kg)
    if h <= 0 or w <= 0:
        raise ValueError("height_cm and weight_kg must be positive values.")

    height_m = h / 100.0
    bmi_value = w / (height_m ** 2)
    bmi_rounded = round(bmi_value, 2)

    # WHO universal categories
    if bmi_value < 18.5:
        category = "Underweight"
    elif bmi_value < 25.0:
        category = "Normal weight"
    elif bmi_value < 30.0:
        category = "Overweight"
    else:
        category = "Obese"

    # South Asian adjusted categories
    if bmi_value < 18.5:
        sa_category = "Underweight"
    elif bmi_value < 23.0:
        sa_category = "Normal weight"
    elif bmi_value < 27.5:
        sa_category = "Overweight (South Asian threshold)"
    else:
        sa_category = "Obese (South Asian threshold)"

    return {
        "bmi": bmi_rounded,
        "bmi_category": category,
        "south_asian_category": sa_category,
        "who_ranges": {
            "underweight": "< 18.5",
            "normal": "18.5 – 24.9",
            "overweight": "25.0 – 29.9",
            "obese": "≥ 30.0",
        },
        "south_asian_ranges": {
            "underweight": "< 18.5",
            "normal": "18.5 – 22.9",
            "overweight": "23.0 – 27.4",
            "obese": "≥ 27.5",
        },
        "disclaimer": BMI_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Calorie target
# ---------------------------------------------------------------------------

def compute_daily_calorie_target(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    activity_level: str,
    dietary_goal: str,
) -> dict:
    """
    Compute daily calorie target using Mifflin-St Jeor equation.

    Returns dict with bmr, tdee, goal_adjustment, target_kcal, explanation, disclaimer.
    """
    if None in (age, gender, height_cm, weight_kg, activity_level, dietary_goal):
        raise ValueError(
            "age, gender, height_cm, weight_kg, activity_level, and dietary_goal are all required."
        )

    h = float(height_cm)
    w = float(weight_kg)
    a = int(age)

    bmr_male = (10 * w) + (6.25 * h) - (5 * a) + 5
    bmr_female = (10 * w) + (6.25 * h) - (5 * a) - 161

    if gender == "male":
        bmr = bmr_male
        gender_label = "Male"
    elif gender == "female":
        bmr = bmr_female
        gender_label = "Female"
    else:
        bmr = (bmr_male + bmr_female) / 2
        gender_label = "Other / prefer_not_to_say (average of male and female)"

    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
    tdee = bmr * multiplier
    goal_adj = GOAL_ADJUSTMENTS.get(dietary_goal, 0)
    target = tdee + goal_adj

    explanation = (
        f"BMR ({gender_label}): {bmr:.1f} kcal/day using Mifflin-St Jeor equation. "
        f"Activity multiplier ({activity_level}): ×{multiplier}. "
        f"TDEE: {tdee:.1f} kcal/day. "
        f"Goal adjustment ({dietary_goal}): {goal_adj:+d} kcal. "
        f"Final target: {target:.0f} kcal/day."
    )

    return {
        "bmr": round(bmr, 1),
        "bmr_kcal": round(bmr, 1),
        "tdee": round(tdee, 1),
        "tdee_kcal": round(tdee, 1),
        "activity_multiplier": multiplier,
        "goal_adjustment_kcal": goal_adj,
        "target_kcal": round(target, 0),
        "target_calories_kcal": round(target, 0),
        "explanation": explanation,
        "disclaimer": CALORIE_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# RDA profile mapping
# ---------------------------------------------------------------------------

def map_to_rda_profile(profile_dict: dict) -> str:
    """
    Map a user profile dict to the nearest ICMR-NIN RDA profile key.

    Priority logic:
    1. manage_diabetes  → diabetic_adult
    2. manage_hypertension → hypertensive_adult
    3. weight_loss / maintain_weight / general_wellness + BMI > 25 → weight_management
    4. female → adult_female_sedentary
    5. default → adult_male_sedentary
    """
    goal = profile_dict.get("dietary_goal", "")
    gender = profile_dict.get("gender", "")
    restrictions = profile_dict.get("dietary_restrictions", []) or []

    if goal == "manage_diabetes" or "diabetic_friendly" in restrictions:
        return "diabetic_adult"
    if goal == "manage_hypertension" or "low_sodium" in restrictions:
        return "hypertensive_adult"
    if goal in ("weight_loss", "maintain_weight"):
        # check BMI
        h = profile_dict.get("height_cm")
        w = profile_dict.get("weight_kg")
        if h and w:
            try:
                bmi_v = float(w) / ((float(h) / 100.0) ** 2)
                if bmi_v > 25.0:
                    return "weight_management"
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        return "weight_management"
    if gender == "female":
        return "adult_female_sedentary"
    return "adult_male_sedentary"


# ---------------------------------------------------------------------------
# Recipe loader (cached)
# ---------------------------------------------------------------------------

_recipe_df: Optional[pd.DataFrame] = None


def _load_recipes() -> pd.DataFrame:
    global _recipe_df
    if _recipe_df is None:
        if RECIPE_CSV.exists():
            _recipe_df = pd.read_csv(str(RECIPE_CSV))
        else:
            _recipe_df = pd.DataFrame()
    return _recipe_df


# ---------------------------------------------------------------------------
# Breakfast suggestion
# ---------------------------------------------------------------------------

def get_breakfast_suggestion(profile_dict: dict) -> dict:
    """
    Suggest up to 3 breakfast recipe options from the recipe feature store.

    Filters:
    - course == 'breakfast' (or falls back to any course)
    - dietary restrictions must match diet column
    - max total_time_mins based on breakfast_preference (light≤20, medium≤40, hearty≤60)
    - Ranked by total_time ascending (simpler / quicker first)

    Returns dict with suggestions list and disclaimer.
    """
    df = _load_recipes()
    if df.empty:
        return {
            "suggestions": [],
            "note": "Recipe data unavailable.",
            "disclaimer": BREAKFAST_DISCLAIMER,
        }

    recipes = df.copy()

    # Time limit by preference
    pref = profile_dict.get("breakfast_preference", "medium")
    max_time_map = {"light": 20, "medium": 40, "hearty": 60}
    max_time_override = profile_dict.get("max_cooking_time_mins")

    pref_time = max_time_map.get(pref, 40)
    if max_time_override is not None:
        try:
            pref_time = min(pref_time, int(max_time_override))
        except (TypeError, ValueError):
            pass

    # Filter by time
    if "total_time_mins" in recipes.columns:
        recipes = recipes[recipes["total_time_mins"] <= pref_time]

    # Filter by dietary restrictions
    restrictions = profile_dict.get("dietary_restrictions", []) or []
    diet_col = "diet" if "diet" in recipes.columns else None

    if diet_col and restrictions:
        def _diet_ok(row_diet):
            if not isinstance(row_diet, str):
                return True
            row_diet_lower = row_diet.lower()
            for r in restrictions:
                if r in ("vegetarian", "vegan", "eggetarian", "jain", "halal", "kosher"):
                    if r.lower() not in row_diet_lower and "vegetarian" not in row_diet_lower:
                        return False
                if r == "diabetic_friendly" and "diabetic" not in row_diet_lower:
                    return False
                if r == "gluten_free" and "gluten" in row_diet_lower:
                    return False
            return True

        # Try to filter; if result is empty fall through without filter
        filtered = recipes[recipes[diet_col].apply(_diet_ok)]
        if not filtered.empty:
            recipes = filtered

    # Prefer breakfast course
    if "course" in recipes.columns:
        breakfast_courses = recipes[recipes["course"].str.lower().str.contains("breakfast", na=False)]
        if not breakfast_courses.empty:
            recipes = breakfast_courses

    # Sort by total_time and take top 3
    if "total_time_mins" in recipes.columns:
        recipes = recipes.sort_values("total_time_mins")

    top3 = recipes.head(3)

    suggestions = []
    for _, row in top3.iterrows():
        suggestions.append({
            "recipe_id": row.get("recipe_id", ""),
            "recipe_name": row.get("recipe_name", "Unknown"),
            "cuisine": row.get("cuisine", ""),
            "diet": row.get("diet", ""),
            "total_time_mins": row.get("total_time_mins", None),
            "complexity_level": row.get("complexity_level", ""),
            "servings": row.get("servings", None),
        })

    # Morning routine note
    wake = profile_dict.get("wake_time")
    bfast_time = profile_dict.get("breakfast_time")
    timing_note = None
    if wake and bfast_time:
        try:
            wh, wm = map(int, wake.split(":"))
            bh, bm = map(int, bfast_time.split(":"))
            diff_mins = (bh * 60 + bm) - (wh * 60 + wm)
            if 0 <= diff_mins <= 120:
                timing_note = (
                    f"Breakfast at {bfast_time} is {diff_mins} min after waking at {wake}. "
                    "Within recommended 2-hour morning window."
                )
            else:
                timing_note = (
                    f"Note: gap between wake time ({wake}) and breakfast time ({bfast_time}) "
                    "is outside the typical 0–2 hour morning window."
                )
        except (ValueError, AttributeError):
            pass

    return {
        "suggestions": suggestions,
        "count": len(suggestions),
        "breakfast_preference": pref,
        "max_time_applied_mins": pref_time,
        "timing_note": timing_note,
        "note": BREAKFAST_DISCLAIMER,
        "disclaimer": BREAKFAST_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Personalized portion guidance
# ---------------------------------------------------------------------------

def personalize_portion_guidance(food_name: str, profile_dict: dict) -> dict:
    """
    Suggest appropriate portion size with reasoning based on user profile.

    Returns dict with recommended_portion_g, unit, reasoning, and profile_context.
    """
    goal = profile_dict.get("dietary_goal", "general_wellness")
    bmi_val = None
    h = profile_dict.get("height_cm")
    w = profile_dict.get("weight_kg")
    if h and w:
        try:
            bmi_val = float(w) / ((float(h) / 100.0) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # Base portion in grams
    base_portion_g = 150

    # Adjust by goal
    goal_adjustments_portion = {
        "weight_loss": -30,
        "weight_gain": +30,
        "muscle_gain": +20,
        "manage_diabetes": -20,
    }
    adj = goal_adjustments_portion.get(goal, 0)

    # Adjust by BMI
    if bmi_val is not None:
        if bmi_val > 30:
            adj -= 20
        elif bmi_val < 18.5:
            adj += 20

    recommended = max(50, base_portion_g + adj)

    reasoning_parts = [
        f"Base reference portion: {base_portion_g}g.",
        f"Dietary goal ({goal}): {adj:+d}g adjustment.",
    ]
    if bmi_val is not None:
        reasoning_parts.append(f"BMI {bmi_val:.1f}: additional BMI-based adjustment applied.")
    reasoning_parts.append(
        "These are general guidance estimates only. "
        "Consult a registered dietitian for clinical portion advice."
    )

    restrictions = profile_dict.get("dietary_restrictions", []) or []
    allergies = profile_dict.get("allergies", []) or []

    return {
        "food_name": food_name,
        "recommended_portion_g": recommended,
        "unit": "grams",
        "reasoning": " ".join(reasoning_parts),
        "profile_context": {
            "dietary_goal": goal,
            "bmi": round(bmi_val, 2) if bmi_val else None,
            "dietary_restrictions": restrictions,
            "allergies": allergies,
        },
        "disclaimer": (
            "Portion guidance is indicative only and not a clinical prescription. "
            "Consult a registered dietitian for personalised dietary advice."
        ),
    }


# ---------------------------------------------------------------------------
# Food suitability for profile
# ---------------------------------------------------------------------------

def get_food_suitability_for_profile(food_name: str, profile_dict: dict) -> dict:
    """
    Score a food against the user's mapped ICMR-NIN RDA profile.

    Uses pre-scored food_suitability_benchmarks.csv when available.
    Returns dict with rda_profile_key, suitability_score, verdict, and notes.
    """
    rda_key = map_to_rda_profile(profile_dict)

    # Try to load pre-scored data
    score = None
    source = "not_found"
    if SUITABILITY_CSV.exists():
        try:
            df = pd.read_csv(str(SUITABILITY_CSV))
            # Normalize food name for matching
            food_lower = food_name.lower().strip()
            name_col = None
            for c in df.columns:
                if "food" in c.lower() or "name" in c.lower():
                    name_col = c
                    break
            if name_col and rda_key in df.columns:
                match = df[df[name_col].str.lower().str.strip() == food_lower]
                if not match.empty:
                    score_val = match.iloc[0][rda_key]
                    if pd.notna(score_val):
                        score = float(score_val)
                        source = "benchmarks_csv"
        except Exception as exc:
            logger.warning("Failed to load suitability benchmarks: %s", exc)

    if score is None:
        score = 5.0  # neutral default
        source = "default_neutral"

    if score >= 7.5:
        verdict = "Highly Suitable"
    elif score >= 5.0:
        verdict = "Moderately Suitable"
    elif score >= 3.0:
        verdict = "Marginally Suitable"
    else:
        verdict = "Not Recommended"

    return {
        "food_name": food_name,
        "rda_profile_key": rda_key,
        "suitability_score": round(score, 2),
        "verdict": verdict,
        "score_source": source,
        "disclaimer": (
            "Food suitability scores are derived from population-level ICMR-NIN RDA reference data. "
            "Individual nutritional needs may vary. Not clinical medical advice."
        ),
    }


# ---------------------------------------------------------------------------
# Profile CRUD
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_profile(data: dict) -> dict:
    """Construct a full profile dict with defaults and computed fields."""
    now = _now_utc()
    user_id = str(data.get("user_id") or uuid.uuid4())
    d_name = str(data.get("display_name") or data.get("name") or "Unnamed Profile").strip()

    # Normalize legacy frontend goal values to backend canonical values
    _goal_legacy_map = {'maintenance': 'maintain_weight', 'general_health': 'general_wellness'}
    raw_goal = data.get("dietary_goal") or data.get("nutrition_goal")
    if raw_goal in _goal_legacy_map:
        data = dict(data)  # don't mutate caller's dict
        data["dietary_goal"] = _goal_legacy_map[raw_goal]
        data["nutrition_goal"] = _goal_legacy_map[raw_goal]

    # Normalize gender/sex aliases
    gender_val = data.get("gender") or data.get("sex") or ""
    sex_val = data.get("sex") or data.get("gender") or ""

    profile = {
        # Identity
        "user_id": user_id,
        "display_name": d_name,
        "name": d_name,
        "created_at": now,
        "updated_at": now,
        # Demographics
        "age": data.get("age"),
        "gender": gender_val or None,
        "sex": sex_val or None,
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        # Activity & Goal
        "activity_level": data.get("activity_level"),
        "dietary_goal": data.get("dietary_goal") or data.get("nutrition_goal") or "maintain_weight",
        "nutrition_goal": data.get("nutrition_goal") or data.get("dietary_goal") or "maintain_weight",
        # Food preferences
        "dietary_preference": data.get("dietary_preference"),
        "preferred_cuisines": data.get("preferred_cuisines", []),
        "disliked_foods": data.get("disliked_foods", []),
        # Allergies & restrictions
        "allergies": data.get("allergies") or data.get("food_allergies", []),
        "food_allergies": data.get("food_allergies") or data.get("allergies", []),
        "dietary_restrictions": data.get("dietary_restrictions") or data.get("food_restrictions", []),
        "food_restrictions": data.get("food_restrictions") or data.get("dietary_restrictions", []),
        # Lifestyle
        "budget": data.get("budget"),
        "max_cooking_time_mins": data.get("max_cooking_time_mins", 60),
        "dietary_considerations": data.get("dietary_considerations", ""),
        # Morning routine
        "wake_time": data.get("wake_time"),
        "breakfast_preference": data.get("breakfast_preference"),
        "breakfast_time": data.get("breakfast_time"),
    }
    return profile


def create_profile(data: dict) -> dict:
    """
    Create and persist a new user profile.

    Returns the created profile dict.
    Raises ValueError if validation fails.
    """
    ok, errors = validate_profile_data(data)
    if not ok:
        raise ValueError(errors)

    profile = _build_profile(data)
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO user_profiles
              (user_id, display_name, gender, age, created_at, updated_at, profile_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile["user_id"],
                profile["display_name"],
                profile.get("gender"),
                profile.get("age"),
                profile["created_at"],
                profile["updated_at"],
                json.dumps(profile),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return profile


def get_profile(user_id: str) -> Optional[dict]:
    """Return the user profile dict or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT profile_blob FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["profile_blob"])
    finally:
        conn.close()


def update_profile(user_id: str, updates: dict) -> Optional[dict]:
    """
    Apply `updates` to an existing profile.

    Returns updated profile or None if user not found.
    Raises ValueError if validation fails on merged data.
    """
    existing = get_profile(user_id)
    if existing is None:
        return None

    if updates.get("name") and not updates.get("display_name"):
        updates["display_name"] = str(updates["name"]).strip()
    if updates.get("display_name"):
        updates["name"] = str(updates["display_name"]).strip()

    # Merge updates (never overwrite user_id or created_at)
    merged = {**existing, **updates}
    merged["user_id"] = user_id
    merged["created_at"] = existing["created_at"]
    merged["updated_at"] = _now_utc()
    d_name = str(merged.get("display_name") or merged.get("name") or "Unnamed Profile").strip()
    merged["display_name"] = d_name
    merged["name"] = d_name

    ok, errors = validate_profile_data(merged)
    if not ok:
        raise ValueError(errors)

    conn = _get_connection()
    try:
        conn.execute(
            """
            UPDATE user_profiles
            SET display_name = ?, gender = ?, age = ?, updated_at = ?, profile_blob = ?
            WHERE user_id = ?
            """,
            (
                merged["display_name"],
                merged.get("gender") or merged.get("sex"),
                merged.get("age"),
                merged["updated_at"],
                json.dumps(merged),
                user_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return merged


def delete_profile(user_id: str) -> bool:
    """Delete a profile. Returns True if deleted, False if not found."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def list_profiles() -> List[dict]:
    """Return summary list of all profiles with display_name, name, demographics and computed BMI."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT user_id, display_name, gender, age, created_at, profile_blob FROM user_profiles ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for r in rows:
            blob = {}
            if r["profile_blob"]:
                try:
                    blob = json.loads(r["profile_blob"])
                except Exception:
                    pass
            d_name = r["display_name"] or blob.get("display_name") or blob.get("name") or "Unnamed Profile"
            entry = {
                "user_id": r["user_id"],
                "display_name": d_name,
                "name": d_name,
                "gender": r["gender"] or blob.get("gender") or blob.get("sex"),
                "sex": blob.get("sex") or r["gender"] or blob.get("gender"),
                "age": r["age"] or blob.get("age"),
                "height_cm": blob.get("height_cm"),
                "weight_kg": blob.get("weight_kg"),
                "activity_level": blob.get("activity_level"),
                "dietary_preference": blob.get("dietary_preference"),
                "nutrition_goal": blob.get("nutrition_goal") or blob.get("dietary_goal"),
                "created_at": r["created_at"],
            }
            h = blob.get("height_cm")
            w = blob.get("weight_kg")
            if h and w:
                try:
                    hf = float(h)
                    wf = float(w)
                    if hf > 0 and wf > 0:
                        entry["bmi"] = round(wf / ((hf / 100) ** 2), 1)
                except (ValueError, TypeError):
                    pass
            result.append(entry)
        return result
    finally:
        conn.close()


def profile_exists(user_id: str) -> bool:
    """Return True if the user_id exists in the database."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()

