"""
src/database_manager.py
NutriSense AI — Phase 10: Backend and Database Integration

Unified SQLite database management and complete subsystem orchestration:
1. Persistent relational storage:
   - user_profiles: Demographics, preferences, restrictions, and serialized profile blob.
   - foods: Master food nutrition knowledge base (59 items) and unified catalog.
   - nutrients: Reference baseline nutrient values per food.
   - recipes: 5,381 curated recipes with culinary metadata.
   - allergens: Standard allergen definitions and keyword lexicons.
   - meal_logs: Daily logged food items with portion sizes, calories, macros, and timestamps.
   - suitability_evaluations: Historical suitability assessments and verdicts.
   - meal_plans: Saved daily and weekly meal plans with target calories.
   - user_feedback: User ratings and comments on predictions and recommendations.
2. Parameterized SQL queries preventing injection vulnerabilities.
3. Input validation, error handling, rollback on failure, and transaction management.
4. Calculation of daily totals, period averages, and progress against user targets.
5. Integrated End-to-End Orchestrator connecting vision, nutrition, safety, scoring, and persistence.
"""

from datetime import datetime, timezone, date
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = WORKSPACE_ROOT / "database"
DB_PATH = DB_DIR / "nutrisense_users.db"
FEATURE_DIR = WORKSPACE_ROOT / "data" / "processed" / "feature_store"
RECIPE_CSV = FEATURE_DIR / "recipe_feature_store.csv"
KB_JSON = FEATURE_DIR / "master_food_nutrition_kb.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection & Schema Management
# ---------------------------------------------------------------------------
def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Returns an open SQLite database connection with row_factory configured.
    Ensures directory and all relational tables exist.
    """
    target_path = Path(db_path or DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    init_database_schema(conn)
    return conn


def init_database_schema(conn: sqlite3.Connection) -> None:
    """
    Initializes all Phase 10 tables using parameterized, idempotent DDL.
    Preserves all pre-existing tables and data.
    """
    with conn:
        # 1. User profiles (Phase 8 integration)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id      TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                gender       TEXT,
                age          INTEGER,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL,
                profile_blob TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_up_display_name ON user_profiles(display_name);")

        # 2. Master Foods
        conn.execute("""
            CREATE TABLE IF NOT EXISTS foods (
                food_id                TEXT PRIMARY KEY,
                class_name             TEXT NOT NULL,
                display_name           TEXT NOT NULL,
                source                 TEXT NOT NULL,
                match_type             TEXT,
                reference_serving_g    REAL DEFAULT 100.0,
                reference_serving_unit TEXT DEFAULT 'g',
                reference_basis        TEXT,
                sodium_flag            TEXT DEFAULT 'ok',
                calorie_flag           TEXT DEFAULT 'ok',
                created_at             TEXT NOT NULL
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_foods_class ON foods(class_name);")

        # 3. Reference Nutrients per Food
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nutrients (
                nutrient_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                food_id       TEXT NOT NULL,
                calories_kcal REAL,
                protein_g     REAL,
                carbs_g       REAL,
                fat_g         REAL,
                fibre_g       REAL,
                sodium_mg     REAL,
                calcium_mg    REAL,
                iron_mg       REAL,
                vitamin_c_mg  REAL,
                folate_ug     REAL,
                per_basis     TEXT DEFAULT 'per_100g',
                FOREIGN KEY (food_id) REFERENCES foods(food_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nutrients_food_id ON nutrients(food_id);")

        # 4. Curated Recipes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                recipe_id        TEXT PRIMARY KEY,
                recipe_name      TEXT NOT NULL,
                cuisine          TEXT,
                course           TEXT,
                diet             TEXT,
                prep_time_mins   REAL,
                cook_time_mins   REAL,
                total_time_mins  REAL,
                complexity_level TEXT,
                servings         REAL,
                ingredients_raw  TEXT,
                ingredient_tokens TEXT,
                source_dataset   TEXT
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recipes_course ON recipes(course);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_recipes_diet ON recipes(diet);")

        # 5. Standard Allergens Reference
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allergens (
                allergen_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                allergen_name   TEXT UNIQUE NOT NULL,
                common_keywords TEXT NOT NULL
            );
        """)

        # 6. User Meal Logs (Core Tracking)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meal_logs (
                log_id                 TEXT PRIMARY KEY,
                user_id                TEXT NOT NULL,
                meal_date              TEXT NOT NULL,
                meal_time              TEXT NOT NULL,
                course                 TEXT NOT NULL,
                food_name              TEXT NOT NULL,
                food_id                TEXT,
                portion_grams          REAL NOT NULL,
                portion_unit           TEXT DEFAULT 'g',
                calories_kcal          REAL NOT NULL,
                protein_g              REAL,
                carbs_g                REAL,
                fat_g                  REAL,
                fibre_g                REAL,
                sodium_mg              REAL,
                image_path             TEXT,
                recognition_confidence REAL,
                notes                  TEXT,
                created_at             TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meal_logs_user_date ON meal_logs(user_id, meal_date);")

        # 7. Historical Suitability Evaluations
        conn.execute("""
            CREATE TABLE IF NOT EXISTS suitability_evaluations (
                eval_id                  TEXT PRIMARY KEY,
                user_id                  TEXT NOT NULL,
                food_or_recipe_id        TEXT NOT NULL,
                item_name                TEXT NOT NULL,
                item_type                TEXT NOT NULL,
                suitability_score        REAL NOT NULL,
                verdict                  TEXT NOT NULL,
                component_breakdown_json TEXT,
                safety_status            TEXT DEFAULT 'SAFE',
                created_at               TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evals_user ON suitability_evaluations(user_id);")

        # 8. Persisted Meal Plans
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meal_plans (
                plan_id              TEXT PRIMARY KEY,
                user_id              TEXT NOT NULL,
                plan_type            TEXT NOT NULL,
                target_calories_kcal REAL NOT NULL,
                plan_data_json       TEXT NOT NULL,
                status               TEXT DEFAULT 'active',
                created_at           TEXT NOT NULL,
                updated_at           TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_meal_plans_user ON meal_plans(user_id);")

        # 9. User Feedback & Model Audit
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id   TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                item_id       TEXT,
                item_type     TEXT NOT NULL,
                rating        INTEGER CHECK(rating >= 1 AND rating <= 5),
                feedback_type TEXT,
                comments      TEXT,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id);")


# ---------------------------------------------------------------------------
# Data Seeding Functions (Baseline KB & Recipes)
# ---------------------------------------------------------------------------
def seed_reference_data_if_empty(conn: sqlite3.Connection) -> Dict[str, int]:
    """
    Populates foods, nutrients, recipes, and allergens from feature store files
    if tables are currently empty. Returns count of seeded rows.
    """
    seeded_counts = {"foods": 0, "nutrients": 0, "recipes": 0, "allergens": 0}

    # 1. Seed Allergens
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM allergens;")
    if cur.fetchone()[0] == 0:
        from src.recommendation_engine import ALLERGEN_LEXICON
        allergen_rows = [
            (allergen, ", ".join(sorted(keywords)))
            for allergen, keywords in ALLERGEN_LEXICON.items()
        ]
        with conn:
            conn.executemany(
                "INSERT INTO allergens (allergen_name, common_keywords) VALUES (?, ?);",
                allergen_rows
            )
        seeded_counts["allergens"] = len(allergen_rows)

    # 2. Seed Foods and Nutrients
    cur.execute("SELECT COUNT(*) FROM foods;")
    if cur.fetchone()[0] == 0 and KB_JSON.exists():
        with open(KB_JSON, "r", encoding="utf-8") as f:
            kb_items = json.load(f)

        now_str = datetime.now(timezone.utc).isoformat()
        food_rows = []
        nutrient_rows = []

        for item in kb_items:
            fid = item.get("food_id") or item.get("class_name")
            cname = item.get("class_name", "")
            dname = item.get("display_name", cname.title())
            src = item.get("data_source", "ICMR_NIN_MASTER_KB")
            mtype = item.get("match_type", "VALID_DIRECT")
            ref_g = float(item.get("serving_size_g", 100.0) or 100.0)
            ref_unit = str(item.get("serving_unit", "standard serving"))
            s_flag = str(item.get("sodium_flag", "ok"))
            c_flag = str(item.get("calorie_flag", "ok"))

            food_rows.append((fid, cname, dname, src, mtype, ref_g, ref_unit, f"Per {ref_g}g {ref_unit}", s_flag, c_flag, now_str))

            # Normalize to per 100g for nutrients table
            factor = 100.0 / ref_g if ref_g > 0 else 1.0
            def _val(k):
                v = item.get(k)
                return round(float(v) * factor, 3) if v is not None and not pd.isna(v) else None

            nutrient_rows.append((
                fid,
                _val("calories_kcal"),
                _val("protein_g"),
                _val("carbs_g"),
                _val("fat_g"),
                _val("fibre_g"),
                _val("sodium_mg"),
                _val("calcium_mg"),
                _val("iron_mg"),
                _val("vitamin_c_mg"),
                _val("folate_ug"),
                "per_100g"
            ))

        with conn:
            conn.executemany("""
                INSERT OR IGNORE INTO foods (
                    food_id, class_name, display_name, source, match_type,
                    reference_serving_g, reference_serving_unit, reference_basis,
                    sodium_flag, calorie_flag, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, food_rows)

            conn.executemany("""
                INSERT INTO nutrients (
                    food_id, calories_kcal, protein_g, carbs_g, fat_g, fibre_g,
                    sodium_mg, calcium_mg, iron_mg, vitamin_c_mg, folate_ug, per_basis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, nutrient_rows)

        seeded_counts["foods"] = len(food_rows)
        seeded_counts["nutrients"] = len(nutrient_rows)

    # 3. Seed Recipes
    cur.execute("SELECT COUNT(*) FROM recipes;")
    if cur.fetchone()[0] == 0 and RECIPE_CSV.exists():
        df_rec = pd.read_csv(RECIPE_CSV)
        recipe_rows = []
        for _, r in df_rec.iterrows():
            recipe_rows.append((
                str(r.get("recipe_id", "")),
                str(r.get("recipe_name", "")),
                str(r.get("cuisine", "")),
                str(r.get("course", "")),
                str(r.get("diet", "")),
                float(r["prep_time_mins"]) if pd.notnull(r.get("prep_time_mins")) else None,
                float(r["cook_time_mins"]) if pd.notnull(r.get("cook_time_mins")) else None,
                float(r["total_time_mins"]) if pd.notnull(r.get("total_time_mins")) else None,
                str(r.get("complexity_level", "MODERATE")),
                float(r["servings"]) if pd.notnull(r.get("servings")) else None,
                str(r.get("ingredients_raw", "")) if pd.notnull(r.get("ingredients_raw")) else "",
                str(r.get("ingredient_tokens", "")) if pd.notnull(r.get("ingredient_tokens")) else "",
                str(r.get("source_dataset", ""))
            ))

        with conn:
            conn.executemany("""
                INSERT OR IGNORE INTO recipes (
                    recipe_id, recipe_name, cuisine, course, diet,
                    prep_time_mins, cook_time_mins, total_time_mins, complexity_level,
                    servings, ingredients_raw, ingredient_tokens, source_dataset
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, recipe_rows)
        seeded_counts["recipes"] = len(recipe_rows)

    return seeded_counts


# ---------------------------------------------------------------------------
# Meal Logging Operations
# ---------------------------------------------------------------------------
def _ensure_user_exists(conn: sqlite3.Connection, user_id: str) -> None:
    """Ensures a record exists in user_profiles to satisfy foreign key integrity."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM user_profiles WHERE user_id = ?;", (user_id,))
    if not cur.fetchone():
        now = datetime.now(timezone.utc).isoformat()
        disp_name = f"User {user_id[:8]}" if len(user_id) >= 8 else f"User {user_id}"
        conn.execute("""
            INSERT OR IGNORE INTO user_profiles (
                user_id, display_name, created_at, updated_at, profile_blob
            ) VALUES (?, ?, ?, ?, ?);
        """, (user_id, disp_name, now, now, json.dumps({"user_id": user_id, "display_name": disp_name})))


def log_meal(
    user_id: str,
    meal_data: Dict[str, Any],
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Saves a logged meal entry using parameterized queries with strict validation.
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a valid non-empty string.")

    # Validate mandatory fields
    food_name = meal_data.get("food_name", "").strip()
    if not food_name:
        raise ValueError("food_name is required in meal_data.")

    try:
        portion_g = float(meal_data.get("portion_grams", 100.0))
        if portion_g <= 0 or math.isnan(portion_g) or math.isinf(portion_g):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("portion_grams must be a positive finite number.")

    try:
        cal = float(meal_data.get("calories_kcal", 0.0))
        if cal < 0 or math.isnan(cal) or math.isinf(cal):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("calories_kcal must be a non-negative number.")

    # Optional nutritional macros
    def _float_opt(k):
        v = meal_data.get(k)
        if v is None:
            return None
        try:
            val = float(v)
            return round(val, 2) if not (math.isnan(val) or math.isinf(val)) else None
        except (ValueError, TypeError):
            return None

    log_id = meal_data.get("log_id") or str(uuid.uuid4())
    meal_date = meal_data.get("meal_date") or date.today().isoformat()
    meal_time = meal_data.get("meal_time") or datetime.now().strftime("%H:%M:%S")
    course = str(meal_data.get("course", "lunch")).lower()
    now_utc = datetime.now(timezone.utc).isoformat()

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        with conn:
            _ensure_user_exists(conn, user_id)
            conn.execute("""
                INSERT INTO meal_logs (
                    log_id, user_id, meal_date, meal_time, course, food_name, food_id,
                    portion_grams, portion_unit, calories_kcal, protein_g, carbs_g, fat_g,
                    fibre_g, sodium_mg, image_path, recognition_confidence, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                log_id,
                user_id,
                meal_date,
                meal_time,
                course,
                food_name,
                meal_data.get("food_id"),
                portion_g,
                str(meal_data.get("portion_unit", "g")),
                round(cal, 2),
                _float_opt("protein_g"),
                _float_opt("carbs_g"),
                _float_opt("fat_g"),
                _float_opt("fibre_g"),
                _float_opt("sodium_mg"),
                meal_data.get("image_path"),
                _float_opt("recognition_confidence"),
                meal_data.get("notes"),
                now_utc,
            ))

        return {
            "log_id": log_id,
            "user_id": user_id,
            "meal_date": meal_date,
            "meal_time": meal_time,
            "course": course,
            "food_name": food_name,
            "portion_grams": portion_g,
            "calories_kcal": round(cal, 2),
            "protein_g": _float_opt("protein_g"),
            "carbs_g": _float_opt("carbs_g"),
            "fat_g": _float_opt("fat_g"),
            "fibre_g": _float_opt("fibre_g"),
            "sodium_mg": _float_opt("sodium_mg"),
            "created_at": now_utc,
        }
    finally:
        if close_conn:
            conn.close()


def get_meal_history(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    course: Optional[str] = None,
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves filtered meal logs for a user using parameterized queries.
    """
    if not user_id:
        return []

    query = "SELECT * FROM meal_logs WHERE user_id = ?"
    params: List[Any] = [user_id]

    if start_date:
        query += " AND meal_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND meal_date <= ?"
        params.append(end_date)
    if course:
        query += " AND LOWER(course) = ?"
        params.append(course.lower().strip())

    query += " ORDER BY meal_date DESC, meal_time DESC LIMIT ?"
    params.append(max(1, int(limit)))

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        if close_conn:
            conn.close()


def delete_meal_log(
    log_id: str,
    user_id: str,
    conn: Optional[sqlite3.Connection] = None
) -> bool:
    """
    Deletes a specific meal log entry. Requires user_id verification.
    """
    if not log_id or not user_id:
        return False

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM meal_logs WHERE log_id = ? AND user_id = ?;",
                (log_id, user_id)
            )
            return cur.rowcount > 0
    finally:
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Daily Totals & Nutritional Aggregations
# ---------------------------------------------------------------------------
def get_daily_nutrition_totals(
    user_id: str,
    target_date: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Calculates aggregated nutrient totals from recorded meals on a specific date,
    and compares against the user's daily dietary targets.
    """
    if not user_id:
        raise ValueError("user_id is required.")

    query_date = target_date or date.today().isoformat()

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        # 1. Fetch meal logs for the given day
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as meal_count,
                COALESCE(SUM(calories_kcal), 0.0) as total_calories,
                COALESCE(SUM(protein_g), 0.0) as total_protein,
                COALESCE(SUM(carbs_g), 0.0) as total_carbs,
                COALESCE(SUM(fat_g), 0.0) as total_fat,
                COALESCE(SUM(fibre_g), 0.0) as total_fibre,
                COALESCE(SUM(sodium_mg), 0.0) as total_sodium
            FROM meal_logs
            WHERE user_id = ? AND meal_date = ?;
        """, (user_id, query_date))
        agg = dict(cur.fetchone())

        # 2. Fetch meals list for the day
        cur.execute("""
            SELECT log_id, meal_time, course, food_name, portion_grams, calories_kcal, protein_g, carbs_g, fat_g
            FROM meal_logs
            WHERE user_id = ? AND meal_date = ?
            ORDER BY meal_time ASC;
        """, (user_id, query_date))
        day_meals = [dict(r) for r in cur.fetchall()]

        # 3. Retrieve user targets if profile exists
        from src.user_profile_manager import get_profile, compute_daily_calorie_target
        profile = get_profile(user_id)
        target_calories = 2000.0  # Default standard reference
        target_source = "POPULATION_REFERENCE_DEFAULT"

        if profile:
            try:
                cal_target_dict = compute_daily_calorie_target(
                    profile["age"], profile["gender"], profile["height_cm"],
                    profile["weight_kg"], profile["activity_level"], profile["dietary_goal"]
                )
                target_calories = float(cal_target_dict["target_calories_kcal"])
                target_source = "PERSONALIZED_MIFFLIN_ST_JEOR"
            except Exception:
                target_calories = 2000.0

        consumed_cal = round(agg["total_calories"], 1)
        remaining_cal = round(target_calories - consumed_cal, 1)
        pct_cal = round((consumed_cal / target_calories) * 100.0, 1) if target_calories > 0 else 0.0

        # Health alerts
        alerts = []
        if agg["total_sodium"] > 2000.0:
            alerts.append({
                "severity": "WARNING",
                "message": f"Daily sodium intake ({agg['total_sodium']:.0f} mg) exceeds WHO guideline of 2,000 mg."
            })
        if pct_cal > 115.0:
            alerts.append({
                "severity": "INFO",
                "message": f"Calories consumed ({consumed_cal:.0f} kcal) exceed daily target by {pct_cal - 100:.1f}%."
            })

        return {
            "user_id": user_id,
            "date": query_date,
            "meals_logged_count": agg["meal_count"],
            "totals": {
                "calories_kcal": consumed_cal,
                "protein_g": round(agg["total_protein"], 1),
                "carbs_g": round(agg["total_carbs"], 1),
                "fat_g": round(agg["total_fat"], 1),
                "fibre_g": round(agg["total_fibre"], 1),
                "sodium_mg": round(agg["total_sodium"], 1),
            },
            "targets": {
                "calories_kcal": target_calories,
                "target_source": target_source,
                "remaining_calories_kcal": remaining_cal,
                "percent_target_achieved": pct_cal,
            },
            "alerts": alerts,
            "meals": day_meals,
        }
    finally:
        if close_conn:
            conn.close()


def get_period_nutrition_summary(
    user_id: str,
    days: int = 7,
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Computes average daily consumption and macro distribution over the past N days.
    """
    if not user_id:
        raise ValueError("user_id is required.")

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                meal_date,
                COUNT(*) as meals_count,
                SUM(calories_kcal) as day_calories,
                SUM(protein_g) as day_protein,
                SUM(carbs_g) as day_carbs,
                SUM(fat_g) as day_fat,
                SUM(fibre_g) as day_fibre
            FROM meal_logs
            WHERE user_id = ?
            GROUP BY meal_date
            ORDER BY meal_date DESC
            LIMIT ?;
        """, (user_id, max(1, int(days))))
        day_rows = [dict(r) for r in cur.fetchall()]

        total_days = len(day_rows)
        if total_days == 0:
            return {
                "user_id": user_id,
                "period_days": days,
                "active_logged_days": 0,
                "averages": {
                    "calories_kcal": 0.0,
                    "protein_g": 0.0,
                    "carbs_g": 0.0,
                    "fat_g": 0.0,
                    "fibre_g": 0.0
                },
                "daily_breakdown": []
            }

        avg_cal = round(sum(d["day_calories"] or 0 for d in day_rows) / total_days, 1)
        avg_prot = round(sum(d["day_protein"] or 0 for d in day_rows) / total_days, 1)
        avg_carbs = round(sum(d["day_carbs"] or 0 for d in day_rows) / total_days, 1)
        avg_fat = round(sum(d["day_fat"] or 0 for d in day_rows) / total_days, 1)
        avg_fibre = round(sum(d["day_fibre"] or 0 for d in day_rows) / total_days, 1)

        return {
            "user_id": user_id,
            "period_days": days,
            "active_logged_days": total_days,
            "averages": {
                "calories_kcal": avg_cal,
                "protein_g": avg_prot,
                "carbs_g": avg_carbs,
                "fat_g": avg_fat,
                "fibre_g": avg_fibre,
            },
            "daily_breakdown": day_rows
        }
    finally:
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Meal Plan Persistence & Feedback Operations
# ---------------------------------------------------------------------------
def save_meal_plan(
    user_id: str,
    plan_data: Dict[str, Any],
    plan_type: str = "daily",
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Saves a generated daily or weekly meal plan to the database.
    """
    if not user_id or not plan_data:
        raise ValueError("user_id and plan_data are required.")

    plan_id = str(uuid.uuid4())
    target_cals = float(plan_data.get("daily_target_calories_kcal", 2000.0) or 2000.0)
    now_utc = datetime.now(timezone.utc).isoformat()
    plan_json = json.dumps(plan_data)

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        with conn:
            _ensure_user_exists(conn, user_id)
            conn.execute("""
                INSERT INTO meal_plans (
                    plan_id, user_id, plan_type, target_calories_kcal,
                    plan_data_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?);
            """, (plan_id, user_id, plan_type, target_cals, plan_json, now_utc, now_utc))

        return {
            "plan_id": plan_id,
            "user_id": user_id,
            "plan_type": plan_type,
            "target_calories_kcal": target_cals,
            "created_at": now_utc
        }
    finally:
        if close_conn:
            conn.close()


def get_latest_meal_plan(
    user_id: str,
    plan_type: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetches the most recent active meal plan for a user.
    """
    if not user_id:
        return None

    query = "SELECT * FROM meal_plans WHERE user_id = ?"
    params: List[Any] = [user_id]
    if plan_type:
        query += " AND plan_type = ?"
        params.append(plan_type)
    query += " ORDER BY created_at DESC LIMIT 1;"

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["plan_data"] = json.loads(res["plan_data_json"])
        return res
    finally:
        if close_conn:
            conn.close()


def log_user_feedback(
    user_id: str,
    feedback_data: Dict[str, Any],
    conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Stores user feedback regarding food recognition, suitability scores, or recommendations.
    """
    if not user_id:
        raise ValueError("user_id is required.")

    rating = feedback_data.get("rating")
    if rating is not None:
        try:
            r_int = int(rating)
            if not (1 <= r_int <= 5):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("rating must be an integer between 1 and 5.")

    fid = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc).isoformat()

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        with conn:
            _ensure_user_exists(conn, user_id)
            conn.execute("""
                INSERT INTO user_feedback (
                    feedback_id, user_id, item_id, item_type, rating,
                    feedback_type, comments, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                fid,
                user_id,
                feedback_data.get("item_id"),
                str(feedback_data.get("item_type", "general")),
                rating,
                feedback_data.get("feedback_type"),
                feedback_data.get("comments"),
                now_utc
            ))

        return {
            "feedback_id": fid,
            "user_id": user_id,
            "rating": rating,
            "created_at": now_utc
        }
    finally:
        if close_conn:
            conn.close()


def get_user_feedback(
    user_id: str,
    limit: int = 20,
    conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves logged feedback entries for a user.
    """
    if not user_id:
        return []

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM user_feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT ?;",
            (user_id, max(1, int(limit)))
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        if close_conn:
            conn.close()


# ---------------------------------------------------------------------------
# Integrated Data Flow Orchestrator
# ---------------------------------------------------------------------------
class BackendOrchestrator:
    """
    End-to-End Orchestrator integrating:
    1. Computer vision recognition or direct food identification
    2. Nutrient calculation & portion scaling
    3. User profile retrieval & safety filtering (allergies, restrictions)
    4. 0–100 suitability scoring with human explanation
    5. Alternative foods suggestions
    6. Database persistence into meal_logs & updated daily summary
    """

    def __init__(self):
        self.conn = get_db_connection()
        seed_reference_data_if_empty(self.conn)

    def process_food_recognition_and_logging(
        self,
        food_name_or_prediction: str,
        portion_value: float,
        unit: str = "g",
        user_id: Optional[str] = None,
        course: str = "lunch",
        save_to_log: bool = False,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete data pipeline across all modules.
        """
        from src.nutrition_analyzer import lookup_food, calculate_nutrients, convert_portion_to_grams
        from src.user_profile_manager import get_profile
        from src.recommendation_engine import SuitabilityScorer, SafetyFilterEngine, AlternativeFoodsEngine

        # 1. Lookup food nutrition record
        food_record = lookup_food(food_name_or_prediction)
        if not food_record:
            raise KeyError(f"Food item '{food_name_or_prediction}' not found in nutrition database.")

        # 2. Convert portion to grams and scale nutrients
        portion_g = convert_portion_to_grams(portion_value, unit=unit, food_name=food_record["class_name"])
        nutrients_scaled = calculate_nutrients(food_record, portion_g)

        # 3. Retrieve user profile
        profile = get_profile(user_id) if user_id else {}

        # 4. Safety filtering (check for allergies & restrictions)
        safety_engine = SafetyFilterEngine()
        recipe_proxy = {
            "recipe_name": food_record["display_name"],
            "ingredients_raw": food_record.get("reference_basis", ""),
            "ingredient_tokens": food_record["class_name"].replace("_", " "),
            "diet": "vegetarian" if "meat" not in food_record["class_name"] else "non_vegetarian",
        }
        safety_status = safety_engine.check_safety(recipe_proxy, profile)

        # 5. Transparent 0–100 Suitability Scoring
        scorer = SuitabilityScorer()
        score_eval = scorer.score_recipe(recipe_proxy, profile)

        # 6. Alternative foods (if score is low or item is fried/dense)
        alt_engine = AlternativeFoodsEngine()
        alternatives = alt_engine.get_alternatives(food_record["class_name"], user_profile=profile)

        # 7. Meal Logging into Database (if requested)
        logged_record = None
        updated_daily_totals = None
        if save_to_log and user_id:
            nut = nutrients_scaled["nutrients"]
            log_payload = {
                "food_name": food_record["display_name"],
                "food_id": food_record["food_id"],
                "course": course,
                "portion_grams": portion_g,
                "portion_unit": unit,
                "calories_kcal": nut["calories_kcal"]["value"] or 0.0,
                "protein_g": nut["protein_g"]["value"] or 0.0,
                "carbs_g": nut["carbs_g"]["value"] or 0.0,
                "fat_g": nut["fat_g"]["value"] or 0.0,
                "fibre_g": nut["fibre_g"]["value"] or 0.0,
                "sodium_mg": nut["sodium_mg"]["value"] or 0.0,
                "notes": notes,
            }
            logged_record = log_meal(user_id, log_payload, conn=self.conn)
            updated_daily_totals = get_daily_nutrition_totals(user_id, conn=self.conn)

        return {
            "success": True,
            "food_record": {
                "food_id": food_record["food_id"],
                "display_name": food_record["display_name"],
                "class_name": food_record["class_name"],
                "source": food_record["source"],
            },
            "portion": {
                "input_value": portion_value,
                "unit": unit,
                "grams": portion_g,
            },
            "nutrients": nutrients_scaled["nutrients"],
            "safety": safety_status,
            "suitability_evaluation": score_eval,
            "healthier_alternatives": alternatives,
            "logged_meal": logged_record,
            "updated_daily_totals": updated_daily_totals,
        }


# Module Singleton
_orchestrator: Optional[BackendOrchestrator] = None

def get_backend_orchestrator() -> BackendOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BackendOrchestrator()
    return _orchestrator
