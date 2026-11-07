# Phase 10: Backend and Database Integration — Comprehensive Report

## 1. Executive Summary & Architecture Overview

Phase 10 delivers the **Backend and Database Integration Layer** for NutriSense AI. It binds all previous subsystems—Computer Vision (`inference_pipeline.py`), Nutrition Analysis (`nutrition_analyzer.py`), User Profiling (`user_profile_manager.py`), and the Recommendation Engine (`recommendation_engine.py`)—into an integrated, production-grade SQLite architecture.

```
+---------------------------------------------------------------------------------------------------+
|                                 NUTRISENSE UNIFIED BACKEND ARCHITECTURE                           |
+---------------------------------------------------------------------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
        +--------------------------+                              +--------------------------+
        | REST API Controllers     |                              | Backend Orchestrator     |
        | - /api/meals/*           |                              | - End-to-End Pipeline    |
        | - /api/meal-plans/*      |                              | - Vision -> Nutrition    |
        | - /api/feedback/*        |                              | - Safety -> Scoring      |
        | - /api/flow/process      |                              | - Dynamic Logging        |
        +--------------------------+                              +--------------------------+
                     |                                                         |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     | Database Manager (src/database_manager.py)              |
                     | - Parameterized SQL Engine (Zero Injection Risk)        |
                     | - Schema Migrations & Auto-Seeding                      |
                     | - ACID Transaction & Rollback Handlers                  |
                     +---------------------------------------------------------+
                                                  |
                                                  v
                     +---------------------------------------------------------+
                     | Relational SQLite Storage (database/nutrisense_users.db)|
                     | - user_profiles           - meal_logs                   |
                     | - foods                   - suitability_evaluations     |
                     | - nutrients               - meal_plans                  |
                     | - recipes                 - user_feedback               |
                     | - allergens                                             |
                     +---------------------------------------------------------+
```

---

## 2. Relational Database Schema Specification

All tables are maintained in `database/nutrisense_users.db` with strict foreign key constraints and indexed queries:

```sql
-- 1. User Profiles
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id      TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    gender       TEXT,
    age          INTEGER,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    profile_blob TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_up_display_name ON user_profiles(display_name);

-- 2. Master Foods
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
CREATE INDEX IF NOT EXISTS idx_foods_class ON foods(class_name);

-- 3. Baseline Nutrients (Normalized per 100g)
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
CREATE INDEX IF NOT EXISTS idx_nutrients_food_id ON nutrients(food_id);

-- 4. Curated Recipes Feature Store (5,381 recipes)
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id         TEXT PRIMARY KEY,
    recipe_name       TEXT NOT NULL,
    cuisine           TEXT,
    course            TEXT,
    diet              TEXT,
    prep_time_mins    REAL,
    cook_time_mins    REAL,
    total_time_mins   REAL,
    complexity_level  TEXT,
    servings          REAL,
    ingredients_raw   TEXT,
    ingredient_tokens TEXT,
    source_dataset    TEXT
);
CREATE INDEX IF NOT EXISTS idx_recipes_course ON recipes(course);
CREATE INDEX IF NOT EXISTS idx_recipes_diet ON recipes(diet);

-- 5. Standard Allergens Reference
CREATE TABLE IF NOT EXISTS allergens (
    allergen_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    allergen_name   TEXT UNIQUE NOT NULL,
    common_keywords TEXT NOT NULL
);

-- 6. User Meal Logs (Core Nutrition Tracking)
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
CREATE INDEX IF NOT EXISTS idx_meal_logs_user_date ON meal_logs(user_id, meal_date);

-- 7. Historical Suitability Evaluations
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
CREATE INDEX IF NOT EXISTS idx_evals_user ON suitability_evaluations(user_id);

-- 8. Persisted Meal Plans (Daily & Weekly)
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
CREATE INDEX IF NOT EXISTS idx_meal_plans_user ON meal_plans(user_id);

-- 9. User Feedback & Audit
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
CREATE INDEX IF NOT EXISTS idx_feedback_user ON user_feedback(user_id);
```

---

## 3. Data Protection, Query Parameterization & Error Handling

1. **SQL Injection Immunity**: 100% of read and write queries use parameterized statement bindings (`?` positional parameters). SQL injection probes (e.g. `' OR '1'='1`) safely return zero results without syntax failure.
2. **Strict Validation**: Portion grams must be strictly positive and finite ($> 0$, non-NaN, non-infinite); calories must be non-negative; user feedback ratings must fall strictly within the integer range $[1, 5]$.
3. **Transaction Safety**: All modifications utilize SQLite context managers (`with conn:`) guaranteeing atomic execution and automatic rollback upon error.
4. **Foreign Key Integrity**: Dynamic user verification automatically resolves foreign key dependencies for guest sessions or direct meal logging.

---

## 4. Analytical Aggregations & Daily Summaries

The system computes real-time nutritional totals and comparisons:

### A. Daily Totals Formulation
For any date $D$ and user $U$:
$$\text{Calories}_{\text{total}} = \sum_{m \in \text{Meals}(U, D)} \text{Calories}_m, \quad \text{Protein}_{\text{total}} = \sum_{m} \text{Protein}_m, \quad \text{Sodium}_{\text{total}} = \sum_{m} \text{Sodium}_m$$
$$\text{Remaining Calories} = \text{Target Calories}(U) - \text{Calories}_{\text{total}}$$
$$\text{Percent Target Achieved} = \frac{\text{Calories}_{\text{total}}}{\text{Target Calories}(U)} \times 100\%$$

### B. Health Guardrail Alerts
- **Sodium Alert**: Triggered whenever $\text{Sodium}_{\text{total}} > 2,000\,\text{mg}$ (WHO daily upper limit).
- **Caloric Surplus Alert**: Triggered whenever $\text{Percent Target} > 115\%$.

---

## 5. End-to-End Integrated Orchestrator Flow

The `BackendOrchestrator` unifies the entire NutriSense pipeline in a single execution:
1. **Food Identification**: Resolves food record from image inference or text query.
2. **Nutrition Scaling**: Computes exact macro/micronutrient breakdown for the given portion.
3. **User Profile Verification**: Checks declared allergies and dietary restrictions.
4. **Transparent Scoring**: Evaluates the 0–100 suitability score with component breakdown.
5. **Alternative Recommendations**: Surfaces healthier culinary alternatives if applicable.
6. **Persistence**: Saves logged meal entry to SQLite and outputs updated daily progress.

---

## 6. REST API Endpoints Specification

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/meals/log` | Save a meal log into SQLite | `201`, `400`, `500` |
| `GET` | `/api/meals/history` | Retrieve filtered meal history (by date, course, user) | `200`, `400`, `500` |
| `DELETE` | `/api/meals/<log_id>` | Delete a meal log with user authorization | `200`, `400`, `404`, `500` |
| `GET` | `/api/meals/daily-totals` | Daily nutrient totals, remaining calories & alerts | `200`, `400`, `500` |
| `GET` | `/api/meals/summary` | Multi-day consumption averages & daily breakdown | `200`, `400`, `500` |
| `POST` | `/api/meal-plans/save` | Persist daily or weekly meal plans | `201`, `400`, `500` |
| `GET` | `/api/meal-plans/latest` | Retrieve user's latest meal plan | `200`, `400`, `404`, `500` |
| `POST` | `/api/feedback/log` | Submit user rating (1–5) and comments | `201`, `400`, `500` |
| `GET` | `/api/feedback` | Retrieve user feedback history | `200`, `400`, `500` |
| `POST` | `/api/flow/process` | Full end-to-end integrated food analysis & logging | `200`, `400`, `404`, `500` |

---

## 7. Verification & Test Suite Results

The dedicated integration test suite [`tests/test_backend_database_integration.py`](file:///C:/Samsung_Project/tests/test_backend_database_integration.py) validated all functional specifications:

| Section | Scope | Tests Run | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| **Section 1** | Relational Database Schema & Table Verification | 11 | 11 | 0 |
| **Section 2** | Baseline Reference Data Seeding (Foods, Nutrients, Recipes) | 5 | 5 | 0 |
| **Section 3** | Parameterized Queries & SQL Injection Immunity | 2 | 2 | 0 |
| **Section 4** | Input Validation & Error Handling (NaN, negative values) | 5 | 5 | 0 |
| **Section 5** | Meal Logging, Course Filtering, History & Deletion | 9 | 9 | 0 |
| **Section 6** | Daily Totals, Target Calibration & Sodium Alerts | 10 | 10 | 0 |
| **Section 7** | Meal Plan Persistence & JSON Deserialization | 5 | 5 | 0 |
| **Section 8** | User Feedback System (1–5 Star Validation) | 5 | 5 | 0 |
| **Section 9** | End-to-End Integrated Orchestrator Flow | 10 | 10 | 0 |
| **Section 10** | Web REST API Endpoints Integration | 16 | 16 | 0 |
| **Section 11** | Preservation of Existing Features (Phases 1–9) | 3 | 3 | 0 |
| **Total** | **Phase 10 Verification Suite** | **81** | **81** | **0** |

---

## 8. Master Regression Scorecard

Execution of `python run_all_tests.py` confirmed 100% test passage across all functional project phases:

```
Test Suite                                       | Status   | Passed/Total   | Time  
---------------------------------------------------------------------------
Phase 1: Preprocessing & Data Cleaning           | OK       | 60/60          |  0.70s
Phase 2: Review & Anomaly Remediation            | OK       | 45/45          |  0.54s
Phase 3: Exploratory Data Analysis (EDA)         | OK       | 57/57          |  0.57s
Phase 4: Feature Engineering & Knowledge Base    | OK       | 87/87          |  0.65s
Phase 5: AI Model Training & Inference Engine    | OK       | 52/52          |  5.01s
Phase 6: Web Application & REST APIs             | OK       | 48/48          |  5.47s
Phase 7: Nutrition Analysis Module               | OK       | 63/63          |  4.48s
Phase 8: Personalization & User Profiling        | OK       | 140/140        |  4.54s
Phase 9: Recommendation System                   | OK       | 106/106        | 69.00s
Phase 10: Backend & Database Integration         | OK       | 81/81          |  4.12s
---------------------------------------------------------------------------
Total Tests Verified: 739 Passed / 739 Run across Phases 1–10
```

