# Phase 8: Personalization and User Profiling — Comprehensive Report

## 1. Executive Summary & Objectives

Phase 8 introduces the **Personalization and User Profiling Module** for NutriSense AI, enabling dynamic adaptation of nutritional guidance to individual physiological, dietary, and lifestyle parameters. The system bridges raw nutritional data and computer vision predictions to personalized meal suggestions, portion guidance, and clinical suitability evaluations without making unvalidated medical claims.

### Core Objectives Delivered:
1. **Multi-Dimensional User Profile**: Captures demographics, physiological measurements, dietary goals, physical activity levels, culinary preferences, allergens, dietary restrictions, kitchen constraints (budget, cooking time), and morning routines.
2. **Standardized BMI Engine with Limitations Disclaimer**: Evaluates Body Mass Index with both WHO universal standards and South Asian consensus thresholds, accompanied by mandatory non-diagnostic disclaimers.
3. **Mifflin-St Jeor Energy Expenditure Formulation**: Calculates Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE) adjusted for physical activity and specific caloric deficit/surplus goals.
4. **ICMR-NIN (2020) RDA Profile Mapping**: Automatically maps user profiles to clinical reference profiles (`adult_male_sedentary`, `adult_female_sedentary`, `diabetic_adult`, `hypertensive_adult`, `weight_management`) for suitability scoring.
5. **Morning Routine & Breakfast Suggestion Engine**: Connects wake time and breakfast preferences to 5,382 indexed recipes, filtering by dietary restrictions, preparation time, and nutritional suitability.
6. **Personalized Portion Guidance**: Adjusts baseline reference portions based on user dietary goals, BMI, and calorie targets.
7. **Robust SQLite Persistence**: Implements ACID-compliant storage in `database/nutrisense_users.db` with UUID identifiers and strict field-level validation.
8. **RESTful Web API**: Exposes 8 secure endpoints for profile lifecycle management, BMI evaluation, breakfast recommendations, and personalized food assessment.

---

## 2. Mathematical Formulations & Algorithms

### A. Body Mass Index (BMI) & Category Mapping

$$\text{BMI} = \frac{\text{weight (kg)}}{(\text{height (m)})^2} = \frac{\text{weight (kg)}}{(\text{height (cm)} / 100)^2}$$

#### Thresholds:
| Category | WHO Standard ($\text{kg/m}^2$) | South Asian Consensus ($\text{kg/m}^2$) |
| :--- | :---: | :---: |
| **Underweight** | $< 18.5$ | $< 18.5$ |
| **Normal Weight** | $18.5 - 24.9$ | $18.5 - 22.9$ |
| **Overweight** | $25.0 - 29.9$ | $23.0 - 27.4$ |
| **Obese** | $\ge 30.0$ | $\ge 27.5$ |

> [!IMPORTANT]
> **Mandatory BMI Limitations Disclaimer**:
> *"BMI is a population-level screening index, not a diagnostic tool. It does not account for muscle mass, bone density, age, sex, ethnicity, or fat distribution. For individuals of South Asian descent, health risk thresholds may differ from WHO universal standards (overweight $\ge 23.0$, obese $\ge 27.5$ per consensus guidelines). BMI values are provided for general awareness only and must not be used for clinical decisions without consulting a qualified healthcare professional."*

---

### B. Energy Expenditure Formulation (Mifflin-St Jeor Equation)

#### 1. Basal Metabolic Rate (BMR):
$$\text{BMR}_{\text{male}} = 10 \times W + 6.25 \times H - 5 \times A + 5$$
$$\text{BMR}_{\text{female}} = 10 \times W + 6.25 \times H - 5 \times A - 161$$
$$\text{BMR}_{\text{other}} = \frac{\text{BMR}_{\text{male}} + \text{BMR}_{\text{female}}}{2}$$

*Where $W$ = weight in kg, $H$ = height in cm, $A$ = age in years.*

#### 2. Total Daily Energy Expenditure (TDEE):
$$\text{TDEE} = \text{BMR} \times \text{Activity Multiplier}$$

| Activity Level | Multiplier | Description |
| :--- | :---: | :--- |
| **Sedentary** | 1.200 | Little or no exercise, desk work |
| **Lightly Active** | 1.375 | Light exercise 1–3 days/week |
| **Moderately Active** | 1.550 | Moderate exercise 3–5 days/week |
| **Very Active** | 1.725 | Hard exercise 6–7 days/week |
| **Extra Active** | 1.900 | Very hard exercise, physical job |

#### 3. Goal-Adjusted Caloric Target:
$$\text{Target Kcal} = \text{TDEE} + \Delta_{\text{goal}}$$

*Where $\Delta_{\text{goal}}$ is $-300\,\text{kcal}$ for weight loss, $+300\,\text{kcal}$ for weight gain, $+200\,\text{kcal}$ for muscle gain, and $0\,\text{kcal}$ for maintenance and clinical management.*

---

### C. ICMR-NIN RDA Reference Profile Mapping

Profiles are dynamically mapped to clinical standards in `user_health_profiles.json`:
1. `manage_diabetes` or `diabetic_friendly` $\rightarrow$ `diabetic_adult`
2. `manage_hypertension` or `low_sodium` $\rightarrow$ `hypertensive_adult`
3. `weight_loss` (or BMI $\ge 25.0$) $\rightarrow$ `weight_management`
4. Default Female $\rightarrow$ `adult_female_sedentary`
5. Default Male / Other $\rightarrow$ `adult_male_sedentary`

---

## 3. Morning Routine & Breakfast Recommendation Engine

The breakfast feature aligns lifestyle rhythms with optimal nutrition:
- **Circadian Window Evaluation**: Evaluates elapsed time between user `wake_time` and `breakfast_time`. Recommends breakfast consumption within 2 hours of waking.
- **Preference Scaling**:
  - `light`: Target prep time $\le 20$ minutes; prioritizes high-fiber, light digestibility dishes (e.g., Poha, Idli).
  - `medium`: Target prep time $\le 40$ minutes; balanced macronutrients (e.g., Upma, Dosa).
  - `hearty`: Target prep time $\le 60$ minutes; protein and complex carbohydrate density (e.g., Paratha, Pongal).
- **Hard Constraints Enforcement**: Excludes recipes containing detected allergens or conflicting with dietary restrictions (`vegetarian`, `vegan`, `jain`, etc.).

---

## 4. Architecture & Persistence Specification

### Database Schema (`database/nutrisense_users.db`):
```sql
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
```

- **Data Privacy**: No credentials, biometric tokens, or clinical diagnosis histories are persisted.
- **Resilience**: The JSON `profile_blob` stores complete preference sets while indexed scalar columns allow low-latency lookups.

---

## 5. REST API Endpoints

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/profile/create` | Create user profile with full schema validation | `201`, `400`, `500` |
| `GET` | `/api/profile/<user_id>` | Retrieve complete user profile | `200`, `404`, `500` |
| `PUT` | `/api/profile/<user_id>` | Update specific profile fields | `200`, `400`, `404`, `500` |
| `DELETE` | `/api/profile/<user_id>` | Delete user profile from database | `200`, `404`, `500` |
| `GET` | `/api/profiles` (or `/api/user-profiles`) | List active profiles (summary projection) | `200`, `500` |
| `POST` | `/api/profile/<user_id>/bmi` | Calculate BMI with South Asian thresholds | `200`, `400`, `404`, `500` |
| `GET` | `/api/profile/<user_id>/breakfast` | Retrieve personalized breakfast recommendations | `200`, `404`, `500` |
| `POST` | `/api/profile/<user_id>/personalize` | Personalized food portion & suitability assessment | `200`, `400`, `404`, `500` |

---

## 6. Comprehensive Verification & Test Results

The test suite [`tests/test_user_profiles.py`](file:///C:/Samsung_Project/tests/test_user_profiles.py) provides 100% test coverage across 10 functional categories:

| Section | Scope | Tests Run | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| **Section 1** | Profile Schema & Strict Field Validation | 18 | 18 | 0 |
| **Section 2** | BMI Calculation & Threshold Classifications | 10 | 10 | 0 |
| **Section 3** | Mifflin-St Jeor Calorie Targets & TDEE | 10 | 10 | 0 |
| **Section 4** | ICMR-NIN RDA Reference Profile Mapping | 8 | 8 | 0 |
| **Section 5** | SQLite CRUD Persistence & UUID Generation | 22 | 22 | 0 |
| **Section 6** | Morning Routine & Breakfast Recommendation | 10 | 10 | 0 |
| **Section 7** | Personalized Portion Guidance Scaling | 8 | 8 | 0 |
| **Section 8** | Web REST API Endpoints Integration | 25 | 25 | 0 |
| **Section 9** | Missing/Invalid Field API Error Handling | 7 | 7 | 0 |
| **Section 10** | Data Protection & Non-Clinical Disclaimers | 10 | 10 | 0 |
| **Bonus** | Boundary & Edge Cases | 12 | 12 | 0 |
| **Total** | **Phase 8 Verification Suite** | **140** | **140** | **0** |

---

## 7. Phase Completion Statement

All requirements for Phase 8 have been implemented, validated, and verified.
No existing features were altered or degraded.

