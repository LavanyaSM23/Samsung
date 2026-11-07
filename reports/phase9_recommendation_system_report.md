# Phase 9: Recommendation System — Comprehensive Report

## 1. Executive Summary & Architecture Overview

Phase 9 establishes the **Intelligent Recommendation System** for NutriSense AI, synthesizing machine learning content-based filtering, algorithmic rule-based nutritional matching, and culturally grounded Indian culinary knowledge. The module delivers personalized recipe recommendations, healthy alternative foods, dynamic ingredient substitutions, and multi-horizon meal planning (daily and weekly) with a transparent 0 to 100 suitability score.

```
+----------------------------------------------------------------------------------------------------+
|                                    NUTRISENSE RECOMMENDATION PIPELINE                              |
+----------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
                     +------------------------------------------------------------+
                     | 1. Mandatory SafetyFilterEngine (PRE-RANKING EXECUTION)     |
                     |    - Allergen Scan: dairy, peanuts, nuts, gluten, eggs,    |
                     |      shellfish, soy, sesame                                |
                     |    - Dietary Restrictions: vegetarian, vegan, eggetarian,  |
                     |      jain, gluten_free, diabetic_friendly                  |
                     |    - Flags incomplete ingredient / allergen data           |
                     |    * ZERO-TOLERANCE: Unsafe recipes are strictly excluded  |
                     +------------------------------------------------------------+
                                                  |
                                                  v (Only Certified Safe Recipes)
                     +------------------------------------------------------------+
                     | 2. Content-Based TF-IDF Recommender                        |
                     |    - 5,381 Indian Recipe Feature Vectors                   |
                     |    - Cosine similarity matching user taste preferences     |
                     +------------------------------------------------------------+
                                                  |
                                                  v
                     +------------------------------------------------------------+
                     | 3. Transparent 0-100 SuitabilityScorer                     |
                     |    - Nutrient & Calorie Alignment (35%)                    |
                     |    - Cuisine & Taste Preference Alignment (25%)            |
                     |    - Cooking Time & Feasibility (15%)                      |
                     |    - Budget & Complexity Alignment (15%)                   |
                     |    - Goal Synergy Bonus (10%)                              |
                     |    * Generates human-understandable natural explanations   |
                     +------------------------------------------------------------+
                                                  |
                     +----------------------------+-------------------------------+
                     |                                                            |
                     v                                                            v
+-----------------------------------------+                  +-----------------------------------------+
| 4. Alternative Foods & Substitutions    |                  | 5. Daily & Weekly Meal Planners         |
|    - Calorie/fat reduction swaps        |                  |    - 4 meals: 25% / 35% / 10% / 30% cals|
|    - Practical Indian pantry swaps      |                  |    - 7-day schedule with variety control|
|      (dairy, gluten, nuts, sugar, salt) |                  |    - Aggregated grocery shopping list   |
+-----------------------------------------+                  +-----------------------------------------+
```

---

## 2. Mandatory Safety Architecture & Pre-Ranking Rules

### A. The Invariant Safety Guarantee
> [!CAUTION]
> **Mandatory Safety Rule**: A high suitability score can **NEVER** override safety constraints. Allergen conflicts and dietary restriction violations are evaluated **prior to candidate ranking**. Any dish that conflicts with declared user safety parameters is either completely excised from the recommendation pool or assigned a strict score of `0.0` with verdict `SAFETY_CONFLICT_EXCLUDED`.

### B. Allergen Detection Matrix
The `SafetyFilterEngine` cross-references dish titles, ingredient lines, and tokenized constituents against 8 core allergen lexicons incorporating both Western culinary terms and authentic Indian culinary vocabulary:

| Allergen | Western / Botanical Identifiers | Indian Culinary Terminology Checked |
| :--- | :--- | :--- |
| **Dairy** | Milk, cheese, butter, cream, yogurt, whey | Paneer, dahi, ghee, malai, khoa, khoya, mawa, chaas, buttermilk |
| **Peanuts** | Peanut, groundnut, peanut oil, peanut butter | Mungfali, moongphali |
| **Tree Nuts** | Almond, cashew, walnut, pistachio, hazelnut, pecan | Badam, kaju, akhrot, pista, chilgoza |
| **Gluten** | Wheat, semolina, all-purpose flour, barley, rye | Maida, atta, sooji, suji, rava, naan, paratha, poori, bhatura |
| **Eggs** | Egg, egg white, egg yolk, albumin | Anda, egg bhurji |
| **Shellfish** | Prawn, shrimp, crab, lobster, mussel, clam | Jhinga, fresh prawns |
| **Soy** | Soybean, tofu, soy milk, soy sauce, edamame | Soya chunks, soyabean |
| **Sesame** | Sesame seeds, tahini, sesame oil | Til, gingelly |

### C. Incomplete Ingredient Data Handling
If a recipe record lacks raw ingredient descriptions or tokenized elements (`ingredients_raw` or `ingredient_tokens` null or empty), the engine marks `has_incomplete_ingredients = True`. For users with active medical allergies or strict lifestyle restrictions (such as veganism or celiac/gluten-free), unverified recipes are flagged with warning `UNVERIFIED_INGREDIENTS` and excluded from safety-certified candidate pools.

---

## 3. Mathematical Formulation of the 0 to 100 Suitability Score

The composite Suitability Score $S_{\text{total}} \in [0, 100]$ is computed as a weighted additive function of 5 independent dimensions:

$$S_{\text{total}} = S_{\text{nutrient}} + S_{\text{preference}} + S_{\text{time}} + S_{\text{budget}} + S_{\text{synergy}}$$

Subject to the safety boundary condition:
$$\text{If } \text{SafetyConflict}(R, U) = \text{True} \implies S_{\text{total}} = 0.0$$

### Weighting Matrix and Component Formulations:

| Dimension | Max Points | Evaluation Function | Clinical & Lifestyle Purpose |
| :--- | :---: | :--- | :--- |
| **1. Nutrient & Calorie Alignment** ($S_{\text{nutrient}}$) | **35.0** | $\text{Baseline}(25.0) + \Delta_{\text{protein}} + \Delta_{\text{glycemic}} + \Delta_{\text{fiber}}$ | Caloric calibration, macronutrient density, glycemic impact |
| **2. Cuisine & Taste Preference** ($S_{\text{preference}}$) | **25.0** | $\text{Baseline}(10.0) + \Delta_{\text{cuisine}}(12.0) + \min(8.0, 10 \cdot \text{CosineSim}) - \text{Penalty}_{\text{disliked}}(15.0)$ | Adherence to regional Indian taste, TF-IDF similarity, dislike avoidance |
| **3. Cooking Time & Feasibility** ($S_{\text{time}}$) | **15.0** | $15.0 \cdot \max\left(0, 1 - \frac{\max(0, T_{\text{recipe}} - T_{\text{max}})}{T_{\text{max}}}\right)$ | Time constraint feasibility vs user available preparation window |
| **4. Budget & Complexity** ($S_{\text{budget}}$) | **15.0** | $\text{MatchScore}(\text{Budget}_{\text{user}}, \text{Complexity}_{\text{recipe}})$ | Aligning ingredient cost and kitchen complexity (EASY, MODERATE, ADVANCED) |
| **5. Goal Synergy Bonus** ($S_{\text{synergy}}$) | **10.0** | Specific diet tags matching active health target | Reward for targeted clinical diets (`diabetic_friendly`, `low_sodium`, `high_protein`) |

### Categorical Verdict Classification:
- **$\ge 80.0$**: `HIGHLY_RECOMMENDED`
- **$65.0 - 79.9$**: `RECOMMENDED`
- **$50.0 - 64.9$**: `MODERATE_FIT`
- **$< 50.0$**: `LOW_FIT`
- **Safety Conflict**: `SAFETY_CONFLICT_EXCLUDED` ($0.0$)

---

## 4. Content-Based Filtering via TF-IDF & Cosine Similarity

The `ContentBasedRecommender` vectorizes all 5,381 recipes across:
$$\vec{d}_i = \text{TF-IDF}\left(\text{RecipeName}_i \circ \text{Cuisine}_i \circ \text{Course}_i \circ \text{Diet}_i \circ \text{Tokens}_i\right)$$

- **Vocabulary**: 3,000 unigrams and bigrams with English stopword exclusion.
- **Query Projection**: User search cues or favorite dishes $\vec{q}$ are transformed into the identical subspace.
- **Cosine Metric**:
  $$\text{CosineSim}(\vec{q}, \vec{d}_i) = \frac{\vec{q} \cdot \vec{d}_i}{\|\vec{q}\|_2 \|\vec{d}_i\|_2}$$

---

## 5. Alternative Foods & Ingredient Substitution Engines

### A. Healthier Alternative Foods Engine
Provides goal-aligned, calorie-saving, and lower-fat alternatives for common Indian dietary temptations:
- **Samosa** (Deep-fried, refined flour, 260 kcal) $\rightarrow$ **Baked Vegetable Samosa** (135 kcal, -60% fat), **Khamand Dhokla** (Steamed, chickpea protein, 150 kcal), or **Moong Dal Chilla** (Savory lentil crepe, 140 kcal).
- **Butter Naan** (Refined flour & saturated ghee, 290 kcal) $\rightarrow$ **Tandoori Whole Wheat Roti** (110 kcal) or **Multigrain Phulka** (Low-GI ragi/oats blend, 95 kcal).
- **Jalebi** (Deep-fried sugar syrup, 390 kcal) $\rightarrow$ **Steamed Modak** (140 kcal, fresh coconut & unrefined jaggery) or **Date and Nut Laddu** (120 kcal, unrefined fruit sugars).
- **Chole Bhature** (Deep-fried maida bhatura, 645 kcal) $\rightarrow$ **Chole with Whole Wheat Kulcha** (380 kcal) or **Sprouted Chickpea Salad** (210 kcal).
- **Chicken / Mutton Biryani** (Polished rice, high oil, 480 kcal) $\rightarrow$ **Brown Rice Vegetable Pulao** (220 kcal, high fiber) or **Millet Biryani** (240 kcal, low glycemic impact).

### B. Culturally Grounded Ingredient Substitution Engine
Offers practical 1:1 culinary swaps for Indian kitchens:

| Target Ingredient | Culinary Swap Options | Ratio | Primary Health Benefit |
| :--- | :--- | :---: | :--- |
| **Dairy Milk** | Unsweetened Almond Milk, Soy Milk, Oat Milk | 1:1 | 100% Lactose-free, dairy-free |
| **Paneer** | Firm Tofu, Sprouted Moong Beans | 1:1 | Zero dairy, high plant isoflavones |
| **Ghee / Butter** | Cold-pressed Mustard Oil, Extra Virgin Olive Oil | 3/4 measure | Replaces saturated fat with MUFA & PUFA |
| **Maida (Refined Flour)** | Whole Wheat Atta, Ragi (Finger Millet) Flour, Besan | 1:1 | High dietary fiber, rich in iron and calcium |
| **Refined Sugar** | Stevia extract, Unrefined Jaggery (Gud), Date puree | Pack equivalent | Zero glycemic spike or whole-fruit minerals |
| **Table Salt** | Lemon juice + Amchur powder + Roasted cumin | To taste | Reduces sodium while activating salivary acid receptors |
| **Cashew Gravy Paste** | Melon seed (Magaz) paste, Poppy seed (Khus Khus) paste | 1:1 | 100% Tree-nut allergen free, identical creamy texture |
| **Whole Egg** | Flaxseed slurry (1 tbsp ground flax + 3 tbsp water), Aquafaba | 1 egg equivalent | Vegan binding loaded with Omega-3 ALA |

---

## 6. Daily and Weekly Meal Planning Engines

### A. Calibrated Daily Meal Planning
Calibrates 4 daily eating occasions against the user's Total Daily Energy Expenditure (TDEE):
- **Breakfast**: 25% of daily target kcal (e.g., 475 kcal for 1,900 kcal budget).
- **Lunch**: 35% of daily target kcal (e.g., 665 kcal).
- **Snack**: 10% of daily target kcal (e.g., 190 kcal).
- **Dinner**: 30% of daily target kcal (e.g., 570 kcal).

Every meal candidate is independently safety-filtered, scores $\ge 65/100$, and satisfies the user's available cooking time and kitchen budget.

### B. 7-Day Weekly Meal Planning & Shopping List
- **7-Day Dynamic Rotation**: Plans Monday through Sunday with automated consecutive-day repetition prevention (no dish repeated on back-to-back days).
- **Aggregated Grocery Shopping List**: Parses and tallies recipe ingredient tokens across the 28 planned meals, ranking the top 25 pantry items required for the week.

---

## 7. Web REST API Endpoints Specification

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/recommendations/recipes` | Retrieve safe, ranked recipes matching profile | `200`, `400`, `404`, `500` |
| `POST` | `/api/recommendations/alternatives` | Retrieve healthier alternatives for a given dish | `200`, `400`, `404`, `500` |
| `POST` | `/api/recommendations/substitutions` | Retrieve ingredient or recipe-level culinary swaps | `200`, `400`, `404`, `500` |
| `POST` | `/api/recommendations/score-recipe` | Transparent 0-100 score with component breakdown | `200`, `400`, `404`, `500` |
| `POST` | `/api/recommendations/daily-plan` | Generate calibrated 4-meal daily plan | `200`, `400`, `404`, `500` |
| `POST` | `/api/recommendations/weekly-plan` | Generate 7-day schedule with grocery shopping list | `200`, `400`, `404`, `500` |

---

## 8. Verification & Test Suite Results

The dedicated test suite [`tests/test_recommendation_system.py`](file:///C:/Samsung_Project/tests/test_recommendation_system.py) validated all functional specifications:

| Section | Scope | Tests Run | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| **Section 1** | Allergen Detection & Strict Exclusion | 10 | 10 | 0 |
| **Section 2** | Dietary Restrictions Filtering (Veg, Vegan, Jain, GF) | 6 | 6 | 0 |
| **Section 3** | Incomplete Data Flagging & Safety Override Guarantee | 7 | 7 | 0 |
| **Section 4** | Transparent 0–100 Suitability Scorer & Breakdown | 13 | 13 | 0 |
| **Section 5** | Content-Based Filtering (TF-IDF & Cosine Similarity) | 7 | 7 | 0 |
| **Section 6** | Alternative Foods Engine | 9 | 9 | 0 |
| **Section 7** | Ingredient Substitution Engine | 10 | 10 | 0 |
| **Section 8** | Daily Meal Planning & Caloric Calibration | 13 | 13 | 0 |
| **Section 9** | Weekly Meal Planning, Variety & Grocery List | 9 | 9 | 0 |
| **Section 10** | Web REST API Endpoints Integration | 18 | 18 | 0 |
| **Section 11** | Edge Cases, Missing Profiles & Integrity Bounds | 4 | 4 | 0 |
| **Total** | **Phase 9 Verification Suite** | **106** | **106** | **0** |

---

## 9. Non-Clinical Safety Guardrails & Disclaimers

All API payloads and recommendation objects strictly convey the educational non-clinical notice:
> *"Recommendations and suitability scores (0–100) are algorithmic heuristics based on user-declared preferences, lifestyle parameters, and published dietary references (ICMR-NIN 2020). They are provided for educational and lifestyle guidance only, not as medically validated prescriptions or clinical treatment plans. Individuals with severe medical conditions, food anaphylaxis, or specific metabolic requirements should consult a qualified physician or registered dietitian before modifying their diet."*

