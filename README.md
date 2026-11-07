# NutriSense AI — Evidence-Based Indian Nutrition & Meal Intelligence

NutriSense AI is an end-to-end, privacy-preserving visual nutrition intelligence and personalized meal planning platform tailored specifically for Indian regional diets. The system integrates fine-tuned computer vision (MobileNetV2), laboratory-standard chemical composition data (ICMR-NIN 2020 & IFCT 2017), pre-ranking allergen/dietary safety filtering, 0–100 multi-factor suitability scoring, and persistent SQLite tracking into an interactive Streamlit application.

---

## 🏗️ System Architecture

```
                                  [ User Input ]
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
[ Food Image Upload / Sample ]                             [ Manual Food Search ]
           │                                                         │
           ▼                                                         ▼
[ MobileNetV2 Classifier (43 Classes) ]              [ Food Catalog Resolution (1,556 items) ]
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        ▼
                   [ Interactive User Confirmation & Disambiguation ]
                                        │
                                        ▼
                  [ Serving Unit & Portion Scaling (g / units) ]
                                        │
                                        ▼
                  [ Laboratory Nutrient Engine (ICMR-NIN / IFCT) ]
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
     [ Active User Profile ]                      [ 5,381 Curated Recipes ]
                   │                                         │
                   └────────────────────┬────────────────────┘
                                        ▼
                  [ SafetyFilterEngine: Pre-Ranking Exclusions ]
                  (Dairy, Peanuts, Nuts, Gluten, Non-Veg, etc.)
                                        │
                                        ▼
                  [ 0–100 SuitabilityScorer & Healthier Alternatives ]
                                        │
                   ┌────────────────────┴────────────────────┐
                   ▼                                         ▼
     [ 1-Day & 7-Day Meal Plans ]                [ SQLite Meal Logger (meal_logs) ]
                   │                                         │
                   ▼                                         ▼
    [ Weekly Grocery Shopping List ]            [ Dashboard: Calories, Macros, Trends ]
```

---

## 📂 Project Structure

```
Samsung_Project/
├── app.py                      # Flask REST API backend endpoints
├── streamlit_app.py            # Primary Streamlit interactive web application
├── run_all_tests.py            # Master test regression suite runner
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── data/
│   ├── raw/                    # Raw vision datasets (food_lens, enhanced-indian-food, etc.)
│   └── processed/
│       ├── feature_store/      # Master nutrition KB & curated recipe feature store
│       ├── images/             # Image manifests and enhanced image cache
│       ├── nutrition/          # Cleaned nutritional reference tables
│       └── vision_splits/      # Train, validation, and test split manifests
├── database/
│   └── nutrisense_users.db     # SQLite relational persistence store
├── models/
│   ├── mobilenetv2_food43_best.pth  # Fine-tuned PyTorch vision model checkpoint
│   ├── class_to_idx.json       # 43-class label index mapping
│   ├── class_weights.json      # Effective inverse-frequency class weights
│   └── evaluation_results.json # Held-out test split evaluation metrics
├── reports/                    # Phase documentation and figures
├── src/                        # Core Python business logic & pipelines
│   ├── database_manager.py     # SQLite management, seeding, and flow orchestrator
│   ├── inference_pipeline.py   # Vision model inference & preprocessing
│   ├── nutrition_analyzer.py   # Nutrient calculation & serving conversion
│   ├── recommendation_engine.py# Safety filters, suitability scoring, & meal planning
│   └── user_profile_manager.py # User profiling, BMI, & Mifflin-St Jeor TDEE
├── static/
│   └── samples/                # Sample dish images for testing and preview
└── tests/                      # Comprehensive test suites for all phases
```

---

## 🚀 Setup & Local Execution (Windows)

### 1. Prerequisites
- **Python**: 3.10, 3.11, 3.12, or 3.13 installed.
- **Git** (optional for cloning).

### 2. Environment Setup
Open PowerShell or Command Prompt in the project root:

```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

*Note on Windows with Intel OpenMP*: If you encounter OpenMP runtime library warnings, set:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### 3. Database Initialization
Reference data (foods, nutrients, curated recipes, allergen classes) will automatically seed into `database/nutrisense_users.db` upon initial launch. To seed explicitly:
```powershell
python -c "from src.database_manager import get_db_connection, seed_reference_data_if_empty; conn = get_db_connection(); seed_reference_data_if_empty(conn); print('Database initialized successfully.')"
```

### 4. Running the Application
Launch the Streamlit frontend:
```powershell
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

To run the Flask REST API instead:
```powershell
python app.py
```

---

## 📊 Datasets & Provenance

| Dataset | Records / Images | Source / Reference | License / Terms |
| :--- | :--- | :--- | :--- |
| **Enhanced Indian Food** | 15,404 images across 43 classes | Primary multi-class vision dataset | Research & educational use |
| **FoodLens Dataset** | 2,523 images across 20 classes | Secondary benchmark dataset | Academic / Non-commercial |
| **ICMR-NIN RDA (2020)** | Nutrient reference intake standards | Indian Council of Medical Research | Government of India Open Access |
| **IFCT (2017)** | 528 key foods with complete profiles | Indian Food Composition Tables (NIN) | National Institute of Nutrition |
| **Curated Recipes** | 5,381 authentic regional recipes | Indian cuisine feature store | Educational / Lifestyle guidance |

---

## 🔬 Core Methodology & Algorithms

1. **Computer Vision & Classification**:
   - Model architecture: Fine-tuned `MobileNetV2` featuring customized classification head with dropout ($p=0.3$) and batch normalization.
   - 43 authentic Indian classes (e.g. *biryani*, *dhokla*, *masala dosa*, *palak paneer*, *samosa*, *white rice*).
   - Loss function: Balanced focal cross-entropy addressing heavy class imbalance.

2. **Laboratory-Grade Nutrient Scaling**:
   - Standardized per-100g chemical compositions.
   - Missing laboratory micronutrients remain `None` rather than fabricated with zeros to maintain scientific integrity.
   - Portion scaling across standard units (grams, piece, katori, bowl, glass).

3. **Pre-Ranking Safety Filtering**:
   - Non-negotiable pre-filtering of declared allergens (dairy, peanuts, nuts, gluten, eggs, shellfish, soy, sesame) and dietary restrictions (vegetarian, vegan, jain, etc.).
   - High suitability scores **never** override safety constraints.

4. **Transparent 0–100 Suitability Scoring**:
   - Explicit factor weights:
     - Nutrient & Calorie Target Alignment: **35%**
     - Cuisine & Taste Preference: **25%**
     - Cooking Time Feasibility: **15%**
     - Kitchen Budget & Complexity: **15%**
     - Goal Synergy Bonus: **10%**

5. **Dynamic Meal Planning & Variety Control**:
   - Calibrated 4-meal daily schedules (Breakfast 25%, Lunch 35%, Snack 10%, Dinner 30%).
   - 7-Day meal plans with zero consecutive-day dish repetitions.
   - Aggregated weekly grocery shopping list generation.

---

## 📈 Model Evaluation (Held-Out Test Set)

Evaluated on the held-out test split (`data/processed/vision_splits/enhanced_test_split.csv`):
- **Top-1 Accuracy**: **74.00%**
- **Top-5 Accuracy**: **96.00%**
- **Low-Confidence Trigger**: Ambiguous/uniform inputs correctly trigger low probability ($< 0.35$), providing full candidate ranking and manual food search fallback.

---

## 🧪 Running the Test Suites

To execute the complete regression test suite:

```powershell
# Phase 11 Frontend Streamlit tests (41 checks)
python tests/test_frontend_streamlit.py

# Phase 12 System Integration tests (25 checks)
python tests/test_system_integration.py

# Phase 13 End-to-End Validation & Evaluation (46 checks)
python tests/test_phase13_comprehensive_validation.py

# Master regression runner
python run_all_tests.py
```

---

## ⚠️ Known Limitations & Clinical Guardrails

1. **Visual Mixed-Dish Disambiguation**: Single-dish classification performs with high accuracy, but complex mixed thalis require user confirmation of individual components.
2. **Volumetric vs Weight Estimation**: Portion estimates use standardized average piece/serving weights (e.g. 1 samosa $\approx$ 65g). For medical-grade tracking, scale weight entry in grams is recommended.
3. **Non-Clinical Notice**: NutriSense AI is an algorithmic lifestyle support tool. It does not formulate diagnostic claims, prescribe medical treatments, or substitute professional consultation with a licensed clinical dietitian or physician.
