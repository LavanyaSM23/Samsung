"""
app.py
NutriSense AI - Phase 6 Web Application Server

Flask backend integrating:
- MobileNetV2 43-Class Food Recognition Inference
- Master Nutrition Knowledge Base Lookup
- Interactive Ambiguous Subtype Disambiguation
- Composite Recipe Ingredient Explorer
- ICMR-NIN 2020 RDA Clinical Health Suitability Scoring
- Dynamic Portion Scaling & Allergen Alerts
- Transparent Data Quality Flags & Medical Disclaimer
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import io
import json
from pathlib import Path
import sys
from typing import Dict, Any

from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np
import pandas as pd
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.inference_pipeline import NutriSenseInferenceEngine, ALLERGEN_MAP
from src.nutrient_scoring_engine import compute_suitability_score, ICMR_RDA_PROFILES
from src.label_scanner import scan_label
from src.nutrition_analyzer import (
    get_analyzer,
    calculate_nutrients,
    convert_portion_to_grams,
    search_foods,
    lookup_food,
    get_available_servings,
    PHASE7_DISCLAIMER,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max image upload

# Initialize Inference Engine
print("=" * 65)
print("INITIALIZING NUTRISENSE AI INFERENCE ENGINE...")
print("=" * 65)
engine = NutriSenseInferenceEngine()
print("ENGINE READY FOR WEB SERVING!")

SAMPLES_DIR = WORKSPACE_ROOT / "static" / "samples"
MODELS_DIR = WORKSPACE_ROOT / "models"


@app.route("/")
def index():
    """Serves the main single-page web application."""
    return render_template("index.html")


@app.route("/reports/figures/<path:filename>")
def serve_report_figure(filename):
    """Serves generated benchmark charts and figures."""
    figures_dir = WORKSPACE_ROOT / "reports" / "figures"
    return send_from_directory(figures_dir, filename)


@app.route("/api/health", methods=["GET"])
def health():
    """Returns system status, active model info, and hardware context."""
    return jsonify({
        "status": "healthy",
        "service": "NutriSense AI Web Application",
        "model_architecture": "MobileNetV2 (ImageNet-1K pretrained + 43-class fine-tuned)",
        "classes_count": len(engine.class_to_idx),
        "device": str(engine.device),
        "is_trained": engine.is_trained,
        "checkpoint_path": str(engine.checkpoint_path),
        "disclaimer": "NutriSense AI provides evidence-based nutritional estimations based on ICMR-NIN standards. This information is intended for educational and dietary awareness and does not constitute clinical medical advice."
    })


@app.route("/api/profiles", methods=["GET"])
def get_profiles():
    """Returns available ICMR-NIN 2020 RDA user health profiles."""
    profiles_meta = []
    for key, p in ICMR_RDA_PROFILES.items():
        profiles_meta.append({
            "key": key,
            "title": key.replace("_", " ").title(),
            "description": p.get("description", ""),
            "calories_target": p.get("calories_kcal", 2000),
            "protein_target_g": p.get("protein_g", 50),
            "carbs_target_g": p.get("carbs_g", 250),
            "fat_target_g": p.get("fat_g", 25),
            "fibre_target_g": p.get("fibre_g", 30),
            "sodium_max_mg": p.get("sodium_mg_max", 2000),
            "clinical_condition": p.get("clinical_condition", "GENERAL_WELLNESS")
        })
    return jsonify(profiles_meta)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns list of curated test food samples with thumbnails."""
    sample_items = [
        {"class_name": "samosa", "title": "Samosa", "description": "Crispy fried pastry (Ambiguous: Aloo / Keema / Paneer)", "url": "/static/samples/samosa.jpg"},
        {"class_name": "biryani", "title": "Biryani", "description": "Fragrant spiced rice layered dish (Ambiguous)", "url": "/static/samples/biryani.jpg"},
        {"class_name": "palak_paneer", "title": "Palak Paneer", "description": "Cottage cheese in spiced spinach gravy (Dairy)", "url": "/static/samples/palak_paneer.jpg"},
        {"class_name": "jalebi", "title": "Jalebi", "description": "Crispy pretzel-shaped sweet soaked in saffron syrup", "url": "/static/samples/jalebi.jpg"},
        {"class_name": "idli", "title": "Idli", "description": "Steamed fermented rice and lentil cakes (Healthy)", "url": "/static/samples/idli.jpg"},
        {"class_name": "aloo_gobi", "title": "Aloo Gobi", "description": "Potato and cauliflower stir-fry (Composite Recipe)", "url": "/static/samples/aloo_gobi.jpg"},
        {"class_name": "chicken_curry", "title": "Chicken Curry", "description": "Tender chicken simmered in onion-tomato gravy (Non-Veg)", "url": "/static/samples/chicken_curry.jpg"},
        {"class_name": "momos", "title": "Momos", "description": "Steamed Himalayan dumplings (Ambiguous filling)", "url": "/static/samples/momos.jpg"},
        {"class_name": "modak", "title": "Modak", "description": "Steamed sweet dumplings with coconut-jaggery (Nuts)", "url": "/static/samples/modak.jpg"},
        {"class_name": "dhokla", "title": "Dhokla", "description": "Steamed savory fermented gram flour snack", "url": "/static/samples/dhokla.jpg"},
    ]
    return jsonify(sample_items)


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Returns Phase 5 model evaluation metrics and benchmark stats."""
    eval_file = MODELS_DIR / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Evaluation results not found"}), 404


@app.route("/api/food-catalog", methods=["GET"])
def get_food_catalog():
    """Returns complete catalog of 43 foods with nutrition and flags."""
    foods = []
    for c, row in engine.master_kb_map.items():
        foods.append({
            "class_name": c,
            "display_name": row.get("display_name", c),
            "match_type": row.get("match_type", "VALID_DIRECT"),
            "serving_size_g": row.get("serving_size_g"),
            "serving_unit": row.get("serving_unit"),
            "calories_kcal": row.get("calories_kcal"),
            "protein_g": row.get("protein_g"),
            "fat_g": row.get("fat_g"),
            "carbs_g": row.get("carbs_g"),
            "sodium_mg": row.get("sodium_mg"),
            "sodium_flag": row.get("sodium_flag", "ok"),
            "calorie_flag": row.get("calorie_flag", "ok"),
            "allergens": ALLERGEN_MAP.get(c, []),
            "has_subtypes": bool(row.get("has_subtypes", False))
        })
    return jsonify(sorted(foods, key=lambda x: x["display_name"]))


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receives food image via file upload or base64 string, runs vision model,
    and returns top predictions, ambiguous subtypes, scaled nutrients,
    clinical health scores, and allergen alerts.
    """
    try:
        user_profile = request.form.get("user_profile") or request.args.get("user_profile") or "adult_male_sedentary"
        portion_g_raw = request.form.get("portion_g") or request.args.get("portion_g")
        portion_g = float(portion_g_raw) if portion_g_raw else None
        selected_subtype_id = request.form.get("selected_subtype_id") or request.args.get("selected_subtype_id")

        pil_image = None

        # Check multipart file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                pil_image = Image.open(file.stream).convert("RGB")

        # Check JSON payload (e.g. base64 or sample path)
        if pil_image is None and request.is_json:
            json_data = request.get_json()
            if not json_data:
                return jsonify({"error": "Empty JSON payload"}), 400

            user_profile = json_data.get("user_profile", user_profile)
            portion_g = json_data.get("portion_g", portion_g)
            selected_subtype_id = json_data.get("selected_subtype_id", selected_subtype_id)

            # Sample path reference
            if "sample_path" in json_data:
                sample_path = WORKSPACE_ROOT / json_data["sample_path"].lstrip("/")
                if sample_path.exists():
                    pil_image = Image.open(sample_path).convert("RGB")

            # Base64 string
            elif "image_base64" in json_data:
                b64_data = json_data["image_base64"]
                if "," in b64_data:
                    b64_data = b64_data.split(",")[1]
                img_bytes = base64.b64decode(b64_data)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if pil_image is None:
            return jsonify({"error": "No valid image provided via 'file' or 'image_base64'"}), 400

        # Execute full analysis pipeline
        result = engine.analyze_food(
            pil_image,
            top_k=5,
            user_profile=user_profile,
            portion_g=portion_g,
            selected_subtype_id=selected_subtype_id
        )

        return jsonify({
            "success": True,
            "data": result,
            "disclaimer": "Nutritional estimations are for informational and lifestyle awareness. Not clinical medical advice."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/adjust-nutrition", methods=["POST"])
def adjust_nutrition():
    """
    Recalculates scaled nutrition, health suitability score, and alerts
    when user modifies portion size or selects a specific subtype,
    without requiring vision re-inference.
    """
    try:
        data = request.get_json() or {}
        class_name = data.get("class_name")
        if not class_name or class_name not in engine.master_kb_map:
            return jsonify({"error": f"Invalid or missing class_name: {class_name}"}), 400

        user_profile = data.get("user_profile", "adult_male_sedentary")
        portion_g = data.get("portion_g")
        selected_subtype_id = data.get("selected_subtype_id")

        # Create dummy image tensor to leverage engine.analyze_food with exact subtype
        # Or run targeted lookup directly:
        base_kb = engine.master_kb_map[class_name].copy()
        chosen_profile = base_kb.copy()

        # Check subtypes
        subtype_info = None
        requires_confirmation = False

        if class_name in engine.subtypes_registry:
            family_meta = engine.subtypes_registry[class_name]
            subtypes_list = family_meta.get("subtypes", [])
            default_sub = next((s for s in subtypes_list if s.get("is_default")), subtypes_list[0])

            if selected_subtype_id:
                match_sub = next((s for s in subtypes_list if s.get("subtype_id") == selected_subtype_id), None)
                if match_sub:
                    chosen_profile.update(match_sub)
                    chosen_profile["selected_subtype"] = selected_subtype_id
                    requires_confirmation = False
                else:
                    chosen_profile.update(default_sub)
                    requires_confirmation = True
            else:
                chosen_profile.update(default_sub)
                requires_confirmation = True

            subtype_info = {
                "requires_confirmation": requires_confirmation,
                "disambiguation_prompt": family_meta.get("disambiguation_prompt", "Please confirm food subtype:"),
                "default_subtype": default_sub,
                "available_subtypes": [
                    {
                        "subtype_id": s["subtype_id"],
                        "display_name": s["display_name"],
                        "calories_kcal": s.get("calories_kcal"),
                        "protein_g": s.get("protein_g"),
                        "fat_g": s.get("fat_g"),
                        "carbs_g": s.get("carbs_g"),
                        "serving_size_g": s.get("serving_size_g"),
                        "serving_unit": s.get("serving_unit")
                    } for s in subtypes_list
                ]
            }

        # Portion scaling
        default_serving_g = float(chosen_profile.get("serving_size_g", 100.0))
        if pd.isna(default_serving_g) or default_serving_g <= 0:
            default_serving_g = engine.serving_weights_map.get(class_name, 100.0)

        actual_portion_g = float(portion_g) if portion_g is not None and float(portion_g) > 0 else default_serving_g
        scale = actual_portion_g / default_serving_g if default_serving_g > 0 else 1.0

        def _scale_val(val):
            if pd.isna(val) or val is None:
                return None
            return round(float(val) * scale, 2)

        scaled_nutrition = {
            "serving_size_g": actual_portion_g,
            "serving_unit": chosen_profile.get("serving_unit", "standard serving"),
            "portion_scale_factor": round(scale, 3),
            "calories_kcal": _scale_val(chosen_profile.get("calories_kcal")),
            "carbs_g": _scale_val(chosen_profile.get("carbs_g")),
            "protein_g": _scale_val(chosen_profile.get("protein_g")),
            "fat_g": _scale_val(chosen_profile.get("fat_g")),
            "fibre_g": _scale_val(chosen_profile.get("fibre_g")),
            "sodium_mg": _scale_val(chosen_profile.get("sodium_mg")),
            "calcium_mg": _scale_val(chosen_profile.get("calcium_mg")),
            "iron_mg": _scale_val(chosen_profile.get("iron_mg")),
            "vitamin_c_mg": _scale_val(chosen_profile.get("vitamin_c_mg")),
            "folate_ug": _scale_val(chosen_profile.get("folate_ug")),
        }

        # Missing micros
        missing_micros = [m for m in ["calcium_mg", "iron_mg", "vitamin_c_mg", "folate_ug"] if scaled_nutrition[m] is None or pd.isna(chosen_profile.get(m))]

        # Flags & health scoring
        sodium_flag = str(chosen_profile.get("sodium_flag", "ok"))
        calorie_flag = str(chosen_profile.get("calorie_flag", "ok"))

        score_input = {
            "calories_kcal": scaled_nutrition["calories_kcal"] or 0,
            "carbs_g": scaled_nutrition["carbs_g"] or 0,
            "protein_g": scaled_nutrition["protein_g"] or 0,
            "fat_g": scaled_nutrition["fat_g"] or 0,
            "fibre_g": scaled_nutrition["fibre_g"] or 1.0,
            "sodium_mg": scaled_nutrition["sodium_mg"] or 200.0,
            "sodium_flag": sodium_flag,
            "calorie_flag": calorie_flag,
            "display_name": chosen_profile.get("display_name", class_name),
            "class_name": class_name
        }
        health_score, verdict = compute_suitability_score(score_input, profile_key=user_profile)

        # Alerts
        alerts = []
        if sodium_flag == "high_sodium_unverified":
            alerts.append({
                "severity": "WARNING",
                "type": "SODIUM_DATA_LIMITATION",
                "message": f"Source sodium is extremely high ({chosen_profile.get('sodium_mg')} mg/serving) and unverified in source literature. Exercise caution."
            })
        if calorie_flag == "discrepant_atwater_over_25pct":
            alerts.append({
                "severity": "INFO",
                "type": "CALORIE_DISCREPANCY_FLAG",
                "message": "Dish has >25% discrepancy between reported calories and Atwater 4-4-9 sum in source literature."
            })
        if user_profile == "hypertensive_adult" and (scaled_nutrition["sodium_mg"] or 0) > 500.0:
            alerts.append({
                "severity": "HIGH",
                "type": "HYPERTENSION_WARNING",
                "message": "High sodium portion exceeds DASH recommendation for hypertensive individuals."
            })
        if user_profile == "diabetic_adult" and (scaled_nutrition["carbs_g"] or 0) > 35.0:
            alerts.append({
                "severity": "MODERATE",
                "type": "DIABETES_CARB_ALERT",
                "message": "High carbohydrate content may cause glycemic spike. Consider pairing with fiber/salad."
            })

        return jsonify({
            "success": True,
            "data": {
                "class_name": class_name,
                "display_name": chosen_profile.get("display_name", class_name),
                "nutrition": scaled_nutrition,
                "metadata": {
                    "match_type": chosen_profile.get("match_type", "VALID_DIRECT"),
                    "data_source": chosen_profile.get("data_source", "master_food_nutrition_kb.csv"),
                    "sodium_flag": sodium_flag,
                    "calorie_flag": calorie_flag,
                    "missing_micronutrients": missing_micros
                },
                "ambiguous_subtype": subtype_info,
                "health_evaluation": {
                    "user_profile": user_profile,
                    "suitability_score": health_score,
                    "verdict": verdict,
                    "dietary_alerts": alerts,
                    "detected_allergens": ALLERGEN_MAP.get(class_name, [])
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Phase 7: Nutrition Analysis Module Endpoints
# =============================================================================

@app.route("/api/nutrition/search", methods=["GET"])
def search_nutrition_foods():
    """
    Manual food search fallback when image recognition fails or user searches directly.
    Searches across Master Food Nutrition KB and Unified Food Lookup (1,556 items).
    """
    try:
        query = request.args.get("q", "").strip()
        limit = int(request.args.get("limit", 15))
        if not query:
            return jsonify({"success": True, "count": 0, "data": []})

        results = search_foods(query, limit=limit)
        return jsonify({
            "success": True,
            "query": query,
            "count": len(results),
            "data": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nutrition/servings", methods=["GET"])
def get_food_servings():
    """Returns list of defined serving options for a given food name or class."""
    try:
        food_name = request.args.get("food_name", "").strip()
        if not food_name:
            return jsonify({"error": "Missing 'food_name' query parameter"}), 400

        servings = get_available_servings(food_name)
        return jsonify({
            "success": True,
            "food_name": food_name,
            "servings": servings
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nutrition/analyze", methods=["POST"])
def analyze_nutrition_item():
    """
    Phase 7 Core Endpoint: Calculates scaled nutrition using the per-100g standard formula:
      nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g
    Handles portion conversion from grams or defined servings (piece, katori, plate, etc.).
    Preserves missing nutrients as null and reports data source & quality flags.
    """
    try:
        data = request.get_json() or {}
        food_name = data.get("food_name") or data.get("class_name")
        if not food_name:
            return jsonify({"error": "Missing 'food_name' in request body"}), 400

        portion_val = data.get("portion_value")
        if portion_val is None:
            portion_val = data.get("portion_g", 100.0)

        unit = data.get("unit", "g")
        subtype_id = data.get("subtype_id") or data.get("selected_subtype_id")
        user_profile = data.get("user_profile", "adult_male_sedentary")

        # 1. Convert portion to grams
        try:
            portion_grams = convert_portion_to_grams(portion_val, unit=unit, food_name=food_name)
        except ValueError as ve:
            return jsonify({"error": f"Invalid portion: {str(ve)}"}), 400

        # 2. Calculate scaled nutrients using standard per-100g formula
        try:
            analysis = calculate_nutrients(food_name, portion_grams, subtype_id=subtype_id)
        except KeyError as ke:
            return jsonify({"error": str(ke)}), 404

        # 3. ICMR-NIN RDA health evaluation
        nut = analysis["nutrients"]
        score_input = {
            "calories_kcal": nut["calories_kcal"]["value"] or 0,
            "carbs_g": nut["carbs_g"]["value"] or 0,
            "protein_g": nut["protein_g"]["value"] or 0,
            "fat_g": nut["fat_g"]["value"] or 0,
            "fibre_g": nut["fibre_g"]["value"] or 1.0,
            "sodium_mg": nut["sodium_mg"]["value"] or 200.0,
            "sodium_flag": analysis["data_quality_flags"]["sodium_flag"],
            "calorie_flag": analysis["data_quality_flags"]["calorie_flag"],
            "display_name": analysis["display_name"],
            "class_name": analysis["class_name"]
        }
        health_score, verdict = compute_suitability_score(score_input, profile_key=user_profile)

        # Generate dietary alerts
        alerts = []
        if analysis["data_quality_flags"]["sodium_flag"] == "high_sodium_unverified":
            alerts.append({
                "severity": "WARNING",
                "type": "SODIUM_DATA_LIMITATION",
                "message": "Source sodium is extremely high and unverified in laboratory literature. Exercise caution."
            })
        if analysis["data_quality_flags"]["calorie_flag"] == "discrepant_atwater_over_25pct":
            alerts.append({
                "severity": "INFO",
                "type": "CALORIE_DISCREPANCY_FLAG",
                "message": "Dish has >25% discrepancy between reported calories and Atwater 4-4-9 sum in source literature."
            })
        if user_profile == "hypertensive_adult" and (nut["sodium_mg"]["value"] or 0) > 500.0:
            alerts.append({
                "severity": "HIGH",
                "type": "HYPERTENSION_WARNING",
                "message": "Portion sodium exceeds DASH guideline for hypertensive individuals."
            })
        if user_profile == "diabetic_adult" and (nut["carbs_g"]["value"] or 0) > 35.0:
            alerts.append({
                "severity": "MODERATE",
                "type": "DIABETES_CARB_ALERT",
                "message": "High carbohydrate portion may cause glycemic spike. Pair with fiber/salad."
            })

        analysis["health_evaluation"] = {
            "user_profile": user_profile,
            "suitability_score": health_score,
            "verdict": verdict,
            "dietary_alerts": alerts,
            "detected_allergens": ALLERGEN_MAP.get(analysis["class_name"], [])
        }

        # Available defined servings for frontend dropdown
        analysis["available_servings"] = get_available_servings(food_name)

        return jsonify({"success": True, "data": analysis})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/scan-label", methods=["POST"])
def scan_packaged_label():
    """
    Accepts a product packaging photo via multipart file upload or base64 JSON.
    Runs EasyOCR and structured parsing, returning all extracted fields.
    Missing fields are reported as null — never invented or zero-filled.
    """
    try:
        pil_image = None

        if "file" in request.files:
            f = request.files["file"]
            if f and f.filename:
                pil_image = Image.open(f.stream).convert("RGB")

        if pil_image is None and request.is_json:
            json_data = request.get_json() or {}
            if "image_base64" in json_data:
                import base64, io as _io
                b64 = json_data["image_base64"]
                if "," in b64:
                    b64 = b64.split(",")[1]
                pil_image = Image.open(_io.BytesIO(base64.b64decode(b64))).convert("RGB")

        if pil_image is None:
            return jsonify({"success": False, "error": "No image provided (use 'file' multipart or 'image_base64' JSON)."}), 400

        result = scan_label(pil_image)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/confirm-label", methods=["POST"])
def confirm_packaged_label():
    """
    Accepts user-confirmed (possibly corrected) label data and evaluates
    ICMR-NIN 2020 RDA suitability for the active user health profile.
    Tracks which fields were user-corrected vs OCR-extracted.
    """
    try:
        data = request.get_json() or {}
        user_profile = data.get("user_profile", "adult_male_sedentary")
        nutrition_confirmed = data.get("nutrition", {})
        corrections = data.get("user_corrections", {})

        def _parse(field):
            """Parse numeric value from confirmed nutrition dict entry."""
            entry = nutrition_confirmed.get(field)
            if entry is None:
                return None
            if isinstance(entry, dict):
                try:
                    return float(entry.get("value", 0) or 0)
                except (TypeError, ValueError):
                    return None
            try:
                return float(entry)
            except (TypeError, ValueError):
                return None

        # Build scoring input from confirmed data
        cal = _parse("calories")
        protein = _parse("protein")
        carbs = _parse("total_carbohydrates")
        fat = _parse("total_fat")
        fibre = _parse("dietary_fibre")
        sodium = _parse("sodium")
        salt = _parse("salt")

        # Use sodium if present; if only salt declared, note conversion is NOT automatic
        effective_sodium = sodium
        sodium_source = "sodium"
        if effective_sodium is None and salt is not None:
            effective_sodium = None  # Do NOT auto-convert salt to sodium
            sodium_source = "salt_not_converted"

        health_score, verdict = None, None
        alerts = []

        if cal is not None:
            score_input = {
                "calories_kcal": cal or 0,
                "carbs_g": carbs or 0,
                "protein_g": protein or 0,
                "fat_g": fat or 0,
                "fibre_g": fibre or 1.0,
                "sodium_mg": effective_sodium or 200.0,
                "sodium_flag": "ok",
                "calorie_flag": "ok",
                "display_name": data.get("product", {}).get("product_name", "Packaged Product"),
                "class_name": "packaged_product"
            }
            health_score, verdict = compute_suitability_score(score_input, profile_key=user_profile)

            if user_profile == "hypertensive_adult" and effective_sodium and effective_sodium > 500:
                alerts.append({
                    "severity": "HIGH",
                    "type": "HYPERTENSION_WARNING",
                    "message": f"Declared sodium ({effective_sodium:.0f} mg/serving) exceeds DASH threshold for hypertensive individuals."
                })
            if user_profile == "diabetic_adult" and carbs and carbs > 35:
                alerts.append({
                    "severity": "MODERATE",
                    "type": "DIABETES_CARB_ALERT",
                    "message": "High carbohydrate content may cause glycemic spike. Review serving size carefully."
                })
            if salt is not None and sodium is None:
                alerts.append({
                    "severity": "INFO",
                    "type": "SALT_NOT_CONVERTED",
                    "message": (
                        f"Label declares salt ({salt} g/serving). Sodium not auto-converted from salt — "
                        "sodium value remains unavailable. Check product label for declared sodium."
                    )
                })

        return jsonify({
            "success": True,
            "data": {
                "user_profile": user_profile,
                "confirmed_nutrition": nutrition_confirmed,
                "user_corrections": corrections,
                "health_evaluation": {
                    "suitability_score": health_score,
                    "verdict": verdict,
                    "dietary_alerts": alerts,
                    "note": (
                        "Health score is based on user-confirmed nutritional values from packaging. "
                        "Not applicable if nutrition panel was not visible or data is incomplete."
                        if health_score is not None
                        else "Health score not computed — calorie value is unavailable."
                    )
                },
                "sodium_source": sodium_source,
                "disclaimer": (
                    "Nutritional assessment is based on values extracted or confirmed from the product label. "
                    "Always verify against the physical product. Not clinical medical advice."
                )
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Phase 8: User Profile Management Endpoints
# =============================================================================

from src.user_profile_manager import (
    validate_profile_data,
    compute_bmi,
    compute_daily_calorie_target,
    map_to_rda_profile,
    get_breakfast_suggestion,
    personalize_portion_guidance,
    get_food_suitability_for_profile,
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
    list_profiles,
    profile_exists,
)


@app.route("/api/profile/create", methods=["POST"])
def api_create_profile():
    """POST /api/profile/create — create a new user profile."""
    try:
        data = request.get_json(force=True, silent=True) or {}
        if not data.get("display_name") and data.get("name"):
            data["display_name"] = str(data["name"]).strip()
        ok, errors = validate_profile_data(data)
        if not ok:
            return jsonify({"success": False, "errors": errors}), 400
        profile = create_profile(data)
        return jsonify({"success": True, "data": profile}), 201
    except ValueError as exc:
        errs = exc.args[0] if exc.args else [str(exc)]
        return jsonify({"success": False, "errors": errs if isinstance(errs, list) else [errs]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/profile/<user_id>", methods=["GET"])
def api_get_profile(user_id):
    """GET /api/profile/<user_id> — retrieve profile by ID."""
    try:
        profile = get_profile(user_id)
        if profile is None:
            return jsonify({"success": False, "error": "Profile not found."}), 404
        return jsonify({"success": True, "data": profile})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/profile/<user_id>", methods=["PUT"])
def api_update_profile(user_id):
    """PUT /api/profile/<user_id> — update profile fields."""
    try:
        if not profile_exists(user_id):
            return jsonify({"success": False, "error": "Profile not found."}), 404
        updates = request.get_json(force=True, silent=True) or {}
        if not updates.get("display_name") and updates.get("name"):
            updates["display_name"] = str(updates["name"]).strip()
        updated = update_profile(user_id, updates)
        if updated is None:
            return jsonify({"success": False, "error": "Profile not found."}), 404
        return jsonify({"success": True, "data": updated})
    except ValueError as exc:
        errs = exc.args[0] if exc.args else [str(exc)]
        return jsonify({"success": False, "errors": errs if isinstance(errs, list) else [errs]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/profile/<user_id>", methods=["DELETE"])
def api_delete_profile(user_id):
    """DELETE /api/profile/<user_id> — remove a profile."""
    try:
        deleted = delete_profile(user_id)
        if not deleted:
            return jsonify({"success": False, "error": "Profile not found."}), 404
        return jsonify({"success": True, "data": {"deleted": user_id}})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/user-profiles", methods=["GET"])
def api_list_profiles():
    """GET /api/user-profiles — list all user profiles (id + display_name + created_at)."""
    try:
        profiles = list_profiles()
        return jsonify({"success": True, "data": profiles, "count": len(profiles)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/profile/<user_id>/bmi", methods=["POST"])
def api_bmi(user_id):
    """POST /api/profile/<user_id>/bmi — compute BMI for a profile."""
    try:
        profile = get_profile(user_id)
        if profile is None:
            return jsonify({"success": False, "error": "Profile not found."}), 404
        bmi_data = compute_bmi(profile.get("height_cm"), profile.get("weight_kg"))
        return jsonify({"success": True, "data": bmi_data})
    except ValueError as exc:
        return jsonify({"success": False, "errors": [str(exc)]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/profile/<user_id>/breakfast", methods=["GET"])
def api_breakfast(user_id):
    """GET /api/profile/<user_id>/breakfast — morning routine breakfast suggestion."""
    try:
        profile = get_profile(user_id)
        if profile is None:
            return jsonify({"success": False, "error": "Profile not found."}), 404
        suggestions = get_breakfast_suggestion(profile)
        return jsonify({"success": True, "data": suggestions})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/profile/<user_id>/personalize", methods=["POST"])
def api_personalize(user_id):
    """POST /api/profile/<user_id>/personalize — personalize nutrition analysis."""
    try:
        profile = get_profile(user_id)
        if profile is None:
            return jsonify({"success": False, "error": "Profile not found."}), 404
        body = request.get_json(force=True, silent=True) or {}
        food_name = body.get("food_name", "")
        if not food_name:
            return jsonify({"success": False, "errors": ["food_name is required."]}), 400
        portion_guidance = personalize_portion_guidance(food_name, profile)
        suitability = get_food_suitability_for_profile(food_name, profile)
        calorie_target = None
        if all(profile.get(k) for k in ("age", "gender", "height_cm", "weight_kg", "activity_level", "dietary_goal")):
            try:
                calorie_target = compute_daily_calorie_target(
                    profile["age"], profile["gender"],
                    profile["height_cm"], profile["weight_kg"],
                    profile["activity_level"], profile["dietary_goal"],
                )
            except Exception:
                calorie_target = None
        return jsonify({
            "success": True,
            "data": {
                "food_name": food_name,
                "portion_guidance": portion_guidance,
                "food_suitability": suitability,
                "daily_calorie_target": calorie_target,
                "rda_profile_key": map_to_rda_profile(profile),
            }
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# =============================================================================
# Phase 9: Recommendation System Endpoints
# =============================================================================

from src.recommendation_engine import (
    get_recommendation_system,
    SafetyFilterEngine,
    SuitabilityScorer,
    AlternativeFoodsEngine,
    IngredientSubstitutionEngine,
    RECOMMENDATION_DISCLAIMER,
    SAFETY_OVERRIDE_NOTICE,
)

_rec_system = None

def _get_rec_engine():
    global _rec_system
    if _rec_system is None:
        _rec_system = get_recommendation_system()
    return _rec_system


def _resolve_profile(body: dict):
    """Helper to resolve profile from user_id or embedded user_profile dict."""
    user_id = body.get("user_id")
    if user_id:
        p = get_profile(user_id)
        if p is None:
            return None, "Profile not found for given user_id."
        return p, None
    profile = body.get("user_profile")
    if isinstance(profile, dict):
        return profile, None
    return {}, None


@app.route("/api/recommendations/recipes", methods=["POST"])
def api_recommend_recipes():
    """
    POST /api/recommendations/recipes — Safe, personalized, ranked recipe recommendations.
    Body options:
      - user_id: string (optional)
      - user_profile: dict (optional)
      - course: string (optional, e.g. "breakfast", "lunch", "snack", "dinner")
      - top_n: int (optional, default 10)
      - query_hint: string (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        profile, err = _resolve_profile(body)
        if err:
            return jsonify({"success": False, "error": err}), 404

        course = body.get("course")
        top_n = int(body.get("top_n", 10))
        query_hint = body.get("query_hint")

        engine = _get_rec_engine()
        results = engine.recommend_recipes(profile, course=course, top_n=top_n, query_hint=query_hint)
        return jsonify({
            "success": True,
            "count": len(results),
            "data": results,
            "disclaimer": RECOMMENDATION_DISCLAIMER,
            "safety_notice": SAFETY_OVERRIDE_NOTICE,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/recommendations/alternatives", methods=["POST"])
def api_recommend_alternatives():
    """
    POST /api/recommendations/alternatives — Healthier alternatives for a given food.
    Body:
      - food_name: string (required)
      - user_id: string (optional)
      - user_profile: dict (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        food_name = body.get("food_name", "").strip()
        if not food_name:
            return jsonify({"success": False, "errors": ["food_name is required."]}), 400

        profile, err = _resolve_profile(body)
        if err:
            return jsonify({"success": False, "error": err}), 404

        engine = _get_rec_engine()
        alts = engine.alternatives_engine.get_alternatives(food_name, user_profile=profile)
        return jsonify({
            "success": True,
            "food_name": food_name,
            "count": len(alts),
            "data": alts,
            "disclaimer": RECOMMENDATION_DISCLAIMER,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/recommendations/substitutions", methods=["POST"])
def api_recommend_substitutions():
    """
    POST /api/recommendations/substitutions — Culinary substitutions for ingredients or recipes.
    Body can contain either:
      - ingredient: string (optional)
      - reason: string (optional)
      OR
      - recipe: dict (optional)
      - user_profile / user_id: (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        engine = _get_rec_engine()

        ingredient = body.get("ingredient")
        if ingredient:
            reason = body.get("reason")
            subs = engine.substitution_engine.get_substitutions_for_ingredient(ingredient, reason_or_allergy=reason)
            return jsonify({
                "success": True,
                "ingredient": ingredient,
                "data": subs,
                "disclaimer": RECOMMENDATION_DISCLAIMER,
            })

        recipe = body.get("recipe")
        if recipe and isinstance(recipe, dict):
            profile, err = _resolve_profile(body)
            if err:
                return jsonify({"success": False, "error": err}), 404
            swaps = engine.substitution_engine.suggest_recipe_substitutions(recipe, profile)
            return jsonify({
                "success": True,
                "data": swaps,
                "disclaimer": RECOMMENDATION_DISCLAIMER,
            })

        return jsonify({"success": False, "errors": ["Either 'ingredient' or 'recipe' must be provided."]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/recommendations/score-recipe", methods=["POST"])
def api_score_recipe():
    """
    POST /api/recommendations/score-recipe — Transparent 0-100 score for a specific recipe.
    Body:
      - recipe: dict (required)
      - user_id: string (optional)
      - user_profile: dict (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        recipe = body.get("recipe")
        if not recipe or not isinstance(recipe, dict):
            return jsonify({"success": False, "errors": ["'recipe' object is required."]}), 400

        profile, err = _resolve_profile(body)
        if err:
            return jsonify({"success": False, "error": err}), 404

        engine = _get_rec_engine()
        score_res = engine.scorer.score_recipe(recipe, profile)
        return jsonify({
            "success": True,
            "data": score_res,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/recommendations/daily-plan", methods=["POST"])
def api_daily_plan():
    """
    POST /api/recommendations/daily-plan — Generate balanced 1-day meal plan.
    Body:
      - user_id: string (optional)
      - user_profile: dict (optional)
      - target_calories_kcal: float (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        profile, err = _resolve_profile(body)
        if err:
            return jsonify({"success": False, "error": err}), 404

        target_cals = body.get("target_calories_kcal")
        if target_cals is not None:
            try:
                target_cals = float(target_cals)
            except (ValueError, TypeError):
                target_cals = None

        engine = _get_rec_engine()
        plan = engine.generate_daily_meal_plan(profile, target_calories_kcal=target_cals)
        return jsonify({
            "success": True,
            "data": plan,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/recommendations/weekly-plan", methods=["POST"])
def api_weekly_plan():
    """
    POST /api/recommendations/weekly-plan — Generate balanced 7-day meal plan + grocery list.
    Body:
      - user_id: string (optional)
      - user_profile: dict (optional)
      - target_calories_kcal: float (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        profile, err = _resolve_profile(body)
        if err:
            return jsonify({"success": False, "error": err}), 404

        target_cals = body.get("target_calories_kcal")
        if target_cals is not None:
            try:
                target_cals = float(target_cals)
            except (ValueError, TypeError):
                target_cals = None

        engine = _get_rec_engine()
        weekly = engine.generate_weekly_meal_plan(profile, target_calories_kcal=target_cals)
        return jsonify({
            "success": True,
            "data": weekly,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# =============================================================================
# Phase 10: Backend and Database Integration Endpoints
# =============================================================================

from src.database_manager import (
    get_db_connection,
    log_meal,
    get_meal_history,
    delete_meal_log,
    get_daily_nutrition_totals,
    get_period_nutrition_summary,
    save_meal_plan,
    get_latest_meal_plan,
    log_user_feedback,
    get_user_feedback,
    get_backend_orchestrator,
)


@app.route("/api/meals/log", methods=["POST"])
def api_log_meal():
    """
    POST /api/meals/log — Save a meal entry into SQLite.
    Body:
      - user_id: string (required)
      - food_name: string (required)
      - portion_grams: float (required)
      - calories_kcal: float (required)
      - protein_g, carbs_g, fat_g, fibre_g, sodium_mg: float (optional)
      - course: string (optional, default 'lunch')
      - meal_date: YYYY-MM-DD (optional, default today)
      - notes: string (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        user_id = body.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id is required."]}), 400

        logged = log_meal(user_id, body)
        return jsonify({"success": True, "data": logged}), 201
    except ValueError as ve:
        return jsonify({"success": False, "errors": [str(ve)]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/meals/history", methods=["GET"])
def api_meal_history():
    """
    GET /api/meals/history — Retrieve filtered meal history.
    Query parameters:
      - user_id: string (required)
      - start_date, end_date: YYYY-MM-DD (optional)
      - course: string (optional)
      - limit: int (optional, default 50)
    """
    try:
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id parameter is required."]}), 400

        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        course = request.args.get("course")
        limit = int(request.args.get("limit", 50))

        history = get_meal_history(user_id, start_date=start_date, end_date=end_date, course=course, limit=limit)
        return jsonify({"success": True, "count": len(history), "data": history})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/meals/<log_id>", methods=["DELETE"])
def api_delete_meal(log_id):
    """
    DELETE /api/meals/<log_id> — Delete a specific logged meal.
    Query/Body parameter: user_id (required).
    """
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            body = request.get_json(force=True, silent=True) or {}
            user_id = body.get("user_id")

        if not user_id:
            return jsonify({"success": False, "errors": ["user_id parameter is required to delete meal log."]}), 400

        deleted = delete_meal_log(log_id, user_id)
        if not deleted:
            return jsonify({"success": False, "error": "Meal log not found or user unauthorized."}), 404

        return jsonify({"success": True, "deleted": log_id})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/meals/daily-totals", methods=["GET"])
def api_daily_totals():
    """
    GET /api/meals/daily-totals — Aggregated nutrient totals & target comparisons for a day.
    Query parameters:
      - user_id: string (required)
      - date: YYYY-MM-DD (optional, default today)
    """
    try:
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id parameter is required."]}), 400

        target_date = request.args.get("date")
        totals = get_daily_nutrition_totals(user_id, target_date=target_date)
        return jsonify({"success": True, "data": totals})
    except ValueError as ve:
        return jsonify({"success": False, "errors": [str(ve)]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/meals/summary", methods=["GET"])
def api_nutrition_summary():
    """
    GET /api/meals/summary — Nutrition consumption averages and daily breakdown over N days.
    Query parameters:
      - user_id: string (required)
      - days: int (optional, default 7)
    """
    try:
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id parameter is required."]}), 400

        days = int(request.args.get("days", 7))
        summary = get_period_nutrition_summary(user_id, days=days)
        return jsonify({"success": True, "data": summary})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/meal-plans/save", methods=["POST"])
def api_save_meal_plan():
    """
    POST /api/meal-plans/save — Save a generated meal plan.
    Body:
      - user_id: string (required)
      - plan_data: dict (required)
      - plan_type: string ('daily' or 'weekly', default 'daily')
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        user_id = body.get("user_id", "").strip()
        plan_data = body.get("plan_data")
        if not user_id or not plan_data:
            return jsonify({"success": False, "errors": ["user_id and plan_data are required."]}), 400

        plan_type = str(body.get("plan_type", "daily"))
        saved = save_meal_plan(user_id, plan_data, plan_type=plan_type)
        return jsonify({"success": True, "data": saved}), 201
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/meal-plans/latest", methods=["GET"])
def api_latest_meal_plan():
    """
    GET /api/meal-plans/latest — Fetch user's most recent meal plan.
    Query parameters:
      - user_id: string (required)
      - plan_type: string ('daily' or 'weekly', optional)
    """
    try:
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id parameter is required."]}), 400

        plan_type = request.args.get("plan_type")
        plan = get_latest_meal_plan(user_id, plan_type=plan_type)
        if not plan:
            return jsonify({"success": False, "error": "No meal plan found for given user."}), 404

        return jsonify({"success": True, "data": plan})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/feedback/log", methods=["POST"])
def api_log_feedback():
    """
    POST /api/feedback/log — Submit user rating and qualitative feedback.
    Body:
      - user_id: string (required)
      - rating: int 1-5 (optional)
      - item_id: string (optional)
      - item_type: string ('recommendation', 'meal_plan', 'prediction', 'general')
      - feedback_type: string (optional)
      - comments: string (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        user_id = body.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id is required."]}), 400

        logged_fb = log_user_feedback(user_id, body)
        return jsonify({"success": True, "data": logged_fb}), 201
    except ValueError as ve:
        return jsonify({"success": False, "errors": [str(ve)]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/feedback", methods=["GET"])
def api_get_feedback():
    """
    GET /api/feedback — Retrieve user feedback history.
    Query parameters:
      - user_id: string (required)
      - limit: int (optional, default 20)
    """
    try:
        user_id = request.args.get("user_id", "").strip()
        if not user_id:
            return jsonify({"success": False, "errors": ["user_id parameter is required."]}), 400

        limit = int(request.args.get("limit", 20))
        feedbacks = get_user_feedback(user_id, limit=limit)
        return jsonify({"success": True, "count": len(feedbacks), "data": feedbacks})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/flow/process", methods=["POST"])
def api_integrated_flow():
    """
    POST /api/flow/process — Integrated End-to-End Orchestrator.
    Connects:
      1. Food recognition & database lookup
      2. Portion scaling & nutrition calculations
      3. User profile allergy & restriction safety verification
      4. Transparent 0-100 suitability scoring
      5. Healthier alternative foods suggestions
      6. Meal logging & updated daily totals (if save_to_log=True)
    Body:
      - food_name: string (required)
      - portion_value: float (required)
      - unit: string (optional, default 'g')
      - user_id: string (optional)
      - course: string (optional, default 'lunch')
      - save_to_log: bool (optional, default False)
      - notes: string (optional)
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        food_name = body.get("food_name", "").strip()
        if not food_name:
            return jsonify({"success": False, "errors": ["food_name is required."]}), 400

        portion_val = body.get("portion_value", 100.0)
        unit = str(body.get("unit", "g"))
        user_id = body.get("user_id")
        course = str(body.get("course", "lunch"))
        save_to_log = bool(body.get("save_to_log", False))
        notes = body.get("notes")

        orchestrator = get_backend_orchestrator()
        flow_result = orchestrator.process_food_recognition_and_logging(
            food_name_or_prediction=food_name,
            portion_value=portion_val,
            unit=unit,
            user_id=user_id,
            course=course,
            save_to_log=save_to_log,
            notes=notes
        )
        return jsonify(flow_result)
    except KeyError as ke:
        return jsonify({"success": False, "error": str(ke)}), 404
    except ValueError as ve:
        return jsonify({"success": False, "errors": [str(ve)]}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
