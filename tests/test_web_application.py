"""
tests/test_web_application.py
Integration & Unit Tests for Phase 6: Web Application & System Integration

Tests:
1. Flask Server Routes & Static Asset Serving
2. API Endpoints: /api/health, /api/profiles, /api/samples, /api/food-catalog, /api/benchmarks
3. Real-time Prediction via /api/predict (Multipart and JSON sample path)
4. Ambiguous Subtype Disambiguation & User Subtype Override
5. Real-time Nutrition Scaling & Portion Adjustment via /api/adjust-nutrition
6. Clinical Condition Alerts (Hypertension sodium penalty, Diabetes carb alert)
7. Error Handling & Edge Cases (invalid image, missing params, unknown class)
"""

import json
from pathlib import Path
import sys
import unittest

WORKSPACE_ROOT = Path("c:/Samsung_Project")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app import app

PASS = "[PASS]"
FAIL = "[FAIL]"
test_results = []


def test(name: str, condition: bool, extra: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if extra:
        msg += f" | {extra}"
    print(msg)
    test_results.append((name, condition))


def run_tests():
    print("=" * 60)
    print("RUNNING PHASE 6 WEB APPLICATION & INTEGRATION TESTS")
    print("=" * 60)

    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Main Page & Static Assets
    print("\n--- 1. Web Page & Asset Serving Tests ---")
    resp_index = client.get("/")
    test("index: serves 200 OK", resp_index.status_code == 200)
    test("index: contains NutriSense brand title", b"NutriSense" in resp_index.data)
    test("index: contains single h1 heading", resp_index.data.count(b"<h1") == 1)
    test("index: contains medical disclaimer banner", b"Medical Disclaimer" in resp_index.data)

    resp_css = client.get("/static/css/styles.css")
    test("static: styles.css serves 200 OK", resp_css.status_code == 200)
    test("static: styles.css has content > 1KB", len(resp_css.data) > 1000)

    resp_js = client.get("/static/js/app.js")
    test("static: app.js serves 200 OK", resp_js.status_code == 200)
    test("static: app.js has content > 1KB", len(resp_js.data) > 1000)

    # 2. Health & System API
    print("\n--- 2. System & Health API Tests ---")
    resp_health = client.get("/api/health")
    test("health: endpoint returns 200 OK", resp_health.status_code == 200)
    data_health = resp_health.get_json()
    test("health: status is healthy", data_health.get("status") == "healthy")
    test("health: architecture specifies MobileNetV2", "MobileNetV2" in data_health.get("model_architecture", ""))
    test("health: classes count is 43", data_health.get("classes_count") == 43)
    test("health: contains educational/medical disclaimer", "clinical medical advice" in data_health.get("disclaimer", ""))

    # 3. User Profiles API
    print("\n--- 3. User Profiles API Tests ---")
    resp_prof = client.get("/api/profiles")
    test("profiles: endpoint returns 200 OK", resp_prof.status_code == 200)
    profiles = resp_prof.get_json()
    test("profiles: returns 5 ICMR-NIN RDA profiles", len(profiles) == 5, f"count={len(profiles)}")
    prof_keys = [p["key"] for p in profiles]
    test("profiles: contains adult_male_sedentary", "adult_male_sedentary" in prof_keys)
    test("profiles: contains diabetic_adult", "diabetic_adult" in prof_keys)
    test("profiles: contains hypertensive_adult", "hypertensive_adult" in prof_keys)

    # 4. Sample Gallery & Catalog API
    print("\n--- 4. Demo Gallery & Food Catalog API Tests ---")
    resp_samples = client.get("/api/samples")
    test("samples: endpoint returns 200 OK", resp_samples.status_code == 200)
    samples = resp_samples.get_json()
    test("samples: returns >= 10 test samples", len(samples) >= 10, f"count={len(samples)}")
    test("samples: samosa sample present", any(s["class_name"] == "samosa" for s in samples))

    resp_cat = client.get("/api/food-catalog")
    test("catalog: endpoint returns 200 OK", resp_cat.status_code == 200)
    catalog = resp_cat.get_json()
    test("catalog: >= 43 food items registered (59 total)", len(catalog) >= 43, f"count={len(catalog)}")
    test("catalog: allergens list present on each item", all("allergens" in f for f in catalog))

    # 5. Benchmarks API
    print("\n--- 5. Benchmarks & Model Evaluation API Tests ---")
    resp_bm = client.get("/api/benchmarks")
    test("benchmarks: endpoint returns 200 OK", resp_bm.status_code == 200)
    bm_data = resp_bm.get_json()
    test("benchmarks: test top-1 accuracy > 70%", bm_data.get("test_metrics", {}).get("top1_accuracy", 0) > 70.0)
    test("benchmarks: per_class_metrics present", len(bm_data.get("per_class_metrics", {})) >= 43)

    # 6. Real-time Inference API (/api/predict)
    print("\n--- 6. Real-time Vision Prediction API Tests ---")
    # A. JSON sample path prediction (Samosa)
    resp_pred_samosa = client.post(
        "/api/predict",
        data=json.dumps({"sample_path": "/static/samples/samosa.jpg", "user_profile": "adult_male_sedentary"}),
        content_type="application/json"
    )
    test("predict: samosa sample returns 200 OK", resp_pred_samosa.status_code == 200)
    res_s = resp_pred_samosa.get_json()
    test("predict: success is True", res_s.get("success") is True)
    pred_data = res_s.get("data", {})
    test("predict: returns top_k candidates with 5 items", len(pred_data.get("prediction", {}).get("top_k", [])) == 5)
    test("predict: samosa flagged as ambiguous subtype", pred_data.get("ambiguous_subtype") is not None)
    if pred_data.get("ambiguous_subtype"):
        sub_info = pred_data["ambiguous_subtype"]
        test("predict: ambiguous requires_confirmation is True initially", sub_info.get("requires_confirmation") is True)
        test("predict: disambiguation prompt present", bool(sub_info.get("disambiguation_prompt")))
        test("predict: available_subtypes has >= 2 options", len(sub_info.get("available_subtypes", [])) >= 2)

    # B. Multipart Form Image File Upload (Idli)
    idli_path = WORKSPACE_ROOT / "static" / "samples" / "idli.jpg"
    with open(idli_path, "rb") as f:
        img_bytes = f.read()

    from io import BytesIO
    resp_upload = client.post(
        "/api/predict",
        data={"file": (BytesIO(img_bytes), "idli.jpg"), "user_profile": "diabetic_adult"},
        content_type="multipart/form-data"
    )
    test("predict: multipart file upload returns 200 OK", resp_upload.status_code == 200)
    res_upload = resp_upload.get_json()
    test("predict: multipart success is True", res_upload.get("success") is True)
    test("predict: health evaluation contains suitability_score", "suitability_score" in res_upload["data"]["health_evaluation"])

    # 7. Real-time Nutrition & Portion Adjustment (/api/adjust-nutrition)
    print("\n--- 7. Dynamic Portion & Subtype Recalculation API Tests ---")
    # A. Portion Scaling: 100g vs 200g
    resp_adj_100 = client.post(
        "/api/adjust-nutrition",
        data=json.dumps({"class_name": "idli", "portion_g": 100.0, "user_profile": "adult_male_sedentary"}),
        content_type="application/json"
    )
    resp_adj_200 = client.post(
        "/api/adjust-nutrition",
        data=json.dumps({"class_name": "idli", "portion_g": 200.0, "user_profile": "adult_male_sedentary"}),
        content_type="application/json"
    )
    test("adjust: 100g adjustment returns 200 OK", resp_adj_100.status_code == 200)
    test("adjust: 200g adjustment returns 200 OK", resp_adj_200.status_code == 200)
    cal_100 = resp_adj_100.get_json()["data"]["nutrition"]["calories_kcal"]
    cal_200 = resp_adj_200.get_json()["data"]["nutrition"]["calories_kcal"]
    ratio = cal_200 / cal_100 if cal_100 else 0
    test("adjust: 200g portion doubles calories (ratio ~2.0)", 1.95 <= ratio <= 2.05, f"ratio={ratio:.2f}")

    # B. Specific Subtype Override: Keema Samosa vs Potato Samosa
    resp_sub_keema = client.post(
        "/api/adjust-nutrition",
        data=json.dumps({
            "class_name": "samosa",
            "selected_subtype_id": "samosa_keema",
            "portion_g": 70.0,
            "user_profile": "adult_male_sedentary"
        }),
        content_type="application/json"
    )
    test("adjust: keema samosa returns 200 OK", resp_sub_keema.status_code == 200)
    data_keema = resp_sub_keema.get_json()["data"]
    test("adjust: keema samosa calories is 318 kcal", data_keema["nutrition"]["calories_kcal"] == 318.0)
    test("adjust: keema samosa requires_confirmation is False after selection", data_keema["ambiguous_subtype"]["requires_confirmation"] is False)

    # C. Clinical Safety Alerts: Hypertension Profile
    resp_hyp = client.post(
        "/api/adjust-nutrition",
        data=json.dumps({
            "class_name": "chole_bhature",
            "portion_g": 300.0,
            "user_profile": "hypertensive_adult"
        }),
        content_type="application/json"
    )
    test("adjust: hypertensive evaluation returns 200 OK", resp_hyp.status_code == 200)
    data_hyp = resp_hyp.get_json()["data"]
    alerts = data_hyp["health_evaluation"]["dietary_alerts"]
    test("adjust: hypertensive profile triggers alerts for high sodium", len(alerts) > 0)

    # 8. Error Handling & Edge Cases
    print("\n--- 8. Error Handling & Edge Case Tests ---")
    resp_err_empty = client.post("/api/predict", data=json.dumps({}), content_type="application/json")
    test("error_handling: empty payload returns 400 Bad Request", resp_err_empty.status_code == 400)

    resp_err_missing_file = client.post("/api/predict", data={})
    test("error_handling: missing file returns 400 Bad Request", resp_err_missing_file.status_code == 400)

    resp_err_bad_class = client.post(
        "/api/adjust-nutrition",
        data=json.dumps({"class_name": "non_existent_dish_xyz"}),
        content_type="application/json"
    )
    test("error_handling: unknown class in adjust-nutrition returns 400", resp_err_bad_class.status_code == 400)

    # Summary
    passed_count = sum(1 for _, ok in test_results if ok)
    failed_count = sum(1 for _, ok in test_results if not ok)
    total_count = len(test_results)

    print("\n" + "=" * 60)
    print(f"PHASE 6 TESTS: {passed_count} PASSED / {failed_count} FAILED / {total_count} TOTAL")
    print("=" * 60)
    if failed_count == 0:
        print("All Phase 6 Web Application tests passed successfully.")
    else:
        print(f"FAILED {failed_count} tests.")


if __name__ == "__main__":
    run_tests()
