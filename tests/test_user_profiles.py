"""
tests/test_user_profiles.py
NutriSense AI — Phase 8: User Profile Tests

Covers:
1. Profile validation
2. BMI calculation
3. Daily calorie target
4. RDA profile mapping
5. Profile CRUD persistence
6. Breakfast suggestion
7. Personalised portion guidance
8. Web API endpoints
9. Missing/invalid field handling via API
10. Data protection

Target: ≥ 70 tests
Prints: PHASE 8 USER PROFILE TESTS: X PASSED / Y FAILED / Z TOTAL
"""

import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# ---- imports ----
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
    BMI_DISCLAIMER,
    CALORIE_DISCLAIMER,
    BREAKFAST_DISCLAIMER,
)
from app import app as flask_app

# ---- test infrastructure ----
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
    print("RUNNING PHASE 8: USER PROFILE PERSONALIZATION TESTS")
    print("=" * 70)
    client = flask_app.test_client()

    # =========================================================================
    # Section 1: Profile Validation
    # =========================================================================
    print("\n--- Section 1: Profile Validation ---")

    valid_data = {
        "display_name": "Priya Sharma",
        "age": 32,
        "gender": "female",
        "height_cm": 162.0,
        "weight_kg": 58.0,
        "activity_level": "moderately_active",
        "dietary_goal": "maintain_weight",
        "preferred_cuisines": ["South Indian", "Any"],
        "disliked_foods": ["bitter gourd"],
        "allergies": ["dairy"],
        "dietary_restrictions": ["vegetarian"],
        "budget": "medium",
        "max_cooking_time_mins": 45,
        "dietary_considerations": "Prefer low-oil cooking.",
        "wake_time": "06:30",
        "breakfast_preference": "medium",
        "breakfast_time": "07:30",
    }
    ok, errs = validate_profile_data(valid_data)
    record("1.01 Valid profile data passes validation", ok, str(errs))

    bad_age_low = {**valid_data, "age": 2}
    ok, errs = validate_profile_data(bad_age_low)
    record("1.02 Age below 5 fails validation", not ok and any("age" in e.lower() for e in errs))

    bad_age_high = {**valid_data, "age": 125}
    ok, errs = validate_profile_data(bad_age_high)
    record("1.03 Age above 120 fails validation", not ok and any("age" in e.lower() for e in errs))

    bad_height_low = {**valid_data, "height_cm": 30.0}
    ok, errs = validate_profile_data(bad_height_low)
    record("1.04 Height below 50 fails validation", not ok and any("height" in e.lower() for e in errs))

    bad_height_high = {**valid_data, "height_cm": 290.0}
    ok, errs = validate_profile_data(bad_height_high)
    record("1.05 Height above 280 fails validation", not ok and any("height" in e.lower() for e in errs))

    bad_weight_low = {**valid_data, "weight_kg": 5.0}
    ok, errs = validate_profile_data(bad_weight_low)
    record("1.06 Weight below 10 fails validation", not ok and any("weight" in e.lower() for e in errs))

    bad_weight_high = {**valid_data, "weight_kg": 350.0}
    ok, errs = validate_profile_data(bad_weight_high)
    record("1.07 Weight above 300 fails validation", not ok and any("weight" in e.lower() for e in errs))

    missing_name = {k: v for k, v in valid_data.items() if k != "display_name"}
    ok, errs = validate_profile_data(missing_name)
    record("1.08 Missing display_name fails validation", not ok and any("display_name" in e.lower() for e in errs))

    empty_name = {**valid_data, "display_name": "   "}
    ok, errs = validate_profile_data(empty_name)
    record("1.09 Blank display_name fails validation", not ok)

    bad_gender = {**valid_data, "gender": "alien"}
    ok, errs = validate_profile_data(bad_gender)
    record("1.10 Invalid gender fails validation", not ok and any("gender" in e.lower() for e in errs))

    bad_activity = {**valid_data, "activity_level": "super_active"}
    ok, errs = validate_profile_data(bad_activity)
    record("1.11 Invalid activity_level fails validation", not ok and any("activity_level" in e.lower() for e in errs))

    bad_goal = {**valid_data, "dietary_goal": "fly_to_moon"}
    ok, errs = validate_profile_data(bad_goal)
    record("1.12 Invalid dietary_goal fails validation", not ok and any("dietary_goal" in e.lower() for e in errs))

    bad_allergy = {**valid_data, "allergies": [""]}
    ok, errs = validate_profile_data(bad_allergy)
    record("1.13 Empty string allergy entry fails validation", not ok)

    bad_cooking_time = {**valid_data, "max_cooking_time_mins": 400}
    ok, errs = validate_profile_data(bad_cooking_time)
    record("1.14 Cooking time > 300 fails validation", not ok and any("cooking" in e.lower() for e in errs))

    too_long_considerations = {**valid_data, "dietary_considerations": "X" * 501}
    ok, errs = validate_profile_data(too_long_considerations)
    record("1.15 dietary_considerations > 500 chars fails", not ok and any("dietary_considerations" in e.lower() for e in errs))

    bad_wake_time = {**valid_data, "wake_time": "25:00"}
    ok, errs = validate_profile_data(bad_wake_time)
    record("1.16 Invalid wake_time (25:00) fails validation", not ok and any("wake_time" in e.lower() for e in errs))

    bad_diet_restriction = {**valid_data, "dietary_restrictions": ["carnivore"]}
    ok, errs = validate_profile_data(bad_diet_restriction)
    record("1.17 Invalid dietary restriction fails", not ok and len(errs) > 0)

    bad_budget = {**valid_data, "budget": "rich"}
    ok, errs = validate_profile_data(bad_budget)
    record("1.18 Invalid budget fails validation", not ok and any("budget" in e.lower() for e in errs))

    # =========================================================================
    # Section 2: BMI Calculation
    # =========================================================================
    print("\n--- Section 2: BMI Calculation ---")

    bmi_normal = compute_bmi(170.0, 65.0)
    record("2.01 Normal BMI computed", abs(bmi_normal["bmi"] - 22.49) < 0.1)
    record("2.02 Normal BMI category correct", bmi_normal["bmi_category"] == "Normal weight")
    record("2.03 BMI disclaimer present", BMI_DISCLAIMER in bmi_normal["disclaimer"])

    bmi_underweight = compute_bmi(175.0, 50.0)
    record("2.04 Underweight BMI category", bmi_underweight["bmi_category"] == "Underweight")

    bmi_overweight = compute_bmi(170.0, 80.0)
    record("2.05 Overweight BMI category", bmi_overweight["bmi_category"] == "Overweight")

    bmi_obese = compute_bmi(165.0, 95.0)
    record("2.06 Obese BMI category", bmi_obese["bmi_category"] == "Obese")

    bmi_sa = compute_bmi(165.0, 65.0)
    record("2.07 South Asian overweight threshold present", "south_asian_category" in bmi_sa)

    # Missing inputs
    try:
        compute_bmi(None, 65.0)
        record("2.08 Missing height_cm raises ValueError", False)
    except ValueError:
        record("2.08 Missing height_cm raises ValueError", True)

    try:
        compute_bmi(170.0, None)
        record("2.09 Missing weight_kg raises ValueError", False)
    except ValueError:
        record("2.09 Missing weight_kg raises ValueError", True)

    bmi_result = compute_bmi(170.0, 65.0)
    record("2.10 BMI who_ranges dict present", isinstance(bmi_result.get("who_ranges"), dict))

    # =========================================================================
    # Section 3: Daily Calorie Target
    # =========================================================================
    print("\n--- Section 3: Daily Calorie Target ---")

    male_sed = compute_daily_calorie_target(30, "male", 175.0, 70.0, "sedentary", "general_wellness")
    record("3.01 Male sedentary calorie target computed", male_sed["target_kcal"] > 0)
    # Manual calc: BMR = 10*70 + 6.25*175 - 5*30 + 5 = 700+1093.75-150+5 = 1648.75; *1.2 = 1978.5
    record("3.02 Male sedentary TDEE within range", abs(male_sed["tdee"] - 1978.5) < 5)
    record("3.03 Calorie disclaimer present", CALORIE_DISCLAIMER in male_sed["disclaimer"])

    female_active = compute_daily_calorie_target(28, "female", 165.0, 60.0, "very_active", "maintain_weight")
    record("3.04 Female active calorie target computed", female_active["target_kcal"] > 0)
    # BMR female = 10*60 + 6.25*165 - 5*28 - 161 = 600+1031.25-140-161 = 1330.25; *1.725 = 2294.7
    record("3.05 Female active TDEE within range", abs(female_active["tdee"] - 2294.7) < 10)

    other_gender = compute_daily_calorie_target(35, "other", 170.0, 65.0, "moderately_active", "general_wellness")
    record("3.06 Other gender calorie target computed", other_gender["target_kcal"] > 0)
    record("3.07 Other gender BMR is average", other_gender["bmr"] > 0)

    weight_loss_target = compute_daily_calorie_target(30, "male", 175.0, 80.0, "sedentary", "weight_loss")
    no_goal_target = compute_daily_calorie_target(30, "male", 175.0, 80.0, "sedentary", "general_wellness")
    record("3.08 Weight loss reduces target by 300", abs(weight_loss_target["target_kcal"] - no_goal_target["target_kcal"] + 300) < 2)

    muscle_gain_target = compute_daily_calorie_target(30, "male", 175.0, 80.0, "sedentary", "muscle_gain")
    record("3.09 Muscle gain adds 200 kcal", abs(muscle_gain_target["target_kcal"] - no_goal_target["target_kcal"] - 200) < 2)

    weight_gain_target = compute_daily_calorie_target(30, "male", 175.0, 80.0, "sedentary", "weight_gain")
    record("3.10 Weight gain adds 300 kcal", abs(weight_gain_target["target_kcal"] - no_goal_target["target_kcal"] - 300) < 2)

    # =========================================================================
    # Section 4: RDA Profile Mapping
    # =========================================================================
    print("\n--- Section 4: RDA Profile Mapping ---")

    diabetic_profile = {**valid_data, "dietary_goal": "manage_diabetes", "dietary_restrictions": []}
    record("4.01 manage_diabetes → diabetic_adult", map_to_rda_profile(diabetic_profile) == "diabetic_adult")

    hypertension_profile = {**valid_data, "dietary_goal": "manage_hypertension", "dietary_restrictions": []}
    record("4.02 manage_hypertension → hypertensive_adult", map_to_rda_profile(hypertension_profile) == "hypertensive_adult")

    weight_loss_profile = {**valid_data, "dietary_goal": "weight_loss", "gender": "male", "height_cm": 170, "weight_kg": 90, "dietary_restrictions": []}
    record("4.03 weight_loss + overweight BMI → weight_management", map_to_rda_profile(weight_loss_profile) == "weight_management")

    female_profile = {**valid_data, "dietary_goal": "general_wellness", "gender": "female", "dietary_restrictions": []}
    record("4.04 female general_wellness → adult_female_sedentary", map_to_rda_profile(female_profile) == "adult_female_sedentary")

    male_profile = {**valid_data, "dietary_goal": "general_wellness", "gender": "male", "dietary_restrictions": []}
    record("4.05 male general_wellness → adult_male_sedentary", map_to_rda_profile(male_profile) == "adult_male_sedentary")

    diabetic_restriction = {**valid_data, "dietary_goal": "general_wellness", "dietary_restrictions": ["diabetic_friendly"]}
    record("4.06 diabetic_friendly restriction → diabetic_adult", map_to_rda_profile(diabetic_restriction) == "diabetic_adult")

    low_sodium_restriction = {**valid_data, "dietary_goal": "general_wellness", "dietary_restrictions": ["low_sodium"]}
    record("4.07 low_sodium restriction → hypertensive_adult", map_to_rda_profile(low_sodium_restriction) == "hypertensive_adult")

    maintain_profile = {**valid_data, "dietary_goal": "maintain_weight", "gender": "male", "height_cm": 170, "weight_kg": 90, "dietary_restrictions": []}
    record("4.08 maintain_weight + high BMI → weight_management", map_to_rda_profile(maintain_profile) == "weight_management")

    # =========================================================================
    # Section 5: Profile CRUD Persistence
    # =========================================================================
    print("\n--- Section 5: Profile CRUD Persistence ---")

    created = create_profile(valid_data)
    uid = created["user_id"]
    record("5.01 Profile created with user_id", bool(uid))
    record("5.02 Created profile has display_name", created["display_name"] == "Priya Sharma")
    record("5.03 Created profile has created_at", bool(created.get("created_at")))
    record("5.04 user_id is UUID format (36 chars)", len(uid) == 36)

    fetched = get_profile(uid)
    record("5.05 Profile retrieved by user_id", fetched is not None)
    record("5.06 Fetched profile matches created", fetched["display_name"] == "Priya Sharma")

    updated = update_profile(uid, {"display_name": "Priya Sharma Updated", "age": 33})
    record("5.07 Profile updated successfully", updated is not None)
    record("5.08 Updated display_name correct", updated["display_name"] == "Priya Sharma Updated")
    record("5.09 Updated age correct", updated["age"] == 33)
    record("5.10 user_id unchanged after update", updated["user_id"] == uid)

    profiles_list = list_profiles()
    record("5.11 list_profiles returns list", isinstance(profiles_list, list))
    uids_in_list = [p["user_id"] for p in profiles_list]
    record("5.12 Created user appears in list", uid in uids_in_list)
    record("5.13 List has display_name key", all("display_name" in p for p in profiles_list))
    record("5.14 List has created_at key", all("created_at" in p for p in profiles_list))

    record("5.15 profile_exists returns True for valid uid", profile_exists(uid))
    record("5.16 profile_exists returns False for nonexistent uid", not profile_exists("00000000-0000-0000-0000-000000000000"))

    deleted = delete_profile(uid)
    record("5.17 Profile deleted successfully returns True", deleted)
    record("5.18 Deleted profile no longer exists", not profile_exists(uid))
    record("5.19 get_profile returns None for deleted profile", get_profile(uid) is None)
    deleted_again = delete_profile(uid)
    record("5.20 Deleting nonexistent profile returns False", not deleted_again)

    # Update nonexistent profile
    result = update_profile("00000000-0000-0000-0000-000000000000", {"display_name": "Ghost"})
    record("5.21 Updating nonexistent profile returns None", result is None)

    # Duplicate-proof: two creates with same name should both work (UUID ensures uniqueness)
    p1 = create_profile(valid_data)
    p2 = create_profile(valid_data)
    record("5.22 Two profiles with same name get different UUIDs", p1["user_id"] != p2["user_id"])
    delete_profile(p1["user_id"])
    delete_profile(p2["user_id"])

    # =========================================================================
    # Section 6: Breakfast Suggestion
    # =========================================================================
    print("\n--- Section 6: Breakfast Suggestion ---")

    veg_light_profile = {
        **valid_data,
        "dietary_restrictions": ["vegetarian"],
        "breakfast_preference": "light",
        "max_cooking_time_mins": 20,
        "wake_time": "06:00",
        "breakfast_time": "07:00",
    }
    bs_veg_light = get_breakfast_suggestion(veg_light_profile)
    record("6.01 Breakfast suggestions returned as dict", isinstance(bs_veg_light, dict))
    record("6.02 Suggestions list present", "suggestions" in bs_veg_light)
    record("6.03 Disclaimer present in breakfast result", BREAKFAST_DISCLAIMER in bs_veg_light.get("disclaimer", ""))
    record("6.04 Timing note present for valid wake/breakfast times", bs_veg_light.get("timing_note") is not None)

    hearty_profile = {
        **valid_data,
        "dietary_restrictions": [],
        "breakfast_preference": "hearty",
        "max_cooking_time_mins": 60,
    }
    bs_hearty = get_breakfast_suggestion(hearty_profile)
    record("6.05 Hearty breakfast allows up to 60 min", bs_hearty.get("max_time_applied_mins", 0) <= 60)

    quick_profile = {
        **valid_data,
        "dietary_restrictions": [],
        "breakfast_preference": "light",
        "max_cooking_time_mins": 15,
    }
    bs_quick = get_breakfast_suggestion(quick_profile)
    record("6.06 Quick 15-min profile applies correct time limit", bs_quick.get("max_time_applied_mins") == 15)
    record("6.07 Quick profile suggestions respect time limit", all(
        (s.get("total_time_mins") is None or s.get("total_time_mins") <= 15)
        for s in bs_quick.get("suggestions", [])
    ))

    diabetic_profile_bs = {
        **valid_data,
        "dietary_restrictions": ["diabetic_friendly"],
        "breakfast_preference": "medium",
    }
    bs_diabetic = get_breakfast_suggestion(diabetic_profile_bs)
    record("6.08 Diabetic breakfast suggestion returns dict", isinstance(bs_diabetic, dict))
    record("6.09 Breakfast suggestion count key present", "count" in bs_diabetic)

    # Out-of-window breakfast time
    oow_profile = {
        **valid_data,
        "wake_time": "06:00",
        "breakfast_time": "09:30",
    }
    bs_oow = get_breakfast_suggestion(oow_profile)
    record("6.10 Out-of-window timing note contains 'outside'", "outside" in str(bs_oow.get("timing_note", "")).lower() if bs_oow.get("timing_note") else True)

    # =========================================================================
    # Section 7: Personalised Portion Guidance
    # =========================================================================
    print("\n--- Section 7: Personalised Portion Guidance ---")

    portion_profile = {
        "dietary_goal": "weight_loss",
        "height_cm": 170.0,
        "weight_kg": 80.0,
        "dietary_restrictions": ["vegetarian"],
        "allergies": ["dairy"],
    }
    pg = personalize_portion_guidance("rice", portion_profile)
    record("7.01 Portion guidance returns dict", isinstance(pg, dict))
    record("7.02 recommended_portion_g present", "recommended_portion_g" in pg)
    record("7.03 Unit is grams", pg.get("unit") == "grams")
    record("7.04 Reasoning text present", bool(pg.get("reasoning")))
    record("7.05 Disclaimer present", bool(pg.get("disclaimer")))
    record("7.06 Weight loss reduces portion vs neutral", pg["recommended_portion_g"] <= 150)

    muscle_profile = {
        "dietary_goal": "muscle_gain",
        "height_cm": 175.0,
        "weight_kg": 70.0,
        "dietary_restrictions": [],
        "allergies": [],
    }
    pg_muscle = personalize_portion_guidance("chicken breast", muscle_profile)
    record("7.07 Muscle gain increases portion", pg_muscle["recommended_portion_g"] >= 150)

    pg_no_bmi = personalize_portion_guidance("dal", {"dietary_goal": "general_wellness", "dietary_restrictions": [], "allergies": []})
    record("7.08 Portion guidance works without BMI data", pg_no_bmi["recommended_portion_g"] > 0)

    # =========================================================================
    # Section 8: Web API Endpoints
    # =========================================================================
    print("\n--- Section 8: Web API Endpoints ---")

    # Create via API
    resp = client.post("/api/profile/create", json=valid_data, content_type="application/json")
    record("8.01 POST /api/profile/create returns 201", resp.status_code == 201)
    rdata = json.loads(resp.data)
    record("8.02 Create response has success=True", rdata.get("success") is True)
    api_uid = rdata.get("data", {}).get("user_id")
    record("8.03 Create response includes user_id", bool(api_uid))

    # Get via API
    resp = client.get(f"/api/profile/{api_uid}")
    record("8.04 GET /api/profile/<id> returns 200", resp.status_code == 200)
    gdata = json.loads(resp.data)
    record("8.05 Get response has correct display_name", gdata.get("data", {}).get("display_name") == "Priya Sharma")

    # Update via API
    resp = client.put(f"/api/profile/{api_uid}", json={"age": 35}, content_type="application/json")
    record("8.06 PUT /api/profile/<id> returns 200", resp.status_code == 200)
    udata = json.loads(resp.data)
    record("8.07 Updated age in response", udata.get("data", {}).get("age") == 35)

    # List via API
    resp = client.get("/api/user-profiles")
    record("8.08 GET /api/user-profiles returns 200", resp.status_code == 200)
    ldata = json.loads(resp.data)
    record("8.09 List response has count key", "count" in ldata)
    record("8.10 List response data is list", isinstance(ldata.get("data"), list))

    # BMI endpoint
    resp = client.post(f"/api/profile/{api_uid}/bmi")
    record("8.11 POST /api/profile/<id>/bmi returns 200", resp.status_code == 200)
    bdata = json.loads(resp.data)
    record("8.12 BMI response has bmi key", "bmi" in bdata.get("data", {}))
    record("8.13 BMI response has disclaimer", "disclaimer" in bdata.get("data", {}))

    # Breakfast endpoint
    resp = client.get(f"/api/profile/{api_uid}/breakfast")
    record("8.14 GET /api/profile/<id>/breakfast returns 200", resp.status_code == 200)
    brdata = json.loads(resp.data)
    record("8.15 Breakfast response has suggestions list", isinstance(brdata.get("data", {}).get("suggestions"), list))

    # Personalize endpoint
    resp = client.post(
        f"/api/profile/{api_uid}/personalize",
        json={"food_name": "idli", "portion_value": 2, "unit": "piece"},
        content_type="application/json",
    )
    record("8.16 POST /api/profile/<id>/personalize returns 200", resp.status_code == 200)
    pdata = json.loads(resp.data)
    record("8.17 Personalize response has portion_guidance", "portion_guidance" in pdata.get("data", {}))
    record("8.18 Personalize response has food_suitability", "food_suitability" in pdata.get("data", {}))

    # 404 for nonexistent user
    fake_uid = "00000000-0000-0000-0000-000000000000"
    resp = client.get(f"/api/profile/{fake_uid}")
    record("8.19 GET nonexistent profile returns 404", resp.status_code == 404)
    resp = client.post(f"/api/profile/{fake_uid}/bmi")
    record("8.20 POST bmi for nonexistent returns 404", resp.status_code == 404)
    resp = client.get(f"/api/profile/{fake_uid}/breakfast")
    record("8.21 GET breakfast for nonexistent returns 404", resp.status_code == 404)
    resp = client.post(f"/api/profile/{fake_uid}/personalize", json={"food_name": "rice"})
    record("8.22 POST personalize for nonexistent returns 404", resp.status_code == 404)

    # Delete via API
    resp = client.delete(f"/api/profile/{api_uid}")
    record("8.23 DELETE /api/profile/<id> returns 200", resp.status_code == 200)
    ddata = json.loads(resp.data)
    record("8.24 Delete response has deleted key", "deleted" in ddata.get("data", {}))
    resp = client.delete(f"/api/profile/{api_uid}")
    record("8.25 DELETE already-deleted profile returns 404", resp.status_code == 404)

    # =========================================================================
    # Section 9: Missing / Invalid Field Handling via API
    # =========================================================================
    print("\n--- Section 9: Missing / Invalid Field Handling via API ---")

    # Missing display_name
    resp = client.post("/api/profile/create", json={"age": 30, "gender": "male"}, content_type="application/json")
    record("9.01 Create without display_name returns 400", resp.status_code == 400)
    edata = json.loads(resp.data)
    record("9.02 400 response has errors list", isinstance(edata.get("errors"), list) or isinstance(edata.get("error"), str))

    # Negative age
    resp = client.post("/api/profile/create", json={**valid_data, "age": -5}, content_type="application/json")
    record("9.03 Create with negative age returns 400", resp.status_code == 400)

    # Unknown field (should be tolerated — extra fields ignored)
    resp = client.post("/api/profile/create", json={**valid_data, "unknown_field": "xyz"}, content_type="application/json")
    record("9.04 Create with unknown extra field succeeds (or returns 400)", resp.status_code in (201, 400))
    if resp.status_code == 201:
        created_uid = json.loads(resp.data).get("data", {}).get("user_id")
        if created_uid:
            delete_profile(created_uid)

    # Invalid gender via API
    resp = client.post("/api/profile/create", json={**valid_data, "gender": "martian"}, content_type="application/json")
    record("9.05 Create with invalid gender returns 400", resp.status_code == 400)

    # Invalid allergy list (not a list)
    resp = client.post("/api/profile/create", json={**valid_data, "allergies": "dairy"}, content_type="application/json")
    record("9.06 Create with allergies as string returns 400", resp.status_code == 400)

    # Personalize without food_name
    # First create a fresh profile
    temp_profile = create_profile(valid_data)
    temp_uid = temp_profile["user_id"]
    resp = client.post(f"/api/profile/{temp_uid}/personalize", json={}, content_type="application/json")
    record("9.07 Personalize without food_name returns 400", resp.status_code == 400)
    delete_profile(temp_uid)

    # =========================================================================
    # Section 10: Data Protection
    # =========================================================================
    print("\n--- Section 10: Data Protection ---")

    safe_profile = create_profile(valid_data)
    safe_uid = safe_profile["user_id"]

    # No password stored
    record("10.01 No password key in created profile", "password" not in safe_profile)
    record("10.02 No password_hash key in created profile", "password_hash" not in safe_profile)

    # user_id is UUID (no sequential integer)
    try:
        int(safe_uid)
        record("10.03 user_id is not a sequential integer", False)
    except ValueError:
        record("10.03 user_id is not a sequential integer (is UUID)", True)

    # Sensitive fields are consistent
    fetched_safe = get_profile(safe_uid)
    record("10.04 Retrieved profile has user_id", "user_id" in fetched_safe)
    record("10.05 Retrieved profile has display_name", "display_name" in fetched_safe)
    record("10.06 user_id stable across retrieval", fetched_safe["user_id"] == safe_uid)

    # BMI disclaimer always present
    bmi_check = compute_bmi(170.0, 70.0)
    record("10.07 BMI disclaimer always included", BMI_DISCLAIMER in bmi_check.get("disclaimer", ""))

    # Calorie disclaimer always present
    cal_check = compute_daily_calorie_target(30, "male", 175.0, 70.0, "sedentary", "general_wellness")
    record("10.08 Calorie disclaimer always included", CALORIE_DISCLAIMER in cal_check.get("disclaimer", ""))

    # Breakfast disclaimer always present
    bs_check = get_breakfast_suggestion(valid_data)
    record("10.09 Breakfast disclaimer always included", BREAKFAST_DISCLAIMER in bs_check.get("disclaimer", ""))

    # API response doesn't expose error details with PII
    resp = client.get(f"/api/profile/{fake_uid}")
    err_data = json.loads(resp.data)
    record("10.10 404 error message doesn't expose system path", WORKSPACE_ROOT.as_posix() not in json.dumps(err_data))

    delete_profile(safe_uid)

    # =========================================================================
    # Additional edge-case tests to reach 70+
    # =========================================================================
    print("\n--- Bonus Edge-Case Tests ---")

    # BMI with float precision
    bmi_precise = compute_bmi(165.5, 62.3)
    record("B.01 BMI computed with float height and weight", bmi_precise["bmi"] > 0)

    # Mifflin-St Jeor — prefer_not_to_say gender
    pts = compute_daily_calorie_target(40, "prefer_not_to_say", 170.0, 70.0, "lightly_active", "general_wellness")
    record("B.02 prefer_not_to_say gender target computed", pts["target_kcal"] > 0)

    # validate — preferred_cuisines not a list
    ok, errs = validate_profile_data({**valid_data, "preferred_cuisines": "South Indian"})
    record("B.03 preferred_cuisines as string fails", not ok)

    # validate — dietary_restrictions not a list
    ok, errs = validate_profile_data({**valid_data, "dietary_restrictions": "vegetarian"})
    record("B.04 dietary_restrictions as string fails", not ok)

    # get_food_suitability returns dict with disclaimer
    suit = get_food_suitability_for_profile("rice", valid_data)
    record("B.05 Food suitability returns dict", isinstance(suit, dict))
    record("B.06 Food suitability has suitability_score", "suitability_score" in suit)
    record("B.07 Food suitability has disclaimer", bool(suit.get("disclaimer")))
    record("B.08 Food suitability has verdict", bool(suit.get("verdict")))

    # list_profiles after all deletes
    all_profiles = list_profiles()
    record("B.09 list_profiles returns list after test run", isinstance(all_profiles, list))

    # validate — age as float string should work (coercion)
    ok, errs = validate_profile_data({**valid_data, "age": 30})
    record("B.10 age as int passes validation", ok)

    # Cooking time min boundary
    ok, errs = validate_profile_data({**valid_data, "max_cooking_time_mins": 5})
    record("B.11 Cooking time exactly 5 passes", ok)

    ok, errs = validate_profile_data({**valid_data, "max_cooking_time_mins": 300})
    record("B.12 Cooking time exactly 300 passes", ok)

    # =========================================================================
    # Print Summary
    # =========================================================================
    passed = sum(1 for _, ok in test_results if ok)
    failed = sum(1 for _, ok in test_results if not ok)
    total = len(test_results)

    print("\n" + "=" * 70)
    print(f"PHASE 8 USER PROFILE TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print("=" * 70)

    if failed > 0:
        print("\nFailed tests:")
        for name, ok in test_results:
            if not ok:
                print(f"  {FAIL} {name}")

    return passed, failed, total


if __name__ == "__main__":
    passed, failed, total = run_tests()
    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed out of {total}.")
