"""
tests/test_ingredient_intelligence.py
Unit & Integration Tests for the Smart Ingredient Intelligence Engine (NutriSense AI)

Covers:
1. Complex packaging ingredient tokenization & nested bracket parsing.
2. INS additive identification and safety/functional classification.
3. Disguised sugar identification & sugar-splitting detection.
4. Maltodextrin glycemic load interpretation.
5. Industrial hydrogenated fats & palm oil alerts.
6. Non-salt sodium compound detection.
7. Profile-specific clinical risk matching (Hypertension, Diabetes, PCOS, Celiac).
8. NOVA Food Processing Grading (NOVA 4 UPF vs NOVA 1 Minimally Processed).
9. Wholesome product benefits extraction.
10. Culturally authentic, budget-friendly Indian healthier alternatives.
11. Verified packaged product catalog lookup (manual fallback).
"""

from pathlib import Path
import sys
import unittest

WORKSPACE_ROOT = Path("c:/Samsung_Project")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.ingredient_intelligence import (
    clean_and_split_ingredients,
    extract_ins_numbers,
    get_ingredient_analyzer,
    lookup_packaged_product,
    INS_DATABASE,
    POPULAR_PACKAGED_FOODS
)

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
    print("=" * 65)
    print("RUNNING SMART INGREDIENT INTELLIGENCE TESTS")
    print("=" * 65)

    analyzer = get_ingredient_analyzer()

    # -------------------------------------------------------------------------
    # 1. Parsing & Bracket Handling
    # -------------------------------------------------------------------------
    print("\n--- 1. Ingredient Parsing & Tokenization ---")
    raw_sample = (
        "Refined Wheat Flour (Maida) (58%), Edible Vegetable Oil (Palmolein), "
        "Sugar, Liquid Glucose, Raising Agents [INS 503(ii), INS 500(ii)], "
        "Iodised Salt, Emulsifiers (INS 322 from Soy, INS 471), Artificial Flavour (Vanilla)."
    )
    tokens = clean_and_split_ingredients(raw_sample)
    record("parse: splits comma items cleanly", len(tokens) >= 7)
    record("parse: preserves nested brackets intact", any("503(ii)" in t for t in tokens))
    record("parse: preserves ingredient percentages", any("58%" in t for t in tokens))

    # -------------------------------------------------------------------------
    # 2. INS Number Resolution
    # -------------------------------------------------------------------------
    print("\n--- 2. INS Additive Resolution ---")
    ins_500 = extract_ins_numbers("Raising Agent (INS 500(ii))")
    record("ins: matches 500(ii) to sodium bicarbonate", len(ins_500) > 0 and "Sodium" in ins_500[0][1]["name"])

    ins_621 = extract_ins_numbers("Flavour Enhancer (INS 621)")
    record("ins: matches 621 to MSG", len(ins_621) > 0 and "Glutamate" in ins_621[0][1]["name"])

    ins_150d = extract_ins_numbers("Colour [150d]")
    record("ins: matches 150d to Caramel IV", len(ins_150d) > 0 and "Caramel" in ins_150d[0][1]["name"])

    ins_955 = extract_ins_numbers("Sweetener (INS 955)")
    record("ins: matches 955 to Sucralose", len(ins_955) > 0 and "Sucralose" in ins_955[0][1]["name"])

    # -------------------------------------------------------------------------
    # 3. Hidden Sugar & Sugar Splitting
    # -------------------------------------------------------------------------
    print("\n--- 3. Hidden Sugar & Sugar Splitting Detection ---")
    multi_sugar_text = "Refined Wheat Flour, Sugar, Invert Sugar Syrup, Liquid Glucose, Maltodextrin, Palm Oil, Salt."
    res_sugar = analyzer.analyze(multi_sugar_text)
    record("sugar: detects multiple hidden sugar sources", any(h["type"] == "MULTIPLE_ADDED_SUGARS" for h in res_sugar["hidden_insights"]))

    sugar_alert = next((h for h in res_sugar["hidden_insights"] if h["type"] == "MULTIPLE_ADDED_SUGARS"), None)
    record("sugar: lists all split sugar sources", sugar_alert is not None and len(sugar_alert["items"]) >= 3)

    # -------------------------------------------------------------------------
    # 4. Maltodextrin Glycemic Warning
    # -------------------------------------------------------------------------
    print("\n--- 4. Maltodextrin Glycemic Impact ---")
    malto_text = "Whey Protein Isolate, Maltodextrin, Cocoa Powder, Stevia, Salt."
    res_malto = analyzer.analyze(malto_text)
    has_malto_flag = any(h["type"] == "MALTODEXTRIN_GLYCEMIC_ALERT" or "maltodextrin" in str(h).lower() for h in res_malto["hidden_insights"])
    record("malto: flags maltodextrin high glycemic index", has_malto_flag)

    # -------------------------------------------------------------------------
    # 5. Palm Oil & Hydrogenated Fats Detection
    # -------------------------------------------------------------------------
    print("\n--- 5. Palm Oil & Hydrogenated Fats Detection ---")
    palm_text = "Wheat Flour, Edible Vegetable Oil (Palmolein), Salt, Spices."
    res_palm = analyzer.analyze(palm_text)
    record("fats: flags palm oil presence and saturated fat load", any(h["type"] == "PALM_OIL_ALERT" for h in res_palm["hidden_insights"]))

    trans_text = "Maida, Partially Hydrogenated Vegetable Fat (Vanaspati), Sugar, Salt."
    res_trans = analyzer.analyze(trans_text)
    record("fats: flags industrial hydrogenated fat with critical warning", any(h["type"] == "HYDROGENATED_FAT_WARNING" for h in res_trans["hidden_insights"]))

    # -------------------------------------------------------------------------
    # 6. Hidden Sodium Sources Beyond Table Salt
    # -------------------------------------------------------------------------
    print("\n--- 6. Hidden Non-Salt Sodium Contributors ---")
    sodium_text = "Noodles: Maida, Palm Oil, Salt. Tastemaker: Salt, Monosodium Glutamate (INS 621), Sodium Bicarbonate (INS 500ii), Spices."
    res_sodium = analyzer.analyze(sodium_text)
    record("sodium: flags non-salt sodium contributors (MSG + Baking Soda)", any(h["type"] == "HIDDEN_SODIUM_SOURCES" for h in res_sodium["hidden_insights"]))

    # -------------------------------------------------------------------------
    # 7. Clinical Health Profile Matching
    # -------------------------------------------------------------------------
    print("\n--- 7. Health Profile Risk Alignment ---")
    hypertensive_profile = {"dietary_restrictions": ["low_sodium"], "dietary_goal": "manage_hypertension"}
    res_hyp = analyzer.analyze(sodium_text, user_profile=hypertensive_profile)
    record("profile: alerts hypertensive user for hidden sodium", any("Hypertension" in a["profile"] for a in res_hyp["profile_alerts"]))

    diabetic_profile = {"dietary_restrictions": ["diabetic_friendly"], "dietary_goal": "manage_diabetes"}
    res_diab = analyzer.analyze(multi_sugar_text, user_profile=diabetic_profile)
    record("profile: alerts diabetic user for rapid-acting sugar spike", any("Diabetes" in a["profile"] for a in res_diab["profile_alerts"]))

    celiac_profile = {"dietary_restrictions": ["gluten_free"]}
    res_celiac = analyzer.analyze("Maida, Sugar, Palm Oil, Salt", user_profile=celiac_profile)
    record("profile: alerts celiac user for refined wheat flour (maida)", any("Gluten" in a["profile"] for a in res_celiac["profile_alerts"]))

    # -------------------------------------------------------------------------
    # 8. NOVA Food Processing Classification
    # -------------------------------------------------------------------------
    print("\n--- 8. NOVA Food Processing Grading ---")
    upf_text = "Maida, Palmolein, Sugar, Flavour Enhancer (INS 621), Colour (INS 150d), Emulsifier (INS 471), TBHQ (INS 319)."
    res_upf = analyzer.analyze(upf_text)
    record("nova: classifies formulation with cosmetic additives as NOVA 4 UPF", res_upf["processing_grade"]["nova_group"] == 4)

    whole_text = "Rolled Oats (100%)."
    res_whole = analyzer.analyze(whole_text)
    record("nova: classifies single whole ingredient as Minimally Processed", res_whole["processing_grade"]["nova_group"] == 1)

    # -------------------------------------------------------------------------
    # 9. Wholesome Benefits Extraction
    # -------------------------------------------------------------------------
    print("\n--- 9. Product Benefits & Positives Extraction ---")
    healthy_snack = "Whole Wheat Atta (70%), Roasted Flax Seeds, Ragi Flour, Rock Salt, Cold-Pressed Mustard Oil."
    res_healthy = analyzer.analyze(healthy_snack)
    record("benefits: identifies whole grain / seed ingredients", len(res_healthy["product_benefits"]) >= 2)
    record("benefits: highlights zero hydrogenated fats", any("hydrogenated" in b.lower() for b in res_healthy["product_benefits"]))

    # -------------------------------------------------------------------------
    # 10. Healthier Alternatives Engine
    # -------------------------------------------------------------------------
    print("\n--- 10. Healthier Alternatives Engine ---")
    noodle_text = "Noodles (Maida, Palm Oil, Salt), Tastemaker (MSG, Salt, Sugar, Starch)."
    res_noodle = analyzer.analyze(noodle_text, user_profile={"budget": "low"})
    alts = res_noodle["healthier_alternatives"]
    record("alts: suggests wholesome Indian alternatives for noodles", len(alts) >= 2)
    record("alts: specifies price tier and calorie savings", any("₹" in a["price_tier"] and "kcal" in a["calories"] for a in alts))

    # -------------------------------------------------------------------------
    # 11. Verified Packaged Food Catalog Lookup
    # -------------------------------------------------------------------------
    print("\n--- 11. Packaged Food Catalog Lookup (Manual Fallback) ---")
    maggi_match = lookup_packaged_product("maggi")
    record("catalog: successfully matches 'maggi' query", maggi_match is not None and "Maggi" in maggi_match["product"]["product_name"])
    record("catalog: contains full pre-parsed nutrition and ingredients", maggi_match["nutrition"]["calories"]["value"] == 427)
    record("catalog: includes pre-computed Smart Ingredient Intelligence", maggi_match["ingredient_intelligence"]["total_ingredients"] > 0)

    goodday_match = lookup_packaged_product("good day butter")
    record("catalog: matches 'good day butter' biscuits", goodday_match is not None and "Good Day" in goodday_match["product"]["product_name"])

    oreo_match = lookup_packaged_product("oreo")
    record("catalog: matches 'oreo' cookies", oreo_match is not None and "Oreo" in oreo_match["product"]["product_name"])

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    passed = sum(1 for _, ok in test_results if ok)
    total = len(test_results)
    print("\n" + "=" * 65)
    print(f"INGREDIENT INTELLIGENCE TESTS: {passed} PASSED / {total - passed} FAILED / {total} TOTAL")
    print("=" * 65)

    if passed != total:
        failed_tests = [name for name, ok in test_results if not ok]
        raise AssertionError(f"Failures in ingredient intelligence suite: {failed_tests}")


if __name__ == "__main__":
    run_tests()
