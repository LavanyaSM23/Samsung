"""
streamlit_app.py
NutriSense AI — Phase 11: Production Frontend Application

Sections:
1. 🏠 Home                   – Mission, live database metrics, guardrails, quick-nav
2. 👤 User Profile           – Anthropometrics, South-Asian BMI, Mifflin-St Jeor TDEE, preferences
3. 📸 Log a Meal             – Image upload → AI prediction → portion scaling → safety check →
                               suitability score → nutrient breakdown → alternatives → DB save
4. 🔬 Nutrition Analysis     – 1,500+ food lookup, full nutrient table, missing-data integrity,
                               ingredient substitution engine
5. 🍽️ Recommendations        – Safe recipe ranking, calibrated daily plan, 7-day weekly schedule +
                               grocery list, recipe instructions
6. 📊 Nutrition Dashboard    – Meal history, consumed-vs-target, macro splits, 7-day trend,
                               nutrient-gap insights
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="NutriSense AI — Intelligent Nutrition & Meal Planning",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Workspace root ─────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).resolve().parent

# ── Backend imports ────────────────────────────────────────────────────────────
from src.database_manager import (
    get_db_connection,
    seed_reference_data_if_empty,
    log_meal,
    get_meal_history,
    delete_meal_log,
    get_daily_nutrition_totals,
    get_period_nutrition_summary,
    save_meal_plan,
    get_latest_meal_plan,
    log_user_feedback,
)
from src.user_profile_manager import (
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
    list_profiles,
    compute_bmi,
    compute_daily_calorie_target,
    get_breakfast_suggestion,
    BMI_DISCLAIMER,
    CALORIE_DISCLAIMER,
    VALID_GENDERS,
    VALID_ACTIVITY_LEVELS,
    VALID_DIETARY_GOALS,
    VALID_CUISINES,
    VALID_ALLERGIES,
    VALID_DIETARY_RESTRICTIONS,
    VALID_BUDGETS,
    VALID_BREAKFAST_PREFERENCES,
)
from src.nutrition_analyzer import (
    lookup_food,
    calculate_nutrients,
    convert_portion_to_grams,
    search_foods,
    get_available_servings,
    NUTRIENT_UNITS,
    NUTRIENT_LABELS,
    PHASE7_DISCLAIMER,
)
from src.recommendation_engine import (
    get_recommendation_system,
    SafetyFilterEngine,
    SuitabilityScorer,
    AlternativeFoodsEngine,
    IngredientSubstitutionEngine,
    RECOMMENDATION_DISCLAIMER,
    SAFETY_OVERRIDE_NOTICE,
)

# ── One-time DB bootstrap ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _init_db():
    conn = get_db_connection()
    seed_reference_data_if_empty(conn)
    return conn

_db_conn = _init_db()

# ── Cached recommendation system ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_rec_sys():
    return get_recommendation_system()

# ── Cached food catalog (all food names for selectbox) ────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def _get_all_food_names() -> List[str]:
    """Return sorted list of all unique food names from the nutrition analyzer."""
    seen: set = set()
    names: List[str] = []
    for q in list("abcdefghijklmnoprstuvwyz") + ["rice", "dal", "roti", "curry"]:
        for r in search_foods(q, limit=100):
            fn = r["food_name"]
            if fn not in seen:
                seen.add(fn)
                names.append(fn)
    return sorted(names)

SAMPLES_DIR = WORKSPACE_ROOT / "static" / "samples"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = 0.0) -> float:
    """Return float from val, substituting default on None or error."""
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _fmt_nut(info: Dict) -> str:
    """Format a nutrient value with its unit, or 'N/A' when missing."""
    v = info.get("value")
    if v is None:
        return "N/A"
    return f"{v} {info.get('unit', '')}"


def _fmt_nut_100g(info: Dict) -> str:
    v = info.get("value_per_100g")
    if v is None:
        return "N/A"
    return f"{v} {info.get('unit', '')}"


def _calorie_target_from_profile(profile: Dict) -> float:
    """Compute calorie target safely; fall back to 2000 kcal."""
    try:
        res = compute_daily_calorie_target(
            int(profile.get("age", 30)),
            profile.get("gender", "prefer_not_to_say"),
            float(profile.get("height_cm", 170) or 170),
            float(profile.get("weight_kg", 65) or 65),
            profile.get("activity_level", "sedentary"),
            profile.get("dietary_goal", "general_wellness"),
        )
        return float(res["target_calories_kcal"])
    except Exception:
        return 2000.0


# ── Session-state bootstrap ────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "active_user_id": None,
        "last_logged_meal": None,
        "current_daily_plan": None,
        "current_weekly_plan": None,
        "nav_target": None,
        "log_meal_food": None,          # pre-select food in Log a Meal
        "infer_result": None,           # last AI prediction result
        "confirm_delete_meal": None,    # log_id pending delete confirm
        "confirm_delete_profile": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Seed active_user_id from first profile if unset
    if st.session_state["active_user_id"] is None:
        profiles = list_profiles()
        if profiles:
            st.session_state["active_user_id"] = profiles[0]["user_id"]

_init_session()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥗 NutriSense AI")
    st.caption("Evidence-Based Indian Nutrition & Meal Intelligence")
    st.markdown("---")

    SECTIONS = [
        "🏠 Home",
        "👤 User Profile",
        "📸 Log a Meal",
        "🔬 Nutrition Analysis",
        "🍽️ Recommendations & Meal Planning",
        "📊 Nutrition Dashboard",
    ]

    # Handle quick-nav button targets
    _nav_target = st.session_state.pop("nav_target", None)
    _default_idx = SECTIONS.index(_nav_target) if _nav_target in SECTIONS else 0

    selected_section = st.radio(
        "Navigate",
        SECTIONS,
        index=_default_idx,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("Active User")

    all_profiles = list_profiles()
    if all_profiles:
        profile_map = {p["user_id"]: f"{p['display_name']} ({p['user_id'][:6]}…)" for p in all_profiles}
        cur_uid = st.session_state["active_user_id"]
        if cur_uid not in profile_map:
            cur_uid = list(profile_map.keys())[0]
            st.session_state["active_user_id"] = cur_uid

        sel_uid = st.selectbox(
            "Profile",
            options=list(profile_map.keys()),
            format_func=lambda x: profile_map[x],
            index=list(profile_map.keys()).index(cur_uid),
            label_visibility="collapsed",
        )
        if sel_uid != st.session_state["active_user_id"]:
            st.session_state["active_user_id"] = sel_uid
            st.rerun()

        active_user: Optional[Dict] = get_profile(st.session_state["active_user_id"])
        if active_user:
            st.success(f"**{active_user['display_name']}**")
            goal_lbl = active_user.get("dietary_goal", "general_wellness").replace("_", " ").title()
            st.caption(f"🎯 {goal_lbl}")
            allergens = active_user.get("allergies") or []
            if allergens:
                st.caption(f"⚠️ Allergens: {', '.join(a.title() for a in allergens)}")
    else:
        active_user = None
        st.warning("No profiles yet. Create one in 👤 User Profile.")

    st.markdown("---")
    st.caption("NutriSense AI v2.4 · ICMR-NIN 2020 · IFCT 2017")


# ===========================================================================
# SECTION 1 — HOME
# ===========================================================================
if selected_section == "🏠 Home":
    st.title("🥗 NutriSense AI")
    st.subheader("Personalized Indian Nutrition, Visual Food Intelligence & Clinical Diet Planning")

    # Live database metrics
    cur = _db_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM foods;")
    kb_food_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM recipes;")
    kb_recipe_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM meal_logs;")
    total_logged_meals = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM user_profiles;")
    total_profiles = cur.fetchone()[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Curated Indian Recipes", f"{kb_recipe_count:,}")
    m2.metric("Laboratory Foods", f"{kb_food_count:,}")
    m3.metric("Meals Logged (All Users)", f"{total_logged_meals:,}")
    m4.metric("Registered Profiles", f"{total_profiles:,}")

    st.markdown("---")

    col_left, col_right = st.columns([1.3, 0.7])
    with col_left:
        st.markdown("### Core Capabilities")
        st.markdown("""
- **📸 Visual Food Recognition** — Fine-tuned MobileNetV2 identifying 43 authentic Indian dish categories with subtype disambiguation.
- **🔬 Laboratory-Grade Nutrition Scaling** — Per-100g composition tables from ICMR-NIN (2020) and IFCT (2017). Missing micronutrients are **never** fabricated.
- **🛡️ Pre-Ranking Safety Filter** — Zero-tolerance allergen and dietary restriction enforcement. High suitability scores can never override safety.
- **🎯 Transparent 0–100 Suitability Scorer** — Multi-factor evaluation: nutrient targets, taste preferences, cooking time, budget, and goal synergy.
- **📅 Dynamic Meal Planning** — Calibrated 4-meal daily plans and 7-day weekly schedules with grocery shopping lists.
- **📊 Trend Dashboard** — Meal history, calorie/macro tracking, and 7-day nutrient-gap insights.
        """)
        st.info("💡 **Quick Start**: Create your profile in **👤 User Profile**, then visit **📸 Log a Meal** to upload a food photo for real-time AI recognition, nutrition analysis, and safety-checked guidance.")

    with col_right:
        st.markdown("### Clinical Guardrails")
        st.warning(
            "**Educational Tool Notice**: NutriSense AI is an algorithmic decision-support tool "
            "for lifestyle awareness. It does not diagnose, prescribe, or replace a licensed "
            "clinical dietitian or physician."
        )
        st.caption(BMI_DISCLAIMER)

        if active_user:
            st.markdown("### Your Snapshot")
            try:
                bmi = compute_bmi(
                    float(active_user.get("height_cm", 170) or 170),
                    float(active_user.get("weight_kg", 65) or 65),
                )
                cal = _calorie_target_from_profile(active_user)
                st.metric("BMI", f"{bmi['bmi']:.1f}", bmi.get("bmi_category", bmi.get("who_category", "")))
                st.metric("Daily Calorie Target", f"{cal:.0f} kcal")
            except Exception:
                pass

    st.markdown("---")
    st.markdown("### Quick Navigation")
    qc1, qc2, qc3, qc4 = st.columns(4)
    if qc1.button("📸 Log a Meal", use_container_width=True):
        st.session_state["nav_target"] = "📸 Log a Meal"
        st.rerun()
    if qc2.button("👤 User Profile", use_container_width=True):
        st.session_state["nav_target"] = "👤 User Profile"
        st.rerun()
    if qc3.button("🍽️ Recommendations", use_container_width=True):
        st.session_state["nav_target"] = "🍽️ Recommendations & Meal Planning"
        st.rerun()
    if qc4.button("📊 Dashboard", use_container_width=True):
        st.session_state["nav_target"] = "📊 Nutrition Dashboard"
        st.rerun()


# ===========================================================================
# SECTION 2 — USER PROFILE
# ===========================================================================
elif selected_section == "👤 User Profile":
    st.title("👤 User Profile & Health Settings")
    st.markdown(
        "Configure your anthropometrics, clinical dietary goals, food preferences, and allergies. "
        "All recommendations and suitability scoring adapt dynamically to these settings."
    )

    tab_view, tab_create = st.tabs(["📋 View / Edit Active Profile", "➕ Create New Profile"])

    # ── VIEW / EDIT ──────────────────────────────────────────────────────────
    with tab_view:
        if not active_user:
            st.info("No active profile selected. Create one in the **➕ Create New Profile** tab.")
        else:
            uid = active_user["user_id"]
            st.subheader(f"Profile: {active_user['display_name']}  `{uid[:8]}…`")

            # Anthropometrics display
            age    = int(active_user.get("age", 30) or 30)
            gender = active_user.get("gender", "prefer_not_to_say") or "prefer_not_to_say"
            height = _safe_float(active_user.get("height_cm"), 170.0)
            weight = _safe_float(active_user.get("weight_kg"), 65.0)

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Age", f"{age} yrs")
            a2.metric("Gender", gender.replace("_", " ").title())
            a3.metric("Height", f"{height:.1f} cm")
            a4.metric("Weight", f"{weight:.1f} kg")

            # BMI
            try:
                bmi_res = compute_bmi(height, weight)
                b1, b2 = st.columns([1, 2])
                with b1:
                    st.metric(
                        "BMI",
                        f"{bmi_res['bmi']:.1f}",
                        f"{bmi_res.get('bmi_category', bmi_res.get('who_category', 'Normal'))} · South Asian: {bmi_res['south_asian_category']}",
                    )
                with b2:
                    st.info(
                        f"**WHO Normal**: {bmi_res['who_ranges']['normal']} kg/m²  \n"
                        f"**South Asian Consensus Range**: ≥23.0 · **Obese**: ≥27.5 kg/m²"
                    )
                    st.caption(bmi_res["disclaimer"])
            except Exception as ex:
                st.error(f"BMI computation error: {ex}")

            # Calorie target
            try:
                cal_res = compute_daily_calorie_target(
                    age, gender, height, weight,
                    active_user.get("activity_level", "sedentary"),
                    active_user.get("dietary_goal", "general_wellness"),
                )
                cc1, cc2 = st.columns([1, 2])
                with cc1:
                    st.metric("Daily Calorie Target", f"{cal_res['target_calories_kcal']:.0f} kcal")
                with cc2:
                    st.markdown(
                        f"**BMR**: {cal_res['bmr_kcal']:.0f} kcal · "
                        f"**TDEE**: {cal_res['tdee_kcal']:.0f} kcal · "
                        f"**Goal Adj**: {cal_res['goal_adjustment_kcal']:+.0f} kcal"
                    )
                    st.caption(cal_res["explanation"])
            except Exception as ex:
                st.error(f"Calorie target error: {ex}")

            # Breakfast suggestion
            try:
                bs = get_breakfast_suggestion(active_user)
                if bs and bs.get("count", 0) > 0:
                    with st.expander("☀️ Today's Breakfast Suggestions", expanded=False):
                        st.caption(bs.get("timing_note", ""))
                        for s in bs.get("suggestions", [])[:3]:
                            st.markdown(
                                f"- **{s.get('recipe_name', s.get('name', 'Unknown'))}** "
                                f"— {s.get('explanation', '')} "
                                f"*(Score: {s.get('suitability_score', 0):.0f}/100)*"
                            )
                        st.caption(bs.get("disclaimer", ""))
            except Exception:
                pass

            st.markdown("---")
            st.markdown("#### ✏️ Edit Profile Parameters")

            with st.form("edit_profile_form", border=True):
                fe1, fe2, fe3 = st.columns(3)
                with fe1:
                    new_name   = st.text_input("Display Name", value=active_user["display_name"])
                    new_age    = st.number_input("Age (years)", min_value=5, max_value=120, value=age, step=1)
                    genders_sorted = sorted(VALID_GENDERS)
                    new_gender = st.selectbox(
                        "Gender", genders_sorted,
                        index=genders_sorted.index(gender) if gender in genders_sorted else 0,
                    )
                with fe2:
                    new_height  = st.number_input("Height (cm)", min_value=50.0, max_value=280.0, value=height, step=0.5)
                    new_weight  = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=weight, step=0.5)
                    act_sorted  = sorted(VALID_ACTIVITY_LEVELS)
                    cur_act     = active_user.get("activity_level", "sedentary") or "sedentary"
                    new_activity = st.selectbox(
                        "Activity Level", act_sorted,
                        index=act_sorted.index(cur_act) if cur_act in act_sorted else 0,
                    )
                with fe3:
                    goal_sorted = sorted(VALID_DIETARY_GOALS)
                    cur_goal    = active_user.get("dietary_goal", "general_wellness") or "general_wellness"
                    new_goal    = st.selectbox(
                        "Dietary Goal", goal_sorted,
                        index=goal_sorted.index(cur_goal) if cur_goal in goal_sorted else 0,
                    )
                    bud_sorted  = sorted(VALID_BUDGETS)
                    cur_bud     = active_user.get("budget", "medium") or "medium"
                    new_budget  = st.selectbox(
                        "Kitchen Budget", bud_sorted,
                        index=bud_sorted.index(cur_bud) if cur_bud in bud_sorted else 1,
                    )
                    new_cook_time = st.slider(
                        "Max Cooking Time (mins)", 10, 180,
                        value=int(active_user.get("max_cooking_time_mins", 45) or 45),
                    )

                pe1, pe2 = st.columns(2)
                with pe1:
                    new_cuisines = st.multiselect(
                        "Preferred Cuisines", sorted(VALID_CUISINES),
                        default=[c for c in (active_user.get("preferred_cuisines") or []) if c in VALID_CUISINES],
                    )
                    new_allergies = st.multiselect(
                        "Declared Allergies (Zero-Tolerance)", sorted(VALID_ALLERGIES),
                        default=[a for a in (active_user.get("allergies") or []) if a in VALID_ALLERGIES],
                    )
                with pe2:
                    new_restrictions = st.multiselect(
                        "Dietary Restrictions", sorted(VALID_DIETARY_RESTRICTIONS),
                        default=[r for r in (active_user.get("dietary_restrictions") or []) if r in VALID_DIETARY_RESTRICTIONS],
                    )
                    new_dislikes = st.text_input(
                        "Disliked Ingredients (comma-separated)",
                        value=", ".join(active_user.get("disliked_foods") or []),
                    )

                st.markdown("##### ☀️ Morning Routine")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    wake_str = active_user.get("wake_time") or "07:00"
                    try:
                        wake_t = datetime.strptime(wake_str, "%H:%M").time()
                    except ValueError:
                        wake_t = time(7, 0)
                    new_wake = st.time_input("Wake-Up Time", value=wake_t)
                with mc2:
                    bfast_str = active_user.get("breakfast_time") or "08:30"
                    try:
                        bfast_t = datetime.strptime(bfast_str, "%H:%M").time()
                    except ValueError:
                        bfast_t = time(8, 30)
                    new_bfast_time = st.time_input("Breakfast Time", value=bfast_t)
                with mc3:
                    pref_sorted = sorted(VALID_BREAKFAST_PREFERENCES)
                    cur_pref    = active_user.get("breakfast_preference", "medium") or "medium"
                    new_bfast_pref = st.selectbox(
                        "Breakfast Style", pref_sorted,
                        index=pref_sorted.index(cur_pref) if cur_pref in pref_sorted else 1,
                    )

                new_notes = st.text_area(
                    "Additional Dietary Notes",
                    value=active_user.get("dietary_considerations", "") or "",
                    max_chars=500,
                )

                submitted = st.form_submit_button("💾 Save Profile Updates", use_container_width=True, type="primary")
                if submitted:
                    if not new_name.strip():
                        st.error("Display Name cannot be empty.")
                    else:
                        updates = {
                            "display_name":          new_name.strip(),
                            "age":                   int(new_age),
                            "gender":                new_gender,
                            "height_cm":             float(new_height),
                            "weight_kg":             float(new_weight),
                            "activity_level":        new_activity,
                            "dietary_goal":          new_goal,
                            "budget":                new_budget,
                            "max_cooking_time_mins": int(new_cook_time),
                            "preferred_cuisines":    new_cuisines,
                            "allergies":             new_allergies,
                            "dietary_restrictions":  new_restrictions,
                            "disliked_foods":        [d.strip() for d in new_dislikes.split(",") if d.strip()],
                            "wake_time":             new_wake.strftime("%H:%M"),
                            "breakfast_time":        new_bfast_time.strftime("%H:%M"),
                            "breakfast_preference":  new_bfast_pref,
                            "dietary_considerations": new_notes.strip(),
                        }
                        try:
                            update_profile(uid, updates)
                            st.success("✅ Profile updated successfully!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Update failed: {ex}")

            st.markdown("---")
            st.markdown("#### ⚠️ Danger Zone")
            if not st.session_state["confirm_delete_profile"]:
                if st.button("🗑️ Delete This Profile", type="secondary"):
                    st.session_state["confirm_delete_profile"] = True
                    st.rerun()
            else:
                st.error(f"Are you sure you want to permanently delete **{active_user['display_name']}** and all their meal logs?")
                dc1, dc2 = st.columns(2)
                if dc1.button("✅ Yes, Delete", type="primary"):
                    try:
                        delete_profile(uid)
                        st.session_state["active_user_id"] = None
                        st.session_state["confirm_delete_profile"] = False
                        st.warning("Profile deleted.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Delete failed: {ex}")
                if dc2.button("❌ Cancel"):
                    st.session_state["confirm_delete_profile"] = False
                    st.rerun()

    # ── CREATE ───────────────────────────────────────────────────────────────
    with tab_create:
        st.subheader("Create a New User Profile")
        with st.form("create_profile_form", border=True):
            cc1, cc2 = st.columns(2)
            with cc1:
                c_name    = st.text_input("Full Name / Display Name *", placeholder="e.g. Priya Sharma")
                c_age     = st.number_input("Age (5–120)", min_value=5, max_value=120, value=28, step=1)
                c_gender  = st.selectbox("Gender", sorted(VALID_GENDERS))
                c_height  = st.number_input("Height (cm)", min_value=50.0, max_value=280.0, value=165.0, step=0.5)
                c_weight  = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0, step=0.5)
            with cc2:
                c_act     = st.selectbox("Physical Activity Level", sorted(VALID_ACTIVITY_LEVELS))
                c_goal    = st.selectbox("Primary Dietary Goal", sorted(VALID_DIETARY_GOALS))
                bud_opts  = sorted(VALID_BUDGETS)
                c_budget  = st.selectbox("Kitchen Budget", bud_opts, index=bud_opts.index("medium") if "medium" in bud_opts else 1)
                c_cook    = st.slider("Max Cooking Window (mins)", 10, 180, value=45)

            cp1, cp2 = st.columns(2)
            with cp1:
                c_cuisines = st.multiselect(
                    "Preferred Cuisines", sorted(VALID_CUISINES),
                    default=["North Indian", "South Indian"] if "North Indian" in VALID_CUISINES else [],
                )
                c_allergies = st.multiselect("Allergies (Zero-Tolerance Filter)", sorted(VALID_ALLERGIES))
            with cp2:
                veg_default = ["vegetarian"] if "vegetarian" in VALID_DIETARY_RESTRICTIONS else []
                c_restrictions = st.multiselect("Dietary Restrictions", sorted(VALID_DIETARY_RESTRICTIONS), default=veg_default)
                c_dislikes = st.text_input("Disliked Ingredients (comma-separated)", placeholder="e.g. bitter gourd, raw onion")

            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                c_wake  = st.time_input("Wake-Up Time", value=time(7, 0))
            with cm2:
                c_bfast = st.time_input("Breakfast Time", value=time(8, 30))
            with cm3:
                pref_opts = sorted(VALID_BREAKFAST_PREFERENCES)
                c_bstyle  = st.selectbox("Breakfast Style", pref_opts, index=pref_opts.index("medium") if "medium" in pref_opts else 1)

            if st.form_submit_button("✨ Create Profile", use_container_width=True, type="primary"):
                if not c_name.strip():
                    st.error("Display Name is required.")
                else:
                    new_data = {
                        "display_name":          c_name.strip(),
                        "age":                   int(c_age),
                        "gender":                c_gender,
                        "height_cm":             float(c_height),
                        "weight_kg":             float(c_weight),
                        "activity_level":        c_act,
                        "dietary_goal":          c_goal,
                        "budget":                c_budget,
                        "max_cooking_time_mins": int(c_cook),
                        "preferred_cuisines":    c_cuisines,
                        "allergies":             c_allergies,
                        "dietary_restrictions":  c_restrictions,
                        "disliked_foods":        [d.strip() for d in c_dislikes.split(",") if d.strip()],
                        "wake_time":             c_wake.strftime("%H:%M"),
                        "breakfast_time":        c_bfast.strftime("%H:%M"),
                        "breakfast_preference":  c_bstyle,
                    }
                    try:
                        p = create_profile(new_data)
                        st.session_state["active_user_id"] = p["user_id"]
                        st.success(f"🎉 Profile **{c_name.strip()}** created! You are now logged in as this profile.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Profile creation failed: {ex}")


# ===========================================================================
# SECTION 3 — LOG A MEAL
# ===========================================================================
elif selected_section == "📸 Log a Meal":
    st.title("📸 Log a Meal")
    st.markdown(
        "Identify food visually or search the database, adjust portion weight, verify safety constraints, "
        "and save to your daily nutrition log."
    )

    if not active_user:
        st.warning("⚠️ Please create or select a user profile in the sidebar before logging meals.")
        st.stop()

    col_input, col_analysis = st.columns([1.1, 1.3])

    # ── LEFT: Food selection ─────────────────────────────────────────────────
    with col_input:
        st.subheader("1. Food Selection & Image Recognition")

        input_mode = st.radio(
            "Input Mode",
            ["📤 Upload Photo", "Select Sample Dish Image", "🔍 Manual Search"],
            horizontal=True,
            label_visibility="collapsed",
        )

        selected_image: Optional[Image.Image] = None
        infer_predicted_class: Optional[str] = None
        infer_display_name: Optional[str] = None
        infer_confidence: float = 0.0
        infer_top_k: List[Dict] = []

        if input_mode == "📤 Upload Photo":
            uploaded = st.file_uploader("Upload food photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                selected_image = Image.open(uploaded).convert("RGB")
                st.image(selected_image, caption="Uploaded Photo", use_container_width=True)

        elif input_mode == "Select Sample Dish Image":
            sample_files = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
            if sample_files:
                sample_map = {f.stem.replace("_", " ").title(): f for f in sample_files}
                chosen = st.selectbox("Choose sample image", list(sample_map.keys()))
                selected_image = Image.open(sample_map[chosen]).convert("RGB")
                st.image(selected_image, caption=f"Sample: {chosen}", use_container_width=True)
            else:
                st.info("No sample images found in `static/samples/`. Use photo upload instead.")

        else:  # Manual Search
            st.info("ℹ️ Type to search the food database directly. No image required.")

        # AI Vision Inference
        if selected_image is not None:
            with st.spinner("Analyzing image with MobileNetV2 Food Classifier…"):
                try:
                    from src.inference_pipeline import NutriSenseInferenceEngine
                    _inf_eng = NutriSenseInferenceEngine()
                    pred_res = _inf_eng.analyze_food(selected_image, top_k=5)
                    pred     = pred_res.get("prediction", {})
                    infer_predicted_class = pred.get("class_name") or None
                    infer_display_name    = pred.get("display_name", "Unknown")
                    infer_confidence      = _safe_float(pred.get("confidence"), 0.0)
                    infer_top_k           = pred.get("top_k", [])

                    conf_pct = infer_confidence * 100
                    if conf_pct >= 60:
                        st.success(f"🤖 AI Prediction: **{infer_display_name}** ({conf_pct:.1f}% confidence)")
                    elif conf_pct >= 30:
                        st.warning(f"🤖 AI Prediction (Low Confidence): **{infer_display_name}** ({conf_pct:.1f}% — low confidence). Please confirm below.")
                    else:
                        st.error(f"🤖 Uncertain prediction ({conf_pct:.1f}%). Please manually select the correct dish.")

                    if len(infer_top_k) > 1:
                        with st.expander("📊 All Top Predictions"):
                            for cand in infer_top_k:
                                st.markdown(
                                    f"- **{cand['display_name']}** — {cand['confidence']*100:.1f}%"
                                )
                except Exception as ex:
                    st.warning(f"Vision model unavailable: {ex}. Please select the food manually below.")

        # Food selection / confirmation
        st.markdown("#### Confirm Food Item")
        all_foods = _get_all_food_names()
        if not all_foods:
            st.error("Food database not loaded. Please check the backend setup.")
            st.stop()

        # Try to auto-select the predicted class
        pre_select_food = st.session_state.get("log_meal_food") or infer_predicted_class

        # Build default index
        def _best_idx(target: Optional[str], options: List[str]) -> int:
            if not target:
                return 0
            # exact match
            for i, fn in enumerate(options):
                if fn == target:
                    return i
            # partial match
            target_norm = target.replace("_", " ").lower()
            for i, fn in enumerate(options):
                if target_norm in fn.lower() or fn.lower() in target_norm:
                    return i
            return 0

        confirmed_food = st.selectbox(
            "Confirmed Dish / Food Item",
            options=all_foods,
            index=_best_idx(pre_select_food, all_foods),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        # Clear the pre-select after use
        if st.session_state.get("log_meal_food"):
            st.session_state["log_meal_food"] = None

        # Portion sizing
        st.markdown("#### Portion Sizing")
        p1, p2 = st.columns(2)
        with p1:
            portion_value = st.number_input(
                "Portion Amount",
                min_value=0.25, max_value=2000.0, value=1.0, step=0.25,
                help="Enter quantity in the unit selected to the right.",
            )
        with p2:
            servings_opts = get_available_servings(confirmed_food)
            unit_map      = {s["unit"]: s["label"] for s in servings_opts}
            # Default to 'standard_serving' if available
            default_unit_idx = 0
            for i, u in enumerate(unit_map.keys()):
                if "default" in str(servings_opts[i].get("is_default", "")):
                    default_unit_idx = i
                    break
            unit_choice = st.selectbox(
                "Serving Unit",
                options=list(unit_map.keys()),
                index=default_unit_idx,
                format_func=lambda u: unit_map.get(u, u),
            )

        try:
            portion_grams = convert_portion_to_grams(portion_value, unit=unit_choice, food_name=confirmed_food)
            st.caption(f"≈ **{portion_grams:.0f} g** will be logged.")
        except Exception:
            portion_grams = portion_value * 100.0
            st.caption(f"≈ **{portion_grams:.0f} g** (estimated)")

        meal_course = st.selectbox("Meal Occasion", ["Breakfast", "Lunch", "Snack", "Dinner"], index=1)
        meal_notes  = st.text_input("Notes (optional)", placeholder="e.g. Home cooked, light oil")

    # ── RIGHT: Nutrition & safety analysis ──────────────────────────────────
    with col_analysis:
        st.subheader("2. Real-Time Nutrition & Safety")

        food_rec = lookup_food(confirmed_food)

        if food_rec is None:
            st.warning(f"⚠️ No detailed nutrition record found for **{confirmed_food.replace('_', ' ').title()}**. Try a different spelling or select from the database.")
        else:
            try:
                scaled_nutrition = calculate_nutrients(food_rec, float(portion_grams))
                nut = scaled_nutrition["nutrients"]

                # Macro metrics
                cal_val  = _safe_float(nut.get("calories_kcal", {}).get("value"))
                prot_val = _safe_float(nut.get("protein_g",    {}).get("value"))
                carb_val = _safe_float(nut.get("carbs_g",      {}).get("value"))
                fat_val  = _safe_float(nut.get("fat_g",        {}).get("value"))
                fib_val  = _safe_float(nut.get("fibre_g",      {}).get("value"))
                sod_val  = _safe_float(nut.get("sodium_mg",    {}).get("value"))

                nm1, nm2, nm3, nm4 = st.columns(4)
                nm1.metric("Calories",  f"{cal_val:.0f} kcal")
                nm2.metric("Protein",   f"{prot_val:.1f} g")
                nm3.metric("Carbs",     f"{carb_val:.1f} g")
                nm4.metric("Fat",       f"{fat_val:.1f} g")

                nf1, nf2 = st.columns(2)
                nf1.metric("Dietary Fibre", f"{fib_val:.1f} g" if fib_val else "N/A")
                nf2.metric("Sodium",        f"{sod_val:.0f} mg" if sod_val else "N/A")

                # Full nutrient table
                with st.expander("📋 Full Nutrient Breakdown", expanded=False):
                    nut_rows = []
                    for k, info in nut.items():
                        nut_rows.append({
                            "Nutrient":          info.get("label", k),
                            f"Per {portion_grams:.0f} g": _fmt_nut(info),
                            "Per 100 g":         _fmt_nut_100g(info),
                            "Status":            info.get("status", ""),
                        })
                    if nut_rows:
                        st.dataframe(pd.DataFrame(nut_rows), use_container_width=True, hide_index=True)

                    if scaled_nutrition.get("missing_nutrients"):
                        st.info("Missing laboratory micronutrients (not fabricated):")
                        for m in scaled_nutrition["missing_nutrients"]:
                            st.markdown(f"- **{m['label']}**: {m['reason']}")

                # Safety check
                st.markdown("#### 🛡️ Safety & Restriction Audit")
                safety_eng = SafetyFilterEngine()
                proxy_rec = {
                    "recipe_name":        food_rec.get("display_name", confirmed_food),
                    "ingredients_raw":    food_rec.get("reference_basis", confirmed_food),
                    "ingredient_tokens":  confirmed_food.replace("_", " "),
                    "diet":               "vegetarian" if "meat" not in confirmed_food else "non_vegetarian",
                }
                safety_res = safety_eng.check_safety(proxy_rec, active_user)

                if safety_res["is_safe"]:
                    st.success("🛡️ **Safety Certified** — No conflicts with your allergies or dietary restrictions.")
                else:
                    st.error(f"⚠️ **Safety Conflict** — {safety_res['safety_reason']}")
                    if safety_res.get("allergen_conflicts"):
                        st.markdown("**Allergen triggers**: " + ", ".join(safety_res["allergen_conflicts"]))
                    if safety_res.get("restriction_conflicts"):
                        st.markdown("**Restriction violations**: " + ", ".join(safety_res["restriction_conflicts"]))

                # Suitability score
                st.markdown("#### 🎯 Personal Suitability Score")
                scorer     = SuitabilityScorer()
                score_eval = scorer.score_recipe(proxy_rec, active_user)
                score_val  = _safe_float(score_eval.get("total_score"), 0.0)

                sc1, sc2 = st.columns([1, 2])
                with sc1:
                    verdict_lbl = score_eval.get("verdict", "").replace("_", " ").title()
                    st.metric("Score", f"{score_val:.0f} / 100", verdict_lbl)
                with sc2:
                    st.info(score_eval.get("explanation", ""))

                with st.expander("📊 Score Component Breakdown"):
                    comp_df = pd.DataFrame([
                        {"Component": k.replace("_", " ").title(), "Points": round(float(v), 1)}
                        for k, v in (score_eval.get("component_scores") or {}).items()
                    ])
                    if not comp_df.empty:
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    st.caption(score_eval.get("disclaimer", ""))

                # Healthier alternatives (shown when score < 75)
                alt_eng = AlternativeFoodsEngine()
                alts    = alt_eng.get_alternatives(food_rec.get("class_name", confirmed_food), user_profile=active_user)
                if alts and score_val < 75.0:
                    with st.expander("💡 Healthier Alternative Suggestions", expanded=score_val < 55):
                        for alt in alts[:3]:
                            st.markdown(
                                f"**{alt['alternative_name']}** — {alt['benefit_explanation']}"
                                + (f" *(~{alt['estimated_calories']:.0f} kcal/100g)*" if alt.get("estimated_calories") else "")
                            )
                        st.caption(alts[0].get("disclaimer", ""))

                st.markdown("---")

                # Save button
                if st.button("💾 Save Meal to My Log", type="primary", use_container_width=True):
                    with st.spinner("Saving…"):
                        log_entry = {
                            "food_name":    food_rec.get("display_name", confirmed_food),
                            "food_id":      food_rec.get("food_id", ""),
                            "course":       meal_course.lower(),
                            "portion_grams": float(portion_grams),
                            "portion_unit": unit_choice,
                            "calories_kcal": cal_val,
                            "protein_g":    prot_val,
                            "carbs_g":      carb_val,
                            "fat_g":        fat_val,
                            "fibre_g":      fib_val,
                            "sodium_mg":    sod_val,
                            "notes":        meal_notes,
                        }
                        try:
                            saved = log_meal(active_user["user_id"], log_entry, conn=_db_conn)
                            st.session_state["last_logged_meal"] = saved
                            st.success(
                                f"✅ Logged **{portion_grams:.0f} g of {food_rec['display_name']}** "
                                f"({cal_val:.0f} kcal) — {meal_course}"
                            )
                            st.balloons()
                        except Exception as ex:
                            st.error(f"Save failed: {ex}")

                # Quick feedback block (non-blocking)
                if st.session_state.get("last_logged_meal"):
                    last_meal = st.session_state["last_logged_meal"]
                    with st.expander("📝 Rate This Prediction (optional)", expanded=False):
                        fb_col1, fb_col2 = st.columns(2)
                        with fb_col1:
                            rating = st.slider("Accuracy Rating", 1, 5, 4,
                                               help="How accurate was the food identification?")
                        with fb_col2:
                            fb_note = st.text_input("Comment (optional)", key="fb_note")
                        if st.button("Submit Feedback", key="submit_fb"):
                            try:
                                log_user_feedback(
                                    active_user["user_id"],
                                    {
                                        "log_id":          last_meal.get("log_id", ""),
                                        "rating":          rating,
                                        "feedback_type":   "prediction_accuracy",
                                        "feedback_text":   fb_note,
                                        "predicted_class": infer_predicted_class or confirmed_food,
                                        "corrected_class": confirmed_food,
                                    },
                                    conn=_db_conn,
                                )
                                st.success("Feedback recorded — thank you!")
                            except Exception as ex:
                                st.warning(f"Feedback not saved: {ex}")

            except Exception as ex:
                st.error(f"Nutrition computation error: {ex}")


# ===========================================================================
# SECTION 4 — NUTRITION ANALYSIS
# ===========================================================================
elif selected_section == "🔬 Nutrition Analysis":
    st.title("🔬 Nutrition Database & Ingredient Analysis")
    st.markdown(
        "Explore verified nutritional profiles across 1,500+ laboratory reference items "
        "(IFCT 2017 & ICMR-NIN 2020). Search ingredients and discover safe culinary substitutions."
    )

    tab_explore, tab_sub = st.tabs(["📊 Food Nutrient Explorer", "🔄 Ingredient Substitution Engine"])

    # ── FOOD EXPLORER ────────────────────────────────────────────────────────
    with tab_explore:
        col_srch, col_scale = st.columns([2, 1])
        with col_srch:
            search_query = st.text_input(
                "🔍 Search Food or Ingredient",
                value="dal makhani",
                placeholder="e.g. roti, dal, palak, paneer, almond",
            )
        with col_scale:
            scale_grams = st.number_input(
                "Portion (grams)", min_value=5.0, max_value=1000.0, value=100.0, step=5.0
            )

        if search_query.strip():
            matches = search_foods(search_query.strip(), limit=20)
        else:
            matches = []

        if not search_query.strip():
            st.info("Enter a food name above to search the nutrition database.")
        elif not matches:
            st.warning(f"No results for **{search_query}**. Try alternate spellings (e.g. 'dal' vs 'daal', 'chapati' vs 'roti').")
        else:
            food_opt_map = {
                m["food_name"]: f"{m['display_name']} ({m['match_type']} · {m['calories_per_100g']:.0f} kcal/100g)"
                for m in matches
            }
            selected_food_key = st.selectbox(
                "Select Matched Record",
                list(food_opt_map.keys()),
                format_func=lambda k: food_opt_map[k],
            )

            food_profile = lookup_food(selected_food_key)
            if not food_profile:
                st.error(f"Could not load detailed nutrition record for **{selected_food_key}**.")
            else:
                st.markdown(f"### {food_profile['display_name']}")
                st.caption(
                    f"**Data Source**: {food_profile.get('source', 'N/A')}  ·  "
                    f"**Reference Basis**: {food_profile.get('reference_basis', 'N/A')}"
                )

                scaled = calculate_nutrients(food_profile, float(scale_grams))
                nut    = scaled["nutrients"]

                # Quick macro summary
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Calories",  _fmt_nut(nut.get("calories_kcal", {})))
                sm2.metric("Protein",   _fmt_nut(nut.get("protein_g",    {})))
                sm3.metric("Carbs",     _fmt_nut(nut.get("carbs_g",      {})))
                sm4.metric("Fat",       _fmt_nut(nut.get("fat_g",        {})))

                # Full table
                rows = []
                for k, info in nut.items():
                    rows.append({
                        "Nutrient":               info.get("label", k),
                        f"Amount ({scale_grams:.0f} g)": _fmt_nut(info),
                        "Per 100 g":              _fmt_nut_100g(info),
                        "Status":                 info.get("status", ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if scaled.get("missing_nutrients"):
                    with st.expander("ℹ️ Missing Laboratory Micronutrients Notice"):
                        st.info(
                            "In compliance with scientific data integrity standards, "
                            "missing laboratory measurements remain None — never zero-filled."
                        )
                        for m in scaled["missing_nutrients"]:
                            st.markdown(f"- **{m['label']}**: {m['reason']}")

                st.caption(PHASE7_DISCLAIMER)

                # Quick-log shortcut
                if active_user:
                    if st.button(f"📸 Log {food_profile['display_name']} as a Meal", use_container_width=True):
                        st.session_state["log_meal_food"] = selected_food_key
                        st.session_state["nav_target"]    = "📸 Log a Meal"
                        st.rerun()

    # ── SUBSTITUTION ENGINE ──────────────────────────────────────────────────
    with tab_sub:
        st.subheader("Culinary Ingredient Substitution Finder")
        st.markdown(
            "Find kitchen-tested substitutions for Indian recipes — manage allergies, "
            "lactose intolerance, diabetes, or sodium intake."
        )

        sub_engine = IngredientSubstitutionEngine()
        sub_keys   = sorted(sub_engine.SUBSTITUTIONS_DB.keys())
        if not sub_keys:
            st.warning("Substitution database not available.")
        else:
            sc1, sc2 = st.columns(2)
            with sc1:
                target_ing = st.selectbox("Select Ingredient to Replace", sub_keys)
            with sc2:
                reason_opts  = ["All / Any", "dairy", "vegan", "gluten", "manage_diabetes", "manage_hypertension", "nuts", "peanuts"]
                reason_filter = st.selectbox("Substitution Reason", reason_opts)

            filter_arg = None if reason_filter == "All / Any" else reason_filter
            subs       = sub_engine.get_substitutions_for_ingredient(target_ing, reason_or_allergy=filter_arg)

            if not subs:
                st.info(f"No substitutions found for **{target_ing}** with filter '{reason_filter}'.")
            else:
                for s in subs:
                    with st.expander(f"➡️ {s['substitute']}", expanded=False):
                        st.markdown(f"**Usage Ratio**: `{s.get('ratio', 'N/A')}`")
                        st.markdown(f"**Culinary Reason**: {s.get('reason', 'N/A')}")
                        suitable = s.get("suitable_for", [])
                        if suitable:
                            st.caption(f"Suitable for: {', '.join(suitable)}")


# ===========================================================================
# SECTION 5 — RECOMMENDATIONS & MEAL PLANNING
# ===========================================================================
elif selected_section == "🍽️ Recommendations & Meal Planning":
    st.title("🍽️ Personalized Recipe Recommendations & Meal Planning")
    st.markdown(
        "Safety-first recipe matching powered by content-based TF-IDF filtering and "
        "ICMR-NIN 2020 RDA calorie targets."
    )

    if not active_user:
        st.warning("⚠️ No active user profile selected. Please create or select a profile in the sidebar.")
        st.stop()

    rec_sys = _get_rec_sys()
    tab_rec, tab_daily, tab_weekly = st.tabs(
        ["🍲 Recipe Recommendations", "📅 Daily Meal Plan", "🗓️ 7-Day Weekly Schedule"]
    )

    # ── RECIPE RECOMMENDATIONS ───────────────────────────────────────────────
    with tab_rec:
        st.subheader(f"Top Recipes for {active_user['display_name']}")
        st.caption(
            f"Safety filter active → Allergens: {', '.join(active_user.get('allergies') or ['None'])} · "
            f"Restrictions: {', '.join(active_user.get('dietary_restrictions') or ['None'])}"
        )

        rf1, rf2, rf3 = st.columns(3)
        with rf1:
            course_filter = st.selectbox("Course Filter", ["All Courses", "Breakfast", "Lunch", "Snack", "Dinner"])
        with rf2:
            query_cue = st.text_input("Keyword / Ingredient Hint", placeholder="e.g. paneer, spicy, dal")
        with rf3:
            top_k_count = st.slider("# Recommendations", min_value=3, max_value=15, value=6)

        c_arg = None if course_filter == "All Courses" else course_filter.lower()
        q_arg = query_cue.strip() or None

        if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Filtering recipes by safety & computing suitability scores…"):
                try:
                    recs = rec_sys.recommend_recipes(active_user, course=c_arg, top_n=top_k_count, query_hint=q_arg)
                    st.session_state["_last_recs"] = recs
                except Exception as ex:
                    st.error(f"Recommendation engine error: {ex}")
                    st.session_state["_last_recs"] = []

        recs: List[Dict] = st.session_state.get("_last_recs", [])

        if not recs and "_last_recs" in st.session_state:
            st.warning("No recipes matched your constraints. Try relaxing course, keyword, or cooking-time filters in your profile.")
        elif recs:
            st.success(f"Showing {len(recs)} safety-verified recipe(s).")
            for r in recs:
                score_lbl = f"{r['suitability_score']:.0f}/100 — {r['verdict'].replace('_', ' ').title()}"
                label = f"⭐ {r['recipe_name']}  ·  {score_lbl}"
                with st.expander(label, expanded=False):
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.markdown(f"**Cuisine**: {r.get('cuisine', 'N/A')}  \n**Course**: {r.get('course', 'N/A').title()}")
                    rc2.markdown(f"**Diet**: {r.get('diet', 'N/A')}  \n**Complexity**: {r.get('complexity_level', 'N/A').title()}")
                    total_time = _safe_float(r.get("total_time_mins"), 0)
                    rc3.markdown(f"**Cook Time**: {total_time:.0f} mins")

                    st.info(f"**Why recommended**: {r.get('explanation', 'N/A')}")

                    comp_scores = r.get("component_scores") or {}
                    if comp_scores:
                        st.markdown("**Score Breakdown**")
                        comp_df = pd.DataFrame([
                            {"Component": k.replace("_", " ").title(), "Points": round(float(v), 1)}
                            for k, v in comp_scores.items()
                        ])
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)

                    ing = r.get("ingredients_raw", "")
                    if ing:
                        st.markdown(f"**Ingredients**: {ing}")

                    if r.get("has_incomplete_ingredients"):
                        st.caption("⚠️ Some ingredient data is incomplete — allergen screening may be partial.")

    # ── DAILY MEAL PLAN ──────────────────────────────────────────────────────
    with tab_daily:
        st.subheader("Calibrated 1-Day Meal Plan")
        st.markdown("4-meal allocation: **Breakfast 25% · Lunch 35% · Snack 10% · Dinner 30%**")

        # Compute calorie target from profile
        default_cal = _calorie_target_from_profile(active_user)
        d_cal_target = st.number_input(
            "Target Daily Calories (kcal)",
            min_value=1200.0, max_value=4500.0,
            value=float(default_cal),
            step=50.0,
        )

        if st.button("🔄 Generate Daily Plan", type="primary", use_container_width=True):
            with st.spinner("Generating safety-verified daily plan…"):
                try:
                    d_plan = rec_sys.generate_daily_meal_plan(active_user, target_calories_kcal=float(d_cal_target))
                    st.session_state["current_daily_plan"] = d_plan
                except Exception as ex:
                    st.error(f"Daily plan error: {ex}")
                    st.session_state["current_daily_plan"] = None

        d_plan = st.session_state.get("current_daily_plan")
        if d_plan:
            avg_score = _safe_float(d_plan.get("average_suitability_score"), 0)
            st.metric("Average Suitability Score", f"{avg_score:.1f} / 100")

            meal_names = ["breakfast", "lunch", "snack", "dinner"]
            m_cols     = st.columns(len(meal_names))
            for idx, m_key in enumerate(meal_names):
                m_item = d_plan.get("meals", {}).get(m_key, {})
                with m_cols[idx]:
                    st.markdown(f"### {m_key.title()}")
                    st.markdown(f"**{m_item.get('recipe_name', 'N/A')}**")
                    tgt_cal   = _safe_float(m_item.get("target_calories_kcal"))
                    tot_time  = _safe_float(m_item.get("total_time_mins"))
                    suit_sc   = _safe_float(m_item.get("suitability_score"))
                    st.caption(f"🎯 ~{tgt_cal:.0f} kcal  ·  ⏱️ {tot_time:.0f} min")
                    st.markdown(f"⭐ `{suit_sc:.0f}/100` — {m_item.get('verdict', '').replace('_',' ').title()}")
                    cuisine = m_item.get("cuisine", "")
                    if cuisine:
                        st.caption(f"🍴 {cuisine}")
                    # Recipe details
                    ing = m_item.get("ingredients_raw", "")
                    if ing:
                        with st.expander("Ingredients"):
                            st.markdown(ing)

            st.caption(d_plan.get("safety_notice", ""))
            st.caption(d_plan.get("disclaimer", ""))

            if st.button("💾 Save Daily Plan to Database", key="save_daily"):
                try:
                    save_meal_plan(active_user["user_id"], d_plan, plan_type="daily", conn=_db_conn)
                    st.success("✅ Daily plan saved to your profile!")
                except Exception as ex:
                    st.error(f"Save failed: {ex}")

    # ── WEEKLY SCHEDULE ──────────────────────────────────────────────────────
    with tab_weekly:
        st.subheader("7-Day Weekly Meal Schedule & Grocery List")
        st.markdown("Varied schedule with zero consecutive-day dish repetitions guaranteed.")

        if st.button("🗓️ Generate 7-Day Plan", type="primary", use_container_width=True):
            with st.spinner("Generating 7-day schedule and aggregating grocery list…"):
                try:
                    w_plan = rec_sys.generate_weekly_meal_plan(active_user)
                    st.session_state["current_weekly_plan"] = w_plan
                except Exception as ex:
                    st.error(f"Weekly plan error: {ex}")
                    st.session_state["current_weekly_plan"] = None

        w_plan = st.session_state.get("current_weekly_plan")
        if w_plan:
            overall_sc = _safe_float(w_plan.get("overall_weekly_suitability_score"), 0)
            st.metric("Overall Weekly Suitability Score", f"{overall_sc:.1f} / 100")

            for day_name, d_data in (w_plan.get("weekly_schedule") or {}).items():
                day_sc = _safe_float(d_data.get("day_suitability_score"), 0)
                with st.expander(f"📅 {day_name}  —  Score: {day_sc:.0f}/100", expanded=False):
                    wc1, wc2, wc3, wc4 = st.columns(4)
                    for col, slot in zip([wc1, wc2, wc3, wc4], ["breakfast", "lunch", "snack", "dinner"]):
                        item = d_data.get(slot, {})
                        col.markdown(f"**{slot.title()}**")
                        col.markdown(item.get("recipe_name", "N/A"))
                        col.caption(
                            f"⭐ {_safe_float(item.get('suitability_score')):.0f}/100  "
                            f"⏱️ {_safe_float(item.get('total_time_mins')):.0f} min"
                        )

            st.markdown("---")
            st.markdown("#### Aggregated Weekly Grocery Shopping List")
            st.caption("Top staple ingredients required for all 28 weekly meals:")
            grocery = w_plan.get("grocery_shopping_list", [])
            if grocery:
                g_df = pd.DataFrame(grocery)
                g_df.columns = [c.replace("_", " ").title() for c in g_df.columns]
                st.dataframe(g_df, use_container_width=True, hide_index=True)
            else:
                st.info("Grocery list not available for this plan.")

            st.caption(w_plan.get("safety_notice", ""))

            if st.button("💾 Save Weekly Plan to Database", key="save_weekly"):
                try:
                    save_meal_plan(active_user["user_id"], w_plan, plan_type="weekly", conn=_db_conn)
                    st.success("✅ Weekly plan and grocery list saved!")
                except Exception as ex:
                    st.error(f"Save failed: {ex}")


# ===========================================================================
# SECTION 6 — NUTRITION DASHBOARD
# ===========================================================================
elif selected_section == "📊 Nutrition Dashboard":
    st.title("📊 Nutrition Dashboard & Meal History")
    st.markdown(
        "Review your recorded meals, track calorie and macro targets, "
        "and identify potential nutrient gaps."
    )

    if not active_user:
        st.warning("⚠️ No active user profile selected. Please select a profile in the sidebar.")
        st.stop()

    uid = active_user["user_id"]

    # Date selector
    view_date = st.date_input("Select Date", value=date.today())
    date_str  = view_date.isoformat()

    daily_totals = get_daily_nutrition_totals(uid, target_date=date_str, conn=_db_conn)
    tot = daily_totals.get("totals", {})
    tgt = daily_totals.get("targets", {})

    # ── Daily Summary Metrics ────────────────────────────────────────────────
    st.subheader(f"Summary — {view_date.strftime('%B %d, %Y')}")

    cal_consumed  = _safe_float(tot.get("calories_kcal"))
    cal_target    = _safe_float(tgt.get("calories_kcal"), 2000.0)
    cal_remaining = _safe_float(tgt.get("remaining_calories_kcal"), cal_target - cal_consumed)
    pct_achieved  = _safe_float(tgt.get("percent_target_achieved"), 0.0)

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Calories Consumed",  f"{cal_consumed:.0f} kcal",  f"{pct_achieved:.1f}% of daily target")
    dc2.metric("Calorie Target",     f"{cal_target:.0f} kcal",   f"Remaining: {cal_remaining:.0f} kcal")
    dc3.metric("Protein Consumed",   f"{_safe_float(tot.get('protein_g')):.1f} g")
    dc4.metric("Dietary Fibre",      f"{_safe_float(tot.get('fibre_g')):.1f} g")

    # Alerts
    for alert in daily_totals.get("alerts", []):
        if alert.get("severity") == "WARNING":
            st.error(f"⚠️ {alert['message']}")
        else:
            st.info(f"ℹ️ {alert['message']}")

    # ── Macro Distribution ───────────────────────────────────────────────────
    st.markdown("#### Macronutrient Distribution")
    prot_g = _safe_float(tot.get("protein_g"))
    carb_g = _safe_float(tot.get("carbs_g"))
    fat_g  = _safe_float(tot.get("fat_g"))
    total_macro_cals = max(prot_g * 4 + carb_g * 4 + fat_g * 9, 1.0)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "Protein Energy",
        f"{prot_g * 4 / total_macro_cals * 100:.1f}%",
        f"{prot_g:.1f} g (target: 15–25%)",
    )
    mc2.metric(
        "Carbohydrates Energy",
        f"{carb_g * 4 / total_macro_cals * 100:.1f}%",
        f"{carb_g:.1f} g (target: 50–60%)",
    )
    mc3.metric(
        "Fat Energy",
        f"{fat_g * 9 / total_macro_cals * 100:.1f}%",
        f"{fat_g:.1f} g (target: 20–30%)",
    )

    # Calorie progress bar
    pct_clamped = min(pct_achieved / 100.0, 1.0)
    st.markdown(f"**Calorie Progress** — {pct_achieved:.1f}% of {cal_target:.0f} kcal target")
    st.progress(pct_clamped)

    st.markdown("---")

    # ── Meal Log Table ───────────────────────────────────────────────────────
    st.subheader(f"Meals Recorded on {date_str}")
    meals = daily_totals.get("meals", [])

    if not meals:
        st.info("No meals logged for this date. Use **📸 Log a Meal** to record what you eat.")
    else:
        # Build display table (fibre_g may not be present in daily-totals meals)
        meal_rows = []
        for m in meals:
            row = {
                "Time":      m.get("meal_time", ""),
                "Course":    m.get("course", "").title(),
                "Food Item": m.get("food_name", ""),
                "Portion":   f"{_safe_float(m.get('portion_grams')):.0f} g",
                "Calories":  f"{_safe_float(m.get('calories_kcal')):.0f} kcal",
                "Protein":   f"{_safe_float(m.get('protein_g')):.1f} g",
                "Carbs":     f"{_safe_float(m.get('carbs_g')):.1f} g",
                "Fat":       f"{_safe_float(m.get('fat_g')):.1f} g",
                "_log_id":   m.get("log_id", ""),
            }
            if "fibre_g" in m:
                row["Fibre"] = f"{_safe_float(m.get('fibre_g')):.1f} g"
            meal_rows.append(row)

        display_df = pd.DataFrame(meal_rows).drop(columns=["_log_id"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Delete a meal log entry
        st.markdown("##### 🗑️ Remove a Meal Entry")
        del_options = {m["log_id"]: f"{m.get('meal_time', '')} — {m.get('food_name', '')} ({m.get('course', '')})" for m in meals}
        del_choice  = st.selectbox("Select entry to remove", options=list(del_options.keys()), format_func=lambda x: del_options[x])

        confirm_key = f"confirm_del_{del_choice}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        if not st.session_state[confirm_key]:
            if st.button("🗑️ Remove Selected Entry", type="secondary"):
                st.session_state[confirm_key] = True
                st.rerun()
        else:
            st.warning(f"Remove **{del_options[del_choice]}**?")
            conf1, conf2 = st.columns(2)
            if conf1.button("✅ Confirm Delete", key=f"del_yes_{del_choice}"):
                try:
                    delete_meal_log(del_choice, uid, conn=_db_conn)
                    st.session_state[confirm_key] = False
                    st.success("Entry removed.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Delete failed: {ex}")
            if conf2.button("❌ Cancel", key=f"del_no_{del_choice}"):
                st.session_state[confirm_key] = False
                st.rerun()

    st.markdown("---")

    # ── 7-Day Trend ──────────────────────────────────────────────────────────
    st.subheader("📈 7-Day Trend & Nutrient-Gap Insights")

    summary_7d = get_period_nutrition_summary(uid, days=7, conn=_db_conn)
    active_days = int(summary_7d.get("active_logged_days", 0))

    if active_days == 0:
        st.info("Log meals over multiple days to unlock multi-day trends and nutrient-gap insights.")
    else:
        breakdown = summary_7d.get("daily_breakdown", [])
        averages  = summary_7d.get("averages", {})

        if breakdown:
            trend_df = pd.DataFrame(breakdown).rename(columns={
                "meal_date":   "Date",
                "day_calories": "Calories (kcal)",
                "day_protein":  "Protein (g)",
                "day_carbs":    "Carbs (g)",
                "day_fat":      "Fat (g)",
                "day_fibre":    "Fibre (g)",
            })
            trend_df = trend_df.set_index("Date")

            chart_tab1, chart_tab2 = st.tabs(["Calories", "Macros"])
            with chart_tab1:
                st.markdown(f"**Daily Calories — {active_days}-day active window**")
                st.bar_chart(trend_df[["Calories (kcal)"]], use_container_width=True)
            with chart_tab2:
                macro_cols = [c for c in ["Protein (g)", "Carbs (g)", "Fat (g)", "Fibre (g)"] if c in trend_df.columns]
                if macro_cols:
                    st.markdown("**Daily Macronutrients**")
                    st.line_chart(trend_df[macro_cols], use_container_width=True)

        # Averages table
        if averages:
            st.markdown(f"**{active_days}-Day Averages**")
            avg_df = pd.DataFrame([{
                "Metric":    k.replace("_", " ").title(),
                "Daily Avg": f"{_safe_float(v):.1f} {'kcal' if 'calorie' in k else 'g'}",
            } for k, v in averages.items()])
            st.dataframe(avg_df, use_container_width=True, hide_index=True)

        # Nutrient gap insights (only shown when enough data)
        if active_days >= 2:
            st.markdown("#### 🔬 Clinical Nutrient-Gap Insights")
            avg_fibre  = _safe_float(averages.get("fibre_g"), 0.0)
            avg_prot   = _safe_float(averages.get("protein_g"), 0.0)
            avg_cals   = _safe_float(averages.get("calories_kcal"), 0.0)

            g1, g2 = st.columns(2)
            with g1:
                if avg_fibre < 25.0:
                    st.warning(
                        f"🌾 **Fiber Deficit** — {active_days}-day average: `{avg_fibre:.1f} g/day`  \n"
                        f"ICMR recommends ≥30 g. Consider sprouted pulses, oats, and leafy greens."
                    )
                else:
                    st.success(f"🌾 **Optimal Fibre** — {active_days}-day average: `{avg_fibre:.1f} g/day` ✓")

            with g2:
                if avg_prot < 50.0:
                    st.warning(
                        f"💪 **Protein Gap** — {active_days}-day average: `{avg_prot:.1f} g/day`  \n"
                        f"Consider moong dal, paneer, tofu, or eggs to support satiety and lean mass."
                    )
                else:
                    st.success(f"💪 **Sufficient Protein** — {active_days}-day average: `{avg_prot:.1f} g/day` ✓")

            if cal_target > 0:
                cal_ratio = avg_cals / cal_target
                if cal_ratio < 0.80:
                    st.warning(
                        f"⚡ **Calorie Under-Target** — {active_days}-day average: `{avg_cals:.0f} kcal/day` "
                        f"({cal_ratio*100:.0f}% of your {cal_target:.0f} kcal target). "
                        f"Sustained under-eating may reduce metabolic rate."
                    )
                elif cal_ratio > 1.20:
                    st.warning(
                        f"⚡ **Calorie Over-Target** — {active_days}-day average: `{avg_cals:.0f} kcal/day` "
                        f"({cal_ratio*100:.0f}% of your {cal_target:.0f} kcal target). "
                        f"Consider reviewing portion sizes."
                    )
                else:
                    st.success(
                        f"⚡ **On-Target** — {active_days}-day calorie average: `{avg_cals:.0f} kcal/day` "
                        f"({cal_ratio*100:.0f}% of target) ✓"
                    )

        elif active_days == 1:
            st.info("Log at least **2 days** of meals to unlock full nutrient-gap analysis.")

    # ── Full Meal History ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Full Meal History")
    with st.expander("View All Logged Meals (last 50)", expanded=False):
        try:
            full_history = get_meal_history(uid, limit=50, conn=_db_conn)
            if not full_history:
                st.info("No meal history yet.")
            else:
                hist_rows = []
                for m in full_history:
                    hist_rows.append({
                        "Date":      m.get("meal_date", ""),
                        "Time":      m.get("meal_time", ""),
                        "Course":    m.get("course", "").title(),
                        "Food":      m.get("food_name", ""),
                        "Portion":   f"{_safe_float(m.get('portion_grams')):.0f} g",
                        "Calories":  f"{_safe_float(m.get('calories_kcal')):.0f} kcal",
                        "Protein":   f"{_safe_float(m.get('protein_g')):.1f} g",
                        "Carbs":     f"{_safe_float(m.get('carbs_g')):.1f} g",
                        "Fat":       f"{_safe_float(m.get('fat_g')):.1f} g",
                        "Notes":     m.get("notes", ""),
                    })
                st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        except Exception as ex:
            st.error(f"Could not load meal history: {ex}")
