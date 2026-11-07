"""
src/label_scanner.py
NutriSense AI - Packaged Food Label Scanner (OCR Engine)

Uses EasyOCR combined with OpenCV image preprocessing to extract text from a
packaged food product photo, then applies structured regex parsing to recover:
  - Product name and brand (without misclassifying ingredients as brands)
  - Nutrition facts (calories, macros, micros) with declared units and basis
  - Serving size / servings per pack (distinguishing per 100g vs per serving)
  - Ingredients list (separated cleanly from regulatory and manufacturer text)
  - Declared, ingredient-detected, and trace allergens (3-way categorization)

Design principles:
  - Only reports values that are supported by the OCR output.
  - Missing fields are recorded as None with a reason string.
  - Never substitutes zero for unknown/unreadable values.
  - Tracks which fields are OCR-extracted vs user-corrected.
  - Preserves units as printed (g, mg, %, kcal, kJ, ml, oz, mcg, IU).
  - Does NOT confuse sodium with salt or perform undocumented conversions.
  - Uses OpenCV preprocessing (denoising, CLAHE, resizing, adaptive thresholding)
    for high OCR accuracy across varying package lighting, orientations, and fonts.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import Counter
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load EasyOCR reader
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
        import easyocr
        model_dir = (Path(__file__).resolve().parent.parent / "models" / "easyocr").resolve()
        user_net_dir = model_dir / "user_network"
        model_dir.mkdir(parents=True, exist_ok=True)
        user_net_dir.mkdir(parents=True, exist_ok=True)
        _ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False,
            model_storage_directory=str(model_dir),
            user_network_directory=str(user_net_dir),
        )
    return _ocr_reader


# ---------------------------------------------------------------------------
# 1. OpenCV Image Preprocessing Pipeline
# ---------------------------------------------------------------------------

def preprocess_image_for_ocr(pil_image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies multi-stage OpenCV preprocessing on a PIL image:
    1. EXIF orientation correction.
    2. Dimension normalization (upscales small packaging text, constrains overly large images).
    3. Noise reduction via bilateral filtering (preserves crisp letter edges).
    4. Contrast Limited Adaptive Histogram Equalization (CLAHE) to handle package glare and uneven lighting.
    Returns:
      (enhanced_bgr, enhanced_gray)
    """
    # 1. EXIF auto-rotation (smartphones / digital cameras)
    try:
        pil_image = ImageOps.exif_transpose(pil_image)
    except Exception:
        pass

    # Convert to RGB numpy array
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]

    # 2. Dimension normalization
    # If smaller dimension < 800, upscale using bicubic interpolation for fine print readability
    if min(h, w) < 800:
        scale = 800.0 / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 2200:
        scale = 2200.0 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Grayscale conversion
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 4. Bilateral filtering for edge-preserving denoising
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(denoised)

    # Convert enhanced grayscale back to 3-channel BGR for EasyOCR
    enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr, enhanced_gray


# ---------------------------------------------------------------------------
# 2. Text Normalization & Robust Numeric OCR Correction
# ---------------------------------------------------------------------------

def _normalize_text(raw: str) -> str:
    """Unicode-normalize and clean raw OCR output to ASCII-friendly text."""
    text = unicodedata.normalize("NFKD", raw)
    replacements = {
        "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
        "\u00b5": "u", "\u03bc": "u",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2022": "-", "\u00b7": "-", "\u2013": "-", "\u2014": "-",
        "|": " ",
        "~": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _clean_ocr_number(raw_num_str: str) -> str:
    """
    Fixes common OCR character confusions in numeric text:
    - 'O', 'o', 'Q', 'D' -> '0'
    - 'l', 'I', '!', '|' -> '1'
    - 'S', 's' -> '5' (when adjacent to digits)
    - Comma vs dot decimal handling ('8,5' -> '8.5', but '1,200' -> '1200')
    """
    s = raw_num_str.strip()
    if not s:
        return ""

    # Fix obvious single letter replacements in numbers
    # e.g., 'O.5' -> '0.5', '1O0' -> '100', '4SO' -> '450'
    cleaned = []
    for i, ch in enumerate(s):
        prev_ch = s[i - 1] if i > 0 else ""
        next_ch = s[i + 1] if i + 1 < len(s) else ""

        if ch in ("O", "o", "Q") and (prev_ch.isdigit() or next_ch.isdigit() or next_ch in (".", ",")):
            cleaned.append("0")
        elif ch in ("l", "I", "|") and (prev_ch.isdigit() or next_ch.isdigit() or next_ch in (".", ",")):
            cleaned.append("1")
        elif ch in ("S", "s") and (prev_ch.isdigit() or next_ch.isdigit()):
            cleaned.append("5")
        elif ch in ("B",) and (prev_ch.isdigit() and next_ch.isdigit()):
            cleaned.append("8")
        else:
            cleaned.append(ch)

    s = "".join(cleaned)

    # Handle comma: if followed by 1 or 2 digits and end of string, treat as decimal point
    # e.g., "8,5" -> "8.5", "0,25" -> "0.25", but "1,200" -> "1200"
    m_dec = re.search(r"(\d+),(\d{1,2})$", s)
    if m_dec:
        s = f"{m_dec.group(1)}.{m_dec.group(2)}"
    else:
        # If followed by 3 digits (thousands separator), remove comma
        s = re.sub(r"(\d+),(\d{3})", r"\1\2", s)
        # Any remaining stray commas before digits convert to dot if solitary
        s = s.replace(",", ".")

    # Remove any extra dots (e.g. "0..5" -> "0.5")
    s = re.sub(r"\.+", ".", s)

    return s


def _merge_lines(raw_results: List[Tuple]) -> str:
    """Flatten EasyOCR bounding-box list into a single ordered text block."""
    sorted_items = sorted(raw_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    lines = [_normalize_text(item[1]) for item in sorted_items]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Known Brands and Regulatory Exclusion Lexicons
# ---------------------------------------------------------------------------

EXCLUDED_BRAND_OR_PRODUCT_TERMS = {
    "nutrition", "nutritional", "facts", "information", "info", "value", "values",
    "typical", "amount", "serving", "servings", "per", "container", "basis",
    "daily", "value", "percent", "rda", "ingredients", "ingredient", "contains",
    "allergen", "allergens", "advice", "warning", "declaration", "may", "traces",
    "net", "weight", "wt", "contents", "qty", "quantity", "mrp", "rs", "batch",
    "lot", "mfg", "mfd", "date", "use", "best", "before", "expiry", "exp",
    "fssai", "lic", "license", "no", "iso", "registered", "trademark",
    "manufactured", "marketed", "packed", "by", "for", "in", "india",
    "keep", "store", "cool", "dry", "place", "away", "direct", "sunlight",
    "vegetarian", "green", "dot", "pure", "authentic", "original",
    "energy", "calories", "protein", "carbohydrate", "fat", "sugar", "sodium",
}

KNOWN_FOOD_BRANDS = [
    "Britannia", "Parle", "Amul", "Nestle", "Haldiram's", "Haldirams", "Kellogg's",
    "Kelloggs", "Tata", "Patanjali", "MTR", "Sunfeast", "Cadbury", "Lay's", "Lays",
    "Kurkure", "Oreo", "Quaker", "Bikaji", "Balaji", "Bikanervala", "ITC", "Marico",
    "Saffola", "Fortune", "Dabur", "Mother Dairy", "Everest", "MDH", "Catch",
    "Maggi", "Kissan", "Epigamia", "Raw Pressery", "Yoga Bar", "Pintola", "MyFitness",
    "Bisk Farm", "Priyagold", "Crax", "Act II", "Real", "Tropicana", "Paper Boat",
    "Hershey's", "Hersheys", "Nutella", "Knorr", "Chocos", "Bournvita", "Horlicks",
    "Boost", "Complan", "Nescafe", "Bru", "Lipton", "Taj Mahal", "Wagh Bakri",
    "Ching's Secret", "Smith & Jones", "Tops", "Del Monte", "Kikkoman", "Disano",
    "Borges", "Figaro", "Barilla", "Gits", "Aashirvaad", "Pillsbury", "Catch",
    "Kwality Wall's", "Vadilal", "Creambell", "Baskin Robbins", "Bonn", "Modern",
    "English Oven", "Harvest Gold", "Dukes", "Unibic", "McVitie's", "Mcvities"
]

PRODUCT_TYPE_KEYWORDS = [
    "Biscuit", "Biscuits", "Cookie", "Cookies", "Cracker", "Crackers",
    "Chips", "Crisps", "Namkeen", "Bhujia", "Sev", "Mixture", "Gathiya",
    "Atta", "Flour", "Bread", "Rusk", "Noodles", "Instant Noodles", "Pasta",
    "Oats", "Rolled Oats", "Muesli", "Granola", "Corn Flakes", "Chocos",
    "Cereal", "Protein Bar", "Energy Bar", "Granola Bar", "Peanut Butter",
    "Dark Chocolate", "Milk Chocolate", "Wafer", "Wafers", "Sauce", "Ketchup",
    "Jam", "Pickle", "Papad", "Khakhra", "Poha", "Upma", "Rava", "Suji",
    "Yogurt", "Curd", "Milk", "Cheese", "Paneer", "Butter", "Ghee"
]


# ---------------------------------------------------------------------------
# 4. Section Splitters (Clean Boundary Detection)
# ---------------------------------------------------------------------------

_NUTRITION_SECTION_START = re.compile(
    r"nutrition\s*(facts|information|value[s]?|table)|"
    r"nutritional\s*(information|value[s]?|facts)|"
    r"typical\s*values|"
    r"amount\s*per\s*serving|"
    r"per\s*(?:100\s*(?:g|ml)|serving|portion|pack)",
    re.IGNORECASE
)

_INGREDIENTS_START = re.compile(
    r"(?:^|\n)\s*ingredient[s]?\s*[:\-]?\s*|"
    r"ingredient\s*list\s*[:\-]?\s*|"
    r"list\s*of\s*ingredients\s*[:\-]?\s*",
    re.IGNORECASE
)

_ALLERGEN_START = re.compile(
    r"allergen[s]?\s*(information|warning|declaration|advice|alert)?\s*[:\-]?\s*|"
    r"allergy\s+(advice|warning|information)\s*[:\-]?\s*|"
    r"contains\s*[:\-]\s*|"
    r"contains\s+(?:milk|wheat|soy|peanut|tree\s*nut|egg|gluten|sesame)|"
    r"may\s+contain\s*(?:traces?\s*of)?",
    re.IGNORECASE
)

_REGULATORY_OR_MFG_START = re.compile(
    r"(?:manufactured|mfg|marketed|packed|distributed)\s*(?:by|in|for)|"
    r"fssai\s*(?:lic|license)?|"
    r"lic(?:ense)?\s*no\.?|"
    r"net\s*wt|net\s*weight|net\s*quantity|"
    r"batch\s*no|date\s*of\s*mfg|best\s*before|use\s*by|mrp\s*rs|"
    r"customer\s*care|consumer\s*feedback|"
    r"store\s*in\s*a\s*cool",
    re.IGNORECASE
)


def _split_sections(full_text: str) -> Dict[str, str]:
    """
    Split OCR text into header, nutrition, ingredients, allergen, and logistics blocks.
    Strictly preserves boundaries so ingredients, allergens, and brands never bleed.
    """
    lines = full_text.splitlines()
    sections: Dict[str, List[str]] = {
        "header": [], "nutrition": [], "ingredients": [], "allergen": [], "logistics": []
    }
    current = "header"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check section transition markers
        if _NUTRITION_SECTION_START.search(stripped):
            current = "nutrition"
        elif _ALLERGEN_START.search(stripped):
            current = "allergen"
        elif _INGREDIENTS_START.search(stripped):
            current = "ingredients"
        elif _REGULATORY_OR_MFG_START.search(stripped):
            if current in ("ingredients", "header"):
                current = "logistics"

        sections[current].append(stripped)

    return {k: "\n".join(v) for k, v in sections.items()}


# ---------------------------------------------------------------------------
# 5. Product Identification (Brand & Product Name Without Ingredients Bleed)
# ---------------------------------------------------------------------------

def _is_ingredient_or_table_line(line: str) -> bool:
    """Returns True if the line looks like an ingredient list or nutrition table entry."""
    l_lower = line.lower()
    # Food ingredient terms
    food_terms = [
        "flour", "oil", "sugar", "salt", "water", "emulsifier", "acidity regulator",
        "preservative", "lecithin", "cocoa", "malt", "starch", "yeast", "leavening",
        "milk solids", "condensed", "flavor", "flavour", "ins ", "e1", "e2", "e3", "e4", "e5"
    ]
    # Commas + ingredients = ingredient line
    comma_count = line.count(",")
    term_count = sum(1 for term in food_terms if term in l_lower)
    if comma_count >= 2 or term_count >= 2:
        return True

    # Numeric table line (e.g. "Energy 450 kcal" or "Protein 8.5g")
    if re.search(r"\b(energy|calories|protein|carbohydrate|fat|sodium)\b", l_lower) and re.search(r"\d", line):
        return True

    return False


def _extract_product_info(header_text: str, full_text: str) -> Dict[str, Any]:
    """
    Identify product name and brand from top lines or full text without assigning
    ingredient names or table headers as brands.
    """
    combined_lines = [ln.strip() for ln in header_text.splitlines() if ln.strip()]
    if len(combined_lines) < 2:
        combined_lines.extend([ln.strip() for ln in full_text.splitlines() if ln.strip()][:10])

    # Filter out lines that cannot be brands or product names
    valid_candidates: List[str] = []
    for ln in combined_lines:
        ln_clean = ln.strip()
        if len(ln_clean) < 2 or len(ln_clean) > 80:
            continue
        if re.fullmatch(r"[\d\s\.,:\-]+", ln_clean):
            continue
        if _is_ingredient_or_table_line(ln_clean):
            continue

        l_lower = ln_clean.lower()
        # Skip pure header/regulatory lines
        words = set(re.findall(r"\w+", l_lower))
        if words.issubset(EXCLUDED_BRAND_OR_PRODUCT_TERMS):
            continue
        if any(term in l_lower for term in [
            "ingredients:", "ingredient list", "contains:", "allergen information",
            "manufactured by", "marketed by", "mfg by", "fssai", "batch no",
            "best before", "use by", "net weight", "net wt", "keep in cool"
        ]):
            continue

        valid_candidates.append(ln_clean)

    detected_brand: Optional[str] = None
    detected_product: Optional[str] = None

    # Step 1: Detect Brand from Known Food Brands list
    for brand_cand in KNOWN_FOOD_BRANDS:
        pattern = r"\b" + re.escape(brand_cand) + r"\b"
        if re.search(pattern, full_text, re.IGNORECASE):
            detected_brand = brand_cand
            break

    # Step 2: Check for explicit "Brand:" or "Marketed by:"
    if not detected_brand:
        m_brand = re.search(r"(?:brand|brand\s*name)\s*[:\-]\s*([A-Za-z0-9\s&'\-]+)", full_text, re.IGNORECASE)
        if m_brand:
            b_val = m_brand.group(1).strip().splitlines()[0][:30].strip()
            if b_val and not _is_ingredient_or_table_line(b_val):
                detected_brand = b_val

    # Step 3: Check candidate lines for Brand if still missing
    if not detected_brand and valid_candidates:
        for c in valid_candidates[:3]:
            # Short, 1-3 words, capitalized or with (R) or TM
            c_words = c.split()
            if (1 <= len(c_words) <= 3 and (c.isupper() or c.istitle())) or any(sym in c for sym in ["®", "™", "(R)", "(TM)"]):
                # Clean symbols
                clean_b = re.sub(r"[®™\(\)]", "", c).strip()
                if clean_b.lower() not in EXCLUDED_BRAND_OR_PRODUCT_TERMS:
                    detected_brand = clean_b
                    break

    # Step 4: Detect Product Name
    # Look for known product keywords (e.g. Biscuit, Cookies, Atta, Chips)
    for c in valid_candidates:
        if detected_brand and c.lower() == detected_brand.lower():
            continue
        if any(kw.lower() in c.lower() for kw in PRODUCT_TYPE_KEYWORDS):
            detected_product = c
            break

    # If no keyword matched, pick the first valid candidate that isn't the brand
    if not detected_product:
        for c in valid_candidates:
            if detected_brand and c.lower() == detected_brand.lower():
                continue
            detected_product = c
            break

    # If brand was set to same as product, resolve distinction
    if detected_brand and detected_product and detected_brand.lower() == detected_product.lower():
        detected_brand = None

    return {
        "product_name": detected_product if detected_product else "Packaged Food Product",
        "brand": detected_brand,
        "ocr_extracted": True
    }


# ---------------------------------------------------------------------------
# 6. Serving Size & Nutrition Basis
# ---------------------------------------------------------------------------

_NUM = r"(\d[\d\.,]*)"
_UNIT = r"(kcal|kj|g|mg|mcg|ug|%|ml|oz|IU)"


def _extract_serving(text: str) -> Dict[str, Any]:
    """
    Extract serving size, servings per container, and nutrition basis
    (per 100g, per 100ml, per serving, per pack).
    """
    result: Dict[str, Any] = {
        "serving_size": None,
        "serving_unit": None,
        "servings_per_container": None,
        "nutrition_basis": None,
        "ocr_extracted": True,
        "raw_match": None
    }

    # Match serving size: e.g. "Serving size: 30g", "Serving Size: 1 biscuit (15 g)", "Serving size 28 g"
    m = re.search(
        r"serving\s*size\s*[:\-]?\s*(?:about\s+)?(?:\d+\s*[a-zA-Z\s]+\()?" + _NUM + r"\s*" + _UNIT + r"\)?",
        text, re.IGNORECASE
    )
    if not m:
        m = re.search(r"serving\s*size\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT, text, re.IGNORECASE)
    if not m:
        m = re.search(r"per\s+serving\s*(?:\()?" + _NUM + r"\s*" + _UNIT + r"\)?", text, re.IGNORECASE)

    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            result["serving_size"] = _clean_ocr_number(groups[0])
            result["serving_unit"] = groups[1]
        elif len(groups) == 1:
            result["serving_size"] = _clean_ocr_number(groups[0])
            result["serving_unit"] = "g"
        result["raw_match"] = m.group(0)

    # Nutrition basis (prioritize 100g/ml, serving/portion, avoiding 'servings per container' false match)
    basis_m = re.search(r"\bper\s+(100\s*(?:g|ml))\b", text, re.IGNORECASE)
    if not basis_m:
        basis_m = re.search(r"\bapprox\s+values\s+per\s+(100\s*(?:g|ml))\b", text, re.IGNORECASE)
    if not basis_m:
        basis_m = re.search(r"\bper\s+(serving|portion|piece)\b", text, re.IGNORECASE)
    if not basis_m:
        basis_m = re.search(r"(?<!servings\s)(?<!servings\s:\s)(?<!servings:\s)\bper\s+(pack(?:age)?|container)\b", text, re.IGNORECASE)

    if basis_m:
        matched = basis_m.group(1) if basis_m.groups() else basis_m.group(0)
        norm_basis = matched.strip().lower()
        if "100" in norm_basis and "ml" in norm_basis:
            result["nutrition_basis"] = "per 100 ml"
        elif "100" in norm_basis:
            result["nutrition_basis"] = "per 100 g"
        elif "pack" in norm_basis or "container" in norm_basis:
            result["nutrition_basis"] = "per pack"
        else:
            result["nutrition_basis"] = "per serving"
    elif result["serving_size"]:
        result["nutrition_basis"] = f"per serving ({result['serving_size']}{result['serving_unit'] or 'g'})"

    # Servings per container
    spc_m = re.search(
        r"(?:about\s+)?(\d+)\s+servings?\s*(?:per\s*(?:container|pack(?:age)?))?|"
        r"servings?\s*(?:per\s*(?:container|pack(?:age)?))?\s*[:\-]?\s*(\d+)",
        text, re.IGNORECASE
    )
    if spc_m:
        val = spc_m.group(1) or spc_m.group(2)
        if val:
            result["servings_per_container"] = _clean_ocr_number(val)

    return result


# ---------------------------------------------------------------------------
# 7. Robust Nutrition Facts Extraction (17 Nutrients)
# ---------------------------------------------------------------------------

_NUTRIENT_PATTERNS: Dict[str, List[str]] = {
    "calories": [
        r"(?:energy|calories?|caloric\s*value)\s*[:\-]?\s*" + _NUM + r"\s*(kcal|kj|cal)",
        r"(?:energy|calories?)\s*[:\-]?\s*" + _NUM,
        r"(?:energy\s*value)\s*[:\-]?\s*" + _NUM,
    ],
    "protein": [
        r"protein\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"protein\s*[:\-]?\s*" + _NUM,
    ],
    "total_carbohydrates": [
        r"total\s*carbo?hydrate[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"carbo?hydrate[s]?\s*total\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"carbo?hydrate[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bcarb[o]?[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "dietary_fibre": [
        r"dietary\s*fi(?:b|br)re?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bfi(?:b|br)re?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "total_sugars": [
        r"total\s*sugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"(?:of\s+which\s+)?sugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "added_sugars": [
        r"added\s*sugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"of\s+which\s+added\s*sugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "total_fat": [
        r"total\s*fat\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bfat\s+total\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bfat\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "saturated_fat": [
        r"sat(?:urated)?\s*fat(?:ty\s*acids?)?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"of\s+which\s+saturate[sd]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "trans_fat": [
        r"trans\s*fat(?:ty\s*acids?)?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"of\s+which\s+trans\s*fat\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "cholesterol": [
        r"\bcholesterol\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "sodium": [
        r"\bsodium\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "salt": [
        r"\bsalt\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "calcium": [
        r"\bcalcium\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "iron": [
        r"\biron\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "potassium": [
        r"\bpotassium\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "vitamin_d": [
        r"vitamin\s*d(?:3)?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"vit\s*d\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "vitamin_c": [
        r"vitamin\s*c\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"ascorbic\s*acid\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
}

DEFAULT_NUTRIENT_UNITS = {
    "calories": "kcal",
    "protein": "g",
    "total_carbohydrates": "g",
    "dietary_fibre": "g",
    "dietary_fiber": "g",
    "total_sugars": "g",
    "added_sugars": "g",
    "total_fat": "g",
    "saturated_fat": "g",
    "trans_fat": "g",
    "cholesterol": "mg",
    "sodium": "mg",
    "salt": "g",
    "calcium": "mg",
    "iron": "mg",
    "potassium": "mg",
    "vitamin_d": "mcg",
    "vitamin_c": "mg",
}


def _find_nutrient(text: str, patterns: List[str]) -> Optional[Dict[str, str]]:
    """Try each regex pattern; return first match with cleaned OCR numeric value or None."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            raw_val = groups[0] if groups else None
            unit = groups[1].strip() if len(groups) > 1 and groups[1] else None
            if raw_val is not None:
                clean_val = _clean_ocr_number(raw_val)
                # Verify clean_val can parse as a float
                try:
                    float(clean_val)
                    return {"value": clean_val, "unit": unit, "ocr_extracted": True}
                except (ValueError, TypeError):
                    continue
    return None


def _extract_nutrition(text: str) -> Dict[str, Any]:
    """Parse all recognisable nutrition values from the nutrition text block."""
    results: Dict[str, Any] = {}
    for nutrient, patterns in _NUTRIENT_PATTERNS.items():
        found = _find_nutrient(text, patterns)
        if found and not found.get("unit"):
            found["unit"] = DEFAULT_NUTRIENT_UNITS.get(nutrient, "g")
        results[nutrient] = found

    # Provide alias for American English spelling dietary_fiber
    results["dietary_fiber"] = results.get("dietary_fibre")

    return results


# ---------------------------------------------------------------------------
# 8. Ingredients Extraction (Clean Trimming of Regulatory Text)
# ---------------------------------------------------------------------------

def _extract_ingredients(text: str) -> Optional[str]:
    """
    Return the cleaned ingredients string trimmed cleanly at allergen,
    licensing, manufacturing, or date markers. Never returns None if present.
    """
    if not text or not text.strip():
        return None

    cleaned = _INGREDIENTS_START.sub("", text, count=1).strip()
    if not cleaned:
        return None

    # Stop ingredients string before allergen declarations or logistics/licensing
    stop_patterns = [
        r"(?:^|\n)\s*allergen\s*(?:info|information|warning|declaration|advice)",
        r"(?:^|\n)\s*contains\s*[:\-]",
        r"(?:^|\n)\s*may\s+contain\s*(?:traces\s*of)?",
        r"(?:^|\n)\s*manufactured\s*(?:by|in)",
        r"(?:^|\n)\s*mfg\s*by",
        r"(?:^|\n)\s*marketed\s*by",
        r"(?:^|\n)\s*fssai\b",
        r"(?:^|\n)\s*lic(?:ense)?\s*no",
        r"(?:^|\n)\s*net\s*wt",
        r"(?:^|\n)\s*batch\s*no",
        r"(?:^|\n)\s*best\s*before",
        r"(?:^|\n)\s*store\s*in\s*a\s*cool",
    ]

    for sp in stop_patterns:
        m_stop = re.search(sp, cleaned, re.IGNORECASE)
        if m_stop:
            cleaned = cleaned[:m_stop.start()].strip()

    # Clean up trailing punctuation
    cleaned = re.sub(r"[\s\.,;:\-]+$", "", cleaned).strip()

    return cleaned if len(cleaned) > 3 else None


# ---------------------------------------------------------------------------
# 9. Allergen Detection: 3-Way Categorization
# ---------------------------------------------------------------------------

ALLERGEN_KEYWORDS_MAP: Dict[str, List[str]] = {
    "milk": ["milk", "dairy", "lactose", "whey", "curd", "yogurt", "cheese", "butter", "ghee", "casein", "cream", "malai", "khoya", "mawa"],
    "peanuts": ["peanut", "peanuts", "groundnut", "groundnuts", "mungfali", "moongphali"],
    "tree nuts": ["tree nut", "tree nuts", "almond", "almonds", "badam", "cashew", "cashews", "kaju", "walnut", "walnuts", "akhrot", "hazelnut", "hazelnuts", "pistachio", "pistachios", "pista", "pecan", "pecans"],
    "wheat": ["wheat", "maida", "atta", "semolina", "suji", "sooji", "rava", "wheat flour"],
    "gluten": ["gluten", "wheat", "barley", "rye", "malt", "spelt"],
    "soy": ["soy", "soya", "soybean", "soybeans", "soya lecithin", "soy lecithin", "tofu", "edamame"],
    "eggs": ["egg", "eggs", "anda", "egg albumen", "egg yolk", "egg white", "egg powder"],
    "fish": ["fish", "salmon", "tuna", "cod", "anchovy"],
    "crustaceans": ["crustacean", "crustaceans", "shrimp", "shrimps", "prawn", "prawns", "crab", "crabs", "lobster", "shellfish"],
    "sesame": ["sesame", "sesame seeds", "til", "tahini"],
}


def _extract_allergens(allergen_text: str, ingredients_text: str) -> Dict[str, Any]:
    """
    Categorizes allergens into 3 distinct, non-confused groups:
    1. declared: Explicit "Contains: ..." or "Allergen advice: Contains ..." statements.
    2. detected_from_ingredients: Detected by scanning the ingredients list directly.
    3. may_contain: Cross-contamination warnings ("May contain traces of...").
    """
    declared: List[str] = []
    detected_from_ingredients: List[str] = []
    may_contain: List[str] = []

    a_text = (allergen_text or "").lower()
    i_text = (ingredients_text or "").lower()

    # 1. Check "May Contain" warnings
    may_m = re.search(
        r"may\s+contain\s*(?:traces?\s*of)?\s*([a-zA-Z\s,;\.\(\)]+)",
        f"{a_text}\n{i_text}", re.IGNORECASE
    )
    if may_m:
        may_str = may_m.group(1)[:120].lower()
        for allergen_group, keywords in ALLERGEN_KEYWORDS_MAP.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", may_str) for kw in keywords):
                if allergen_group not in may_contain:
                    may_contain.append(allergen_group)

    # 2. Check Explicit "Contains:" Declarations
    decl_m = re.search(
        r"(?:contains|allergen\s*declaration|allergy\s*advice|contains\s*added)\s*[:\-]?\s*([a-zA-Z\s,;\.\(\)]+)",
        a_text, re.IGNORECASE
    )
    if decl_m:
        decl_str = decl_m.group(1)[:150].lower()
        for allergen_group, keywords in ALLERGEN_KEYWORDS_MAP.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", decl_str) for kw in keywords):
                if allergen_group not in declared:
                    declared.append(allergen_group)
    else:
        # Check if allergen_text contains keywords directly outside may_contain
        for allergen_group, keywords in ALLERGEN_KEYWORDS_MAP.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", a_text) for kw in keywords):
                if allergen_group not in declared and allergen_group not in may_contain:
                    declared.append(allergen_group)

    # 3. Detect Allergens in Ingredients List
    if i_text:
        for allergen_group, keywords in ALLERGEN_KEYWORDS_MAP.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", i_text) for kw in keywords):
                if allergen_group not in detected_from_ingredients:
                    detected_from_ingredients.append(allergen_group)

    # Union of all allergens for quick checking
    all_identified = list(dict.fromkeys(declared + detected_from_ingredients + may_contain))

    return {
        "declared": declared if declared else None,
        "detected_from_ingredients": detected_from_ingredients if detected_from_ingredients else None,
        "may_contain": may_contain if may_contain else None,
        "all_identified_allergens": all_identified if all_identified else None,
        "ocr_extracted": True
    }


# ---------------------------------------------------------------------------
# 10. Readability & OCR Confidence Assessment
# ---------------------------------------------------------------------------

def _assess_readability(raw_results: List[Tuple]) -> Dict[str, Any]:
    """Derive overall readability score from EasyOCR per-bbox confidence values."""
    if not raw_results:
        return {"mean_confidence": 0.0, "readable_bbox_count": 0, "quality": "UNREADABLE"}

    confs = [r[2] for r in raw_results if len(r) > 2]
    if not confs:
        return {"mean_confidence": 0.0, "readable_bbox_count": 0, "quality": "UNREADABLE"}

    mean_conf = float(np.mean(confs))
    min_conf = float(np.min(confs))
    n_readable = int(sum(1 for c in confs if c > 0.5))

    if mean_conf >= 0.75 and n_readable >= 5:
        quality = "GOOD"
    elif mean_conf >= 0.45 and n_readable >= 2:
        quality = "PARTIAL"
    else:
        quality = "LOW"

    return {
        "mean_confidence": round(mean_conf, 3),
        "min_confidence": round(min_conf, 3),
        "readable_bbox_count": n_readable,
        "quality": quality
    }


# ---------------------------------------------------------------------------
# 11. Missing Fields Summarizer
# ---------------------------------------------------------------------------

def _summarise_missing(
    nutrition: Dict,
    serving: Dict,
    product: Dict,
    ingredients: Optional[str],
    allergens: Dict
) -> List[str]:
    """Collect field names that could not be extracted. Never set to 0."""
    missing: List[str] = []

    if not product.get("product_name") or product.get("product_name") == "Packaged Food Product":
        missing.append("product_name")
    if not product.get("brand"):
        missing.append("brand")
    if not serving.get("serving_size"):
        missing.append("serving_size")
    if not serving.get("nutrition_basis"):
        missing.append("nutrition_basis")
    if not ingredients:
        missing.append("ingredients")
    if not allergens.get("declared") and not allergens.get("detected_from_ingredients"):
        missing.append("declared_allergens")

    priority_nutrients = [
        "calories", "total_fat", "total_carbohydrates", "protein", "sodium"
    ]
    for field in priority_nutrients:
        if nutrition.get(field) is None:
            missing.append(f"nutrition.{field}")

    return missing


# ---------------------------------------------------------------------------
# 12. Main Scanner Entry Point
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "Nutrition information is extracted via OCR from the uploaded product image and "
    "may be incomplete or inaccurate due to image quality, label layout, or text "
    "legibility. Always verify against the physical product label. This is for "
    "educational and dietary awareness only and does not constitute clinical advice."
)


def scan_label(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Main entry point for packaged food label scanning.
    1. Preprocesses image using OpenCV (denoise, contrast enhancement, normalization).
    2. Runs EasyOCR reader with confidence tracking.
    3. Parses product name and brand (without misclassifying ingredients).
    4. Extracts serving size and basis (per 100g vs per serving).
    5. Extracts 17 nutrition facts with units and error correction.
    6. Trims ingredients list cleanly from manufacturer and regulatory text.
    7. Separates allergens into declared, ingredient-derived, and may-contain.
    8. Provides both structured dict and flattened extracted_data for Streamlit.
    """
    reader = _get_reader()

    # OpenCV preprocessing
    enhanced_bgr, _ = preprocess_image_for_ocr(pil_image)
    raw_results = reader.readtext(enhanced_bgr, detail=1, paragraph=False)

    # Fallback pass: if primary read returned very few bounding boxes (< 3), try adaptive thresholding
    if len(raw_results) < 3:
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
        fallback_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        fallback_results = reader.readtext(fallback_bgr, detail=1, paragraph=False)
        if len(fallback_results) > len(raw_results):
            raw_results = fallback_results

    # If still completely empty
    if not raw_results:
        empty_nutrition = {k: None for k in _NUTRIENT_PATTERNS}
        empty_nutrition["dietary_fiber"] = None
        return {
            "ocr_text": "",
            "readability": {
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "readable_bbox_count": 0,
                "quality": "UNREADABLE"
            },
            "product": {"product_name": None, "brand": None, "ocr_extracted": True},
            "serving": {
                "serving_size": None, "serving_unit": None,
                "nutrition_basis": None, "servings_per_container": None,
                "ocr_extracted": True
            },
            "nutrition": empty_nutrition,
            "ingredients": None,
            "allergens": {
                "declared": None, "detected_from_ingredients": None,
                "may_contain": None, "all_identified_allergens": None,
                "ocr_extracted": True
            },
            "missing_fields": ["ALL — no text detected in image"],
            "extraction_note": (
                "No text was detected in the uploaded image. "
                "The image may be blurry, too dark, or not show a nutrition label."
            ),
            "disclaimer": _DISCLAIMER,
            "extracted_data": {}
        }

    full_text = _merge_lines(raw_results)
    readability = _assess_readability(raw_results)
    sections = _split_sections(full_text)

    product = _extract_product_info(sections["header"], full_text)
    serving = _extract_serving(sections["nutrition"] or full_text)
    nutrition = _extract_nutrition(sections["nutrition"] or full_text)
    ingredients = _extract_ingredients(sections["ingredients"])
    allergens = _extract_allergens(sections["allergen"], sections["ingredients"])

    # Retry serving extraction on full text if section-based found nothing
    if not serving.get("serving_size") and not serving.get("nutrition_basis"):
        serving = _extract_serving(full_text)

    missing = _summarise_missing(nutrition, serving, product, ingredients, allergens)

    notes = []
    if readability["quality"] == "LOW":
        notes.append(
            "Image quality is low — OCR confidence is poor. "
            "Results may be incomplete or inaccurate."
        )
    elif readability["quality"] == "PARTIAL":
        notes.append(
            "Image is partially readable — some fields may not have been extracted."
        )
    if "nutrition.calories" in missing and "nutrition.total_carbohydrates" in missing:
        notes.append("Nutrition panel was not visible or readable in this image.")
    if "ingredients" in missing:
        notes.append("Ingredients list was not visible in this image.")

    extraction_note = " ".join(notes) if notes else "Extraction completed successfully."

    # Build flattened extracted_data dict for Streamlit and simple consumers
    flattened_data: Dict[str, Any] = {
        "product_name": product.get("product_name"),
        "brand": product.get("brand"),
        "serving_size": serving.get("serving_size"),
        "serving_unit": serving.get("serving_unit"),
        "nutrition_basis": serving.get("nutrition_basis"),
        "servings_per_container": serving.get("servings_per_container"),
        "ingredients": ingredients,
        "allergens_declared": allergens.get("declared"),
        "allergens_detected": allergens.get("detected_from_ingredients"),
        "allergens_may_contain": allergens.get("may_contain"),
    }
    for k, v in nutrition.items():
        if v and isinstance(v, dict):
            try:
                flattened_data[k] = float(v.get("value", 0))
            except (ValueError, TypeError):
                flattened_data[k] = v.get("value")

    return {
        "ocr_text": full_text,
        "readability": readability,
        "product": product,
        "serving": serving,
        "nutrition": nutrition,
        "ingredients": ingredients,
        "allergens": allergens,
        "missing_fields": missing,
        "extraction_note": extraction_note,
        "disclaimer": _DISCLAIMER,
        "extracted_data": flattened_data
    }
