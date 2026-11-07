"""
src/ingredient_intelligence.py
NutriSense AI — Smart Ingredient Intelligence Engine

Authoritative, scientifically-grounded ingredient interpretation engine for packaged foods.
Complies with FSSAI, ICMR-NIN 2020 RDA, Codex Alimentarius, and WHO guidelines.

Capabilities:
1. Structured Ingredient Extraction & Parsing (handles complex packaging syntax, nested brackets, percentages).
2. Comprehensive INS (International Numbering System) Additive Resolution (colours, preservatives, emulsifiers, flavor enhancers).
3. Detection of Disguised / Hidden Sugars, Refined Fats, Hydrogenated Oils, and Non-Salt Sodium Sources.
4. Ingredient Functional Profiling (Category, Manufacturing Purpose, Nutritional Relevance, Scientifically Grounded Concerns, Regulatory Safety Status).
5. Personalized Health Profile Matching (Hypertension, Diabetes, PCOS, Celiac/Gluten-Free, Allergies, Dyslipidemia).
6. Objective Product Benefits & Positives Extraction.
7. Culturally Grounded, Affordable Healthier Alternatives Engine.
"""

from collections import OrderedDict
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. INS Additive Database (Codex Alimentarius & FSSAI Permitted Additives)
# ---------------------------------------------------------------------------

INS_DATABASE: Dict[str, Dict[str, Any]] = {
    # Colours
    "100": {
        "name": "Curcumin",
        "category": "Natural Food Colour",
        "purpose": "Provides vibrant yellow-gold colour extracted from turmeric rhizomes.",
        "health_relevance": "Natural polyphenol with well-documented antioxidant and mild anti-inflammatory properties.",
        "concerns": "Extremely safe in food-grade quantities.",
        "safety_status": "Generally Recognized as Safe (Natural Colour)",
        "profile_flags": []
    },
    "102": {
        "name": "Tartrazine",
        "category": "Synthetic Food Colour (Yellow 5)",
        "purpose": "Imparts a stable bright lemon-yellow colour resistant to light and heat.",
        "health_relevance": "Synthetic azo dye permitted within strict maximum statutory limits by FSSAI and FDA.",
        "concerns": "May trigger mild pseudo-allergic reactions or exacerbate symptoms in aspirin-sensitive or asthmatic individuals.",
        "safety_status": "Permitted with Statutory Quantitative Limits",
        "profile_flags": ["asthma_sensitivity"]
    },
    "110": {
        "name": "Sunset Yellow FCF",
        "category": "Synthetic Food Colour (Yellow 6)",
        "purpose": "Imparts an attractive sunset orange-yellow shade to confectioneries, snacks, and beverages.",
        "health_relevance": "Synthetic petroleum-derived azo dye with established Acceptable Daily Intake (ADI 0-4 mg/kg bw).",
        "concerns": "Regulated under strict permissible limits. Potential sensitivity in rare hypersensitive individuals.",
        "safety_status": "Permitted with Statutory Quantitative Limits",
        "profile_flags": []
    },
    "122": {
        "name": "Azorubine / Carmoisine",
        "category": "Synthetic Food Colour (Red)",
        "purpose": "Provides a distinctive red-to-maroon hue in sweets, jams, and confectioneries.",
        "health_relevance": "Synthetic azo dye permitted under specific regional food standards (ADI 0-4 mg/kg bw).",
        "concerns": "Requires quantitative labeling under FSSAI. Not recommended in excessive quantities.",
        "safety_status": "Permitted with Statutory Quantitative Limits",
        "profile_flags": []
    },
    "129": {
        "name": "Allura Red AC",
        "category": "Synthetic Food Colour (Red 40)",
        "purpose": "Provides a vibrant, stable red colour resistant to light and acidity.",
        "health_relevance": "Synthetic azo colour permitted widely across global standards within maximum limits.",
        "concerns": "Potential mild reactivity in sensitive individuals; strict maximum use limits apply.",
        "safety_status": "Permitted with Statutory Quantitative Limits",
        "profile_flags": []
    },
    "133": {
        "name": "Brilliant Blue FCF",
        "category": "Synthetic Food Colour (Blue 1)",
        "purpose": "Provides a bright blue hue or combines with yellow to produce vivid green shades.",
        "health_relevance": "Triarylmethane dye poorly absorbed by the digestive tract (<5% absorption).",
        "concerns": "Safe within established ADI (0-6 mg/kg bw); strictly limited in dairy products.",
        "safety_status": "Permitted with Statutory Quantitative Limits",
        "profile_flags": []
    },
    "150a": {
        "name": "Plain Caramel (Caramel I)",
        "category": "Food Colour",
        "purpose": "Provides pale yellow to dark brown colour via controlled heating of edible carbohydrates.",
        "health_relevance": "Traditional natural caramelization without ammonia or sulphite reactants.",
        "concerns": "Considered the cleanest form of caramel colouring.",
        "safety_status": "Generally Recognized as Safe (GRAS)",
        "profile_flags": []
    },
    "150c": {
        "name": "Ammonia Caramel (Caramel III)",
        "category": "Food Colour",
        "purpose": "Provides reddish-brown colour in baked goods, soy sauces, and confectioneries.",
        "health_relevance": "Prepared using food-grade ammonium compounds to stabilize colour in specific pH ranges.",
        "concerns": "Contains trace amounts of 4-methylimidazole (4-MEI), regulated under rigorous maximum safety limits.",
        "safety_status": "Permitted Food Additive",
        "profile_flags": []
    },
    "150d": {
        "name": "Sulphite Ammonia Caramel (Caramel IV)",
        "category": "Food Colour",
        "purpose": "Imparts deep brown hue and acid-stability in carbonated soft drinks and dark sauces.",
        "health_relevance": "Manufactured using ammonium and sulphite compounds; by far the most widely used caramel color globally.",
        "concerns": "Contains trace 4-MEI maintained strictly below statutory safety thresholds. High dietary consumption often correlates with ultra-processed sugary drinks.",
        "safety_status": "Permitted Food Additive",
        "profile_flags": []
    },
    "160a": {
        "name": "Beta-Carotene",
        "category": "Natural Food Colour & Provitamin",
        "purpose": "Provides yellow to orange colour and acts as a precursor to Vitamin A.",
        "health_relevance": "Beneficial plant pigment with antioxidant and provitamin A activity.",
        "concerns": "Extremely safe in food applications.",
        "safety_status": "Safe & Nutritive",
        "profile_flags": []
    },

    # Preservatives & Antioxidants
    "200": {
        "name": "Sorbic Acid",
        "category": "Antimicrobial Preservative",
        "purpose": "Inhibits mold, yeast, and fungal growth to extend product freshness.",
        "health_relevance": "Naturally occurring organic acid metabolized like a fatty acid into water and CO2 in the human body.",
        "concerns": "Extremely low toxicity profile; safe at permitted levels.",
        "safety_status": "Generally Recognized as Safe (GRAS)",
        "profile_flags": []
    },
    "202": {
        "name": "Potassium Sorbate",
        "category": "Antimicrobial Preservative",
        "purpose": "Prevents mold and yeast spoilage in cheeses, dips, yogurts, and baked goods.",
        "health_relevance": "Salt of sorbic acid with broad antimicrobial efficacy; fully metabolized by normal bodily enzymes.",
        "concerns": "Very low allergy potential; safe within permitted limits.",
        "safety_status": "Generally Recognized as Safe (GRAS)",
        "profile_flags": []
    },
    "211": {
        "name": "Sodium Benzoate",
        "category": "Preservative & Sodium Source",
        "purpose": "Prevents bacterial and fungal growth in acidic foods, fruit juices, condiments, and carbonated drinks.",
        "health_relevance": "Contributes dietary sodium (~16% sodium by molecular weight). Metabolized in the liver into hippuric acid and excreted.",
        "concerns": "Adds to overall dietary sodium load. When combined with Vitamin C (ascorbic acid) under heat/light, trace benzene formation can occur, though commercial levels are heavily monitored.",
        "safety_status": "Permitted Preservative (Limit Intake for Hypertension)",
        "profile_flags": ["hypertension"]
    },
    "220": {
        "name": "Sulphur Dioxide",
        "category": "Antioxidant & Preservative",
        "purpose": "Prevents oxidative browning and preserves appearance in dried fruits and fruit pulps.",
        "health_relevance": "Permitted chemical preservative with mandatory allergen declaration thresholds (>10 mg/kg).",
        "concerns": "Known respiratory irritant and allergen for individuals with asthma or sulphite sensitivity.",
        "safety_status": "Permitted with Mandatory Allergen Warning",
        "profile_flags": ["asthma_sensitivity"]
    },
    "223": {
        "name": "Sodium Metabisulphite",
        "category": "Antioxidant, Preservative & Sodium Source",
        "purpose": "Acts as an antioxidant, dough conditioner, and bleaching/preservative agent in biscuits and starches.",
        "health_relevance": "Releases sulphur dioxide in acidic environments; contributes trace sodium.",
        "concerns": "Must be flagged for sulphite-sensitive individuals and asthmatics.",
        "safety_status": "Permitted Additive",
        "profile_flags": ["asthma_sensitivity", "hypertension"]
    },
    "250": {
        "name": "Sodium Nitrite",
        "category": "Curing Agent, Preservative & Sodium Source",
        "purpose": "Inhibits Clostridium botulinum spores and preserves pink colour in cured meats.",
        "health_relevance": "Contributes sodium; regulated under very strict parts-per-million limits.",
        "concerns": "High intake of processed meats containing nitrites is classified by WHO IARC as Group 1 carcinogenic due to potential nitrosamine formation during high-heat cooking.",
        "safety_status": "Strictly Regulated Preservative",
        "profile_flags": ["cardiovascular", "hypertension"]
    },
    "282": {
        "name": "Calcium Propionate",
        "category": "Antimicrobial Preservative",
        "purpose": "Prevents mold and 'rope' bacterial growth in commercial bread and bakery products.",
        "health_relevance": "Propionate is a short-chain fatty acid naturally produced in the gut during fiber fermentation. Supplies dietary calcium.",
        "concerns": "Non-toxic; safe for general consumption within permitted limits.",
        "safety_status": "Generally Recognized as Safe (GRAS)",
        "profile_flags": []
    },
    "300": {
        "name": "Ascorbic Acid (Vitamin C)",
        "category": "Antioxidant & Essential Vitamin",
        "purpose": "Prevents oxidation, preserves colour, and strengthens gluten structure in dough.",
        "health_relevance": "Essential water-soluble micronutrient and powerful cellular antioxidant.",
        "concerns": "Completely safe and nutritionally beneficial.",
        "safety_status": "Safe & Nutritive",
        "profile_flags": []
    },
    "319": {
        "name": "TBHQ (Tertiary Butylhydroquinone)",
        "category": "Synthetic Antioxidant",
        "purpose": "Prevents oxidative rancidity in vegetable oils, frying fats, and high-fat packaged snacks.",
        "health_relevance": "Highly effective fat stabilizer allowing long shelf-life without requiring refrigeration.",
        "concerns": "Strictly regulated ADI (0-0.7 mg/kg bw). While permitted by FSSAI and FDA, European food authorities apply lower acceptable thresholds. Marker of ultra-processed fried foods.",
        "safety_status": "Permitted with Strict Quantitative Limits",
        "profile_flags": ["cardiovascular"]
    },
    "320": {
        "name": "BHA (Butylated Hydroxyanisole)",
        "category": "Synthetic Antioxidant",
        "purpose": "Slows oxidation and preserves flavour in fats, oils, and dry snack seasonings.",
        "health_relevance": "Synthetic phenolic antioxidant with strict statutory inclusion limits (max 200 ppm).",
        "concerns": "Classified by some health bodies as a potential endocrine disruptor at high experimental doses; strictly limited in food use.",
        "safety_status": "Permitted with Strict Quantitative Limits",
        "profile_flags": []
    },
    "321": {
        "name": "BHT (Butylated Hydroxytoluene)",
        "category": "Synthetic Antioxidant",
        "purpose": "Prevents fat degradation and maintains crispness in packaging and oily snack foods.",
        "health_relevance": "Fat-soluble synthetic antioxidant often paired synergistically with BHA.",
        "concerns": "Strict ADI (0-0.3 mg/kg bw). Best avoided in large quantities.",
        "safety_status": "Permitted with Strict Quantitative Limits",
        "profile_flags": []
    },
    "322": {
        "name": "Lecithin (Soy / Sunflower)",
        "category": "Natural Emulsifier",
        "purpose": "Blends water and fat phases seamlessly; reduces viscosity in chocolate and bakery goods.",
        "health_relevance": "Naturally derived phospholipid rich in dietary choline. Typically sourced from non-GMO soy or sunflower.",
        "concerns": "Soy-derived lecithin requires allergen declaration for individuals with severe soy allergy.",
        "safety_status": "Generally Recognized as Safe (Natural Emulsifier)",
        "profile_flags": ["allergen_soy"]
    },
    "330": {
        "name": "Citric Acid",
        "category": "Acidity Regulator & Antioxidant Synergist",
        "purpose": "Provides natural tartness, balances pH, and enhances antioxidant efficacy.",
        "health_relevance": "Natural intermediate in the human Krebs energy cycle; non-toxic organic acid.",
        "concerns": "Completely safe at culinary and industrial levels.",
        "safety_status": "Generally Recognized as Safe (GRAS)",
        "profile_flags": []
    },
    "331": {
        "name": "Sodium Citrate",
        "category": "Acidity Regulator, Emulsifier & Sodium Source",
        "purpose": "Acts as a buffer to stabilize pH, control gel formation, and prevent milk curdling in beverages.",
        "health_relevance": "Supplies sodium (~24% sodium by weight) along with metabolizable citrate.",
        "concerns": "Contributes to overall dietary sodium intake for hypertensive consumers.",
        "safety_status": "Generally Recognized as Safe (Sodium Source)",
        "profile_flags": ["hypertension"]
    },

    # Thickeners, Stabilizers & Emulsifiers
    "407": {
        "name": "Carrageenan",
        "category": "Thickener & Gelling Agent",
        "purpose": "Extracted from red edible seaweeds to thicken, stabilize, and suspend solids in dairy and plant milks.",
        "health_relevance": "Permitted food-grade carrageenan (undegraded polysaccharide) is non-absorbed dietary fiber.",
        "concerns": "Some gastrointestinal research suggests food-grade carrageenan may trigger gut discomfort in sensitive individuals with inflammatory bowel disease.",
        "safety_status": "Permitted Food Additive",
        "profile_flags": ["gut_sensitivity"]
    },
    "412": {
        "name": "Guar Gum",
        "category": "Natural Thickener & Dietary Fiber",
        "purpose": "Extracted from Indian cluster beans (guar); provides viscosity, texture, and moisture retention.",
        "health_relevance": "Soluble prebiotic dietary fiber (galactomannan) that supports beneficial gut bacteria and slows carbohydrate absorption.",
        "concerns": "Extremely safe; high doses may cause temporary mild flatulence due to healthy fermentation.",
        "safety_status": "Generally Recognized as Safe (Soluble Prebiotic Fiber)",
        "profile_flags": []
    },
    "415": {
        "name": "Xanthan Gum",
        "category": "Stabilizer & Thickener",
        "purpose": "Produced by fermentation of sugars; stabilizes emulsions, prevents ingredient separation, and provides elasticity in gluten-free products.",
        "health_relevance": "Soluble dietary fiber with zero caloric absorption; highly effective at minimal doses (<0.5%).",
        "concerns": "Very safe; widely utilized in diabetic and gluten-free diets.",
        "safety_status": "Generally Recognized as Safe (GRAS)",
        "profile_flags": []
    },
    "440": {
        "name": "Pectin",
        "category": "Gelling Agent & Soluble Fiber",
        "purpose": "Extracted from citrus peels and apples; forms firm gels in jams, jellies, and fruit spreads.",
        "health_relevance": "Healthy soluble dietary fiber known to support healthy cholesterol levels and digestive regularity.",
        "concerns": "Completely safe and wholesome.",
        "safety_status": "Safe & Beneficial Soluble Fiber",
        "profile_flags": []
    },
    "471": {
        "name": "Mono- and Di-glycerides of Fatty Acids",
        "category": "Emulsifier & Texture Modifier",
        "purpose": "Binds oil and water to maintain softness, prevent staling, and give ice creams and baked goods a velvety texture.",
        "health_relevance": "Formed from dietary fats and oils; metabolized by normal pancreatic lipases like regular fats.",
        "concerns": "Marker of ultra-processed food formulations; can contain trace trans fats if manufactured from partially hydrogenated oils.",
        "safety_status": "Permitted Food Additive",
        "profile_flags": ["cardiovascular"]
    },
    "472e": {
        "name": "DATEM (Diacetyl Tartaric Acid Esters of Mono- and Diglycerides)",
        "category": "Dough Conditioner & Emulsifier",
        "purpose": "Strengthens gluten network in commercial bread dough, allowing high loaf volume and uniform crumb structure.",
        "health_relevance": "Synthetic lipid ester widely used in commercial industrial bakeries.",
        "concerns": "Permitted under FSSAI and FDA; characteristic marker of industrial factory-made bread.",
        "safety_status": "Permitted Additive",
        "profile_flags": []
    },
    "476": {
        "name": "PGPR (Polyglycerol Polyricinoleate)",
        "category": "Emulsifier",
        "purpose": "Lowers chocolate viscosity during molding, reducing the expensive cocoa butter needed for smooth flow.",
        "health_relevance": "Made from glycerol and condensed castor oil fatty acids. Metabolized normally.",
        "concerns": "Used industrially as a cost-cutting fat replacer in commercial chocolates.",
        "safety_status": "Permitted with Statutory Limits",
        "profile_flags": []
    },

    # Leavening Agents & Sodium Sources
    "500": {
        "name": "Sodium Carbonate / Bicarbonate (Baking Soda)",
        "category": "Leavening Agent, Acidity Regulator & Sodium Source",
        "purpose": "Releases carbon dioxide gas when heated or exposed to acids, causing cakes, biscuits, and batters to rise.",
        "health_relevance": "Significant hidden contributor to dietary sodium (~27% sodium by weight). One teaspoon contains ~1,200 mg sodium.",
        "concerns": "Often overlooked by consumers checking labels for salt. Important consideration for hypertension and heart disease.",
        "safety_status": "Safe Leavening Agent (Count Sodium towards Daily Limit)",
        "profile_flags": ["hypertension"]
    },
    "503": {
        "name": "Ammonium Bicarbonate / Carbonate",
        "category": "Leavening Agent",
        "purpose": "Traditional baker's ammonia used for extra-crisp biscuits, crackers, and cookies; evaporates completely during high-heat baking.",
        "health_relevance": "Does NOT contain sodium; breaks down cleanly into ammonia, carbon dioxide, and water vapor during baking.",
        "concerns": "Safe; leaves no residual sodium in finished baked goods.",
        "safety_status": "Generally Recognized as Safe (Sodium-Free Leavening)",
        "profile_flags": []
    },

    # Flavour Enhancers
    "621": {
        "name": "Monosodium Glutamate (MSG)",
        "category": "Flavour Enhancer & Sodium Source",
        "purpose": "Activates specific umami (savory) taste receptors, intensifying richness in broths, soups, and seasoned snacks.",
        "health_relevance": "Sodium salt of glutamic acid, an amino acid naturally abundant in tomatoes, mushrooms, and aged cheese. Contains ~12% sodium by weight (roughly one-third of table salt).",
        "concerns": "Debunked as a neurotoxin at normal culinary intake by FSSAI, US FDA, and WHO JECFA. However, it still contributes to total dietary sodium load and may trigger transient sensitivity in rare individuals.",
        "safety_status": "Generally Recognized as Safe (Contributes Dietary Sodium)",
        "profile_flags": ["hypertension"]
    },
    "627": {
        "name": "Disodium Guanylate",
        "category": "Flavour Enhancer (Nucleotide) & Sodium Source",
        "purpose": "Acts in powerful synergy with MSG (up to 8x taste enhancement) to deliver intense savory depth.",
        "health_relevance": "Nucleotide additive typically derived from yeast extract or tapioca starch.",
        "concerns": "Metabolizes to purines; individuals with severe gout or uric acid kidney stones should monitor high intake.",
        "safety_status": "Permitted Flavour Enhancer",
        "profile_flags": ["gout_uric_acid", "hypertension"]
    },
    "631": {
        "name": "Disodium Inosinate",
        "category": "Flavour Enhancer (Nucleotide) & Sodium Source",
        "purpose": "Synergistic umami enhancer paired with MSG and guanylate in noodles, chips, and seasonings.",
        "health_relevance": "Purine nucleotide; supplies trace sodium.",
        "concerns": "Metabolized to uric acid; safe for general population within permitted food usage levels.",
        "safety_status": "Permitted Flavour Enhancer",
        "profile_flags": ["gout_uric_acid", "hypertension"]
    },

    # Artificial & Intense Sweeteners
    "950": {
        "name": "Acesulfame Potassium (Ace-K)",
        "category": "Non-Nutritive High-Intensity Sweetener",
        "purpose": "200 times sweeter than sucrose; zero calories; provides clean onset sweetness heat-stable for baking.",
        "health_relevance": "Passes through the human digestive tract unabsorbed and excreted unchanged in urine.",
        "concerns": "Extensively studied and approved by WHO JECFA, US FDA, and FSSAI (ADI 0-15 mg/kg bw). Often blended with sucralose or aspartame to mask bitter aftertaste.",
        "safety_status": "Permitted Artificial Sweetener (Zero Calorie)",
        "profile_flags": ["sweetener_intense"]
    },
    "951": {
        "name": "Aspartame",
        "category": "Low-Calorie High-Intensity Sweetener",
        "purpose": "200 times sweeter than sugar; used in diet colas, sugar-free desserts, and chewing gum.",
        "health_relevance": "Dipeptide methyl ester (aspartic acid + phenylalanine). Yields 4 kcal/g, but used in negligible microgram amounts.",
        "concerns": "Mandatory warning required for individuals with Phenylketonuria (PKU). Classified as 'possibly carcinogenic to humans' (IARC Group 2B) with WHO maintaining ADI of 0-40 mg/kg bw as safe.",
        "safety_status": "Permitted Sweetener (Mandatory PKU Warning)",
        "profile_flags": ["pku_phenylalanine", "sweetener_intense"]
    },
    "954": {
        "name": "Saccharin",
        "category": "Non-Nutritive Artificial Sweetener",
        "purpose": "300 to 400 times sweeter than sucrose; stable at high cooking temperatures.",
        "health_relevance": "Zero calories; zero glycemic impact on blood glucose.",
        "concerns": "Early animal study concerns were dismissed as species-specific; permitted globally within ADI (0-5 mg/kg bw). Has a distinct metallic aftertaste.",
        "safety_status": "Permitted Artificial Sweetener",
        "profile_flags": ["sweetener_intense"]
    },
    "955": {
        "name": "Sucralose",
        "category": "Non-Nutritive High-Intensity Sweetener",
        "purpose": "600 times sweeter than sugar; synthesized by chlorinating sucrose; highly stable in cooking and baking.",
        "health_relevance": "Zero calories; does not trigger an insulin spike or elevate blood glucose.",
        "concerns": "Widely authorized across international standards (ADI 0-15 mg/kg bw). Some recent human gut microbiome studies suggest potential alterations in gut microbiota with chronic high intake.",
        "safety_status": "Permitted Artificial Sweetener (Diabetic Friendly)",
        "profile_flags": ["sweetener_intense"]
    },
    "960": {
        "name": "Steviol Glycosides (Stevia)",
        "category": "Plant-Derived Non-Nutritive Sweetener",
        "purpose": "Natural high-potency sweetening extracted from leaves of Stevia rebaudiana (200-300x sweeter than sugar).",
        "health_relevance": "Plant-derived zero-calorie sweetener with zero glycemic impact. Does not stimulate insulin production.",
        "concerns": "Considered the most natural high-potency sweetener alternative; safe within ADI (0-4 mg/kg bw).",
        "safety_status": "Natural Non-Nutritive Sweetener (Diabetic Friendly)",
        "profile_flags": []
    },
}

# ---------------------------------------------------------------------------
# 2. Comprehensive Ingredient Lexicon & Hidden Ingredient Rules
# ---------------------------------------------------------------------------

# Patterns to match hidden sugar sources
SUGAR_SYNONYMS = [
    r"\bsugar\b", r"\bsucrose\b", r"\bdextrose\b", r"\bglucose\b",
    r"\bliquid\s*glucose\b", r"\bglucose\s*syrup\b", r"\bfructose\b",
    r"\bhigh\s*fructose\b", r"\bhfcs\b", r"\bmaltose\b", r"\bmalt\s*syrup\b",
    r"\bmalt\s*extract\b", r"\bbarley\s*malt\b", r"\binvert\s*sugar\b",
    r"\binvert\s*syrup\b", r"\bcorn\s*syrup\b", r"\bcane\s*sugar\b",
    r"\bcane\s*juice\b", r"\bbrown\s*sugar\b", r"\bcaramel\s*sugar\b",
    r"\bjaggery\b", r"\bgur\b", r"\bhoney\b", r"\bmolasses\b",
    r"\btreacle\b", r"\bagave\b", r"\bmaple\s*syrup\b", r"\bdate\s*syrup\b",
    r"\bdehydrated\s*cane\b", r"\bisoglucose\b"
]

# Patterns for maltodextrin (treated with special glycemic transparency)
MALTODEXTRIN_PATTERNS = [
    r"\bmaltodextrin\b", r"\bdextrin\b", r"\bmalto\-dextrin\b"
]

# Patterns for industrial fats and oils
FATS_AND_OILS = {
    "palm_oil": [
        r"\bpalmolein\b", r"\bpalm\s*oil\b", r"\bpalm\s*kernel\b",
        r"\bfractionated\s*palm\b", r"\brefined\s*palm\b", r"\bred\s*palm\b"
    ],
    "hydrogenated_fat": [
        r"\bhydrogenated\s*vegetable\s*(?:oil|fat)\b",
        r"\bpartially\s*hydrogenated\b",
        r"\bvanaspati\b", r"\bvegetable\s*shortening\b",
        r"\bshortening\b", r"\btrans\s*fat\b", r"\binteresterified\s*fat\b"
    ],
    "refined_vegetable_oils": [
        r"\bsoybean\s*oil\b", r"\bcottonseed\s*oil\b", r"\bsunflower\s*oil\b",
        r"\bcanola\s*oil\b", r"\bcorn\s*oil\b", r"\brice\s*bran\s*oil\b",
        r"\bvegetable\s*oil\b", r"\bedible\s*vegetable\s*oil\b"
    ],
    "traditional_fats": [
        r"\bghee\b", r"\bbutter\b", r"\bmilk\s*fat\b", r"\bcocoa\s*butter\b",
        r"\bmustard\s*oil\b", r"\bcoconut\s*oil\b", r"\bgroundnut\s*oil\b",
        r"\bsesame\s*oil\b", r"\bolive\s*oil\b"
    ]
}

# Patterns for sodium-containing ingredients beyond regular salt
NON_SALT_SODIUM = [
    (r"\bsodium\s*bicarbonate\b|\bbaking\s*soda\b", "Sodium Bicarbonate (Leavening agent; ~27% sodium by weight)"),
    (r"\bmonosodium\s*glutamate\b|\bmsg\b|\bins\s*621\b", "Monosodium Glutamate (MSG - Umami flavor enhancer; ~12% sodium)"),
    (r"\bsodium\s*benzoate\b|\bins\s*211\b", "Sodium Benzoate (Chemical preservative; contributes to total sodium)"),
    (r"\bsodium\s*citrate\b|\bins\s*331\b", "Sodium Citrate (Acidity buffer; ~24% sodium by weight)"),
    (r"\bsodium\s*metabisulphite\b|\bins\s*223\b", "Sodium Metabisulphite (Dough conditioner & preservative; adds sodium)"),
    (r"\bdisodium\s*(?:phosphate|guanylate|inosinate)\b|\bins\s*627\b|\bins\s*631\b", "Disodium Nucleotides / Phosphates (Flavor & texture additives adding sodium)"),
    (r"\bsodium\s*nitrite\b|\bins\s*250\b", "Sodium Nitrite (Curing preservative)")
]

# Refined Grains & Starches
REFINED_GRAINS = [
    (r"\bmaida\b|\brefined\s*wheat\s*flour\b", "Refined Wheat Flour (Maida) - Stripped of bran & germ; fast-digesting starch with low dietary fiber"),
    (r"\bcorn\s*starch\b|\bmaize\s*starch\b", "Corn Starch - Pure isolated starch used as binder; rapid glucose absorption"),
    (r"\bmodified\s*starch\b|\bins\s*1422\b|\bins\s*1442\b", "Modified Starch - Chemically or enzymatically altered starch for freeze-thaw and heat stability"),
    (r"\bwhite\s*flour\b|\bbleached\s*flour\b", "Bleached / White Flour - Highly refined flour")
]

# Whole / Wholesome Grains & Seeds (Positives)
WHOLESOME_INGREDIENTS = [
    (r"\bwhole\s*wheat\b|\batta\b", "Whole Wheat Flour (Atta) - Contains intact bran, germ, and dietary fiber"),
    (r"\boats\b|\boat\s*flakes\b|\brolled\s*oats\b", "Oats - Rich in beta-glucan soluble fiber beneficial for heart and glucose control"),
    (r"\bragi\b|\bfinger\s*millet\b", "Ragi (Finger Millet) - Exceptionally rich in dietary calcium and minerals"),
    (r"\bjowar\b|\bsorghum\b", "Jowar (Sorghum) - Gluten-free ancient whole grain with rich antioxidants"),
    (r"\bbajra\b|\bpearl\s*millet\b", "Bajra (Pearl Millet) - High iron and complex carbohydrate source"),
    (r"\bquinoa\b", "Quinoa - Complete plant protein containing all 9 essential amino acids"),
    (r"\bchia\s*seeds\b|\bflax\s*seeds\b|\balssi\b", "Flax / Chia Seeds - Rich in plant-based Omega-3 (ALA) fatty acids and lignans"),
    (r"\balmonds\b|\bcashews\b|\bwalnuts\b", "Tree Nuts - Wholesome source of monounsaturated fats, protein, and Vitamin E")
]

# ---------------------------------------------------------------------------
# 3. Intelligent Ingredient Parser
# ---------------------------------------------------------------------------

def clean_and_split_ingredients(raw_text: str) -> List[str]:
    """
    Intelligently splits complex ingredients lists respecting nested parentheses,
    brackets, and regulatory prefixes (e.g. 'Contains 2% or less of:').
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()
    # Remove leading 'Ingredients:' or 'Contains:' header
    text = re.sub(r"^(?:ingredients|contains|ingredient\s*list)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)

    # Replace newlines with commas
    text = re.sub(r"[\r\n]+", ", ", text)

    # Tokenize while respecting nested brackets [ ... ] and parentheses ( ... )
    items: List[str] = []
    current: List[str] = []
    paren_depth = 0
    bracket_depth = 0

    for char in text:
        if char == '(':
            paren_depth += 1
            current.append(char)
        elif char == ')':
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
        elif char == '[':
            bracket_depth += 1
            current.append(char)
        elif char == ']':
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
        elif char in (',', ';', '•') and paren_depth == 0 and bracket_depth == 0:
            part = "".join(current).strip()
            if part:
                items.append(part)
            current = []
        else:
            current.append(char)

    if current:
        part = "".join(current).strip()
        if part:
            items.append(part)

    # Clean items and filter out noise
    cleaned: List[str] = []
    for it in items:
        # Strip trailing dots or dashes
        c = re.sub(r"^[\s\.\-–—*]+|[\s\.\-–—*]+$", "", it).strip()
        # Filter regulatory/manufacturing clutter
        if re.search(r"^(?:manufactured|mfg|marketed|fssai|lic|batch|net\s*wt|best\s*before|store\s*in)", c, re.IGNORECASE):
            continue
        if len(c) >= 2:
            cleaned.append(c)

    return cleaned


# ---------------------------------------------------------------------------
# 4. INS Extractor & Matcher
# ---------------------------------------------------------------------------

def extract_ins_numbers(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Identifies all INS / E-numbers inside an ingredient string.
    Matches formats: 'INS 500(ii)', 'INS500', 'E621', '(621)', 'INS 150d', '500(ii)'.
    """
    matches: List[Tuple[str, Dict[str, Any]]] = []
    # Pattern to find INS / E numbers
    pattern = r"(?:INS|E)?\s*([0-9]{3,4}\s*(?:[a-z]|\([iIvVxX0-9]+\))?)"
    found_codes = re.findall(pattern, text, re.IGNORECASE)

    for code in found_codes:
        raw_code = code.strip().lower()
        # Normalize code: remove parentheses and whitespace e.g. "500(ii)" -> "500" or match exact
        base_num = re.match(r"^(\d{3,4})", raw_code)
        num_str = base_num.group(1) if base_num else raw_code

        # Check full key first e.g. "150d" or "500"
        normalized_key = raw_code.replace("(", "").replace(")", "").strip()
        info = INS_DATABASE.get(normalized_key) or INS_DATABASE.get(num_str)

        if info:
            matches.append((code.strip(), info))

    return matches


# ---------------------------------------------------------------------------
# 5. Core Smart Ingredient Analyzer Engine
# ---------------------------------------------------------------------------

class SmartIngredientAnalyzer:
    """
    Intelligent Packaged Food Ingredient Engine that delivers:
      1. Functional categorisation of every ingredient.
      2. Detection of disguised / hidden sugars, fats, and sodium sources.
      3. INS additive deciphering and safety evaluation.
      4. Scientific risk profiling for specific health profiles.
      5. Positive product benefits and wholesome ingredient detection.
      6. Culturally appropriate, affordable healthier alternatives.
    """

    def analyze(self, ingredients_text: str, user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point to perform complete ingredient intelligence analysis.
        """
        if not ingredients_text or not ingredients_text.strip():
            return self._empty_analysis()

        raw_list = clean_and_split_ingredients(ingredients_text)
        if not raw_list:
            return self._empty_analysis()

        structured_ingredients: List[Dict[str, Any]] = []
        hidden_sugars_found: List[str] = []
        maltodextrin_found: List[str] = []
        palm_oil_found: List[str] = []
        hydrogenated_fats_found: List[str] = []
        refined_oils_found: List[str] = []
        hidden_sodium_found: List[Dict[str, str]] = []
        sweeteners_found: List[str] = []
        refined_grains_found: List[str] = []
        positive_wholesome_found: List[str] = []
        ins_additives_found: List[Dict[str, Any]] = []

        # Categories for presentation
        categories_map: Dict[str, List[Dict[str, Any]]] = OrderedDict([
            ("Primary Ingredients", []),
            ("Added Sugars & Sweeteners", []),
            ("Fats & Oils", []),
            ("Preservatives & Antioxidants", []),
            ("Emulsifiers & Stabilizers", []),
            ("Colours & Flavours", []),
            ("Leavening & Sodium Additives", []),
            ("Refined Grains & Starches", []),
            ("Other Additives", [])
        ])

        for idx, item in enumerate(raw_list):
            item_lower = item.lower()
            ing_details = self._inspect_single_ingredient(item, is_primary=(idx < 3))

            # Track hidden sugars
            for s_pat in SUGAR_SYNONYMS:
                if re.search(s_pat, item_lower):
                    hidden_sugars_found.append(item)
                    ing_details["is_hidden_sugar"] = True
                    break

            # Track maltodextrin specifically
            for m_pat in MALTODEXTRIN_PATTERNS:
                if re.search(m_pat, item_lower):
                    maltodextrin_found.append(item)
                    ing_details["is_maltodextrin"] = True
                    break

            # Track palm oil
            for p_pat in FATS_AND_OILS["palm_oil"]:
                if re.search(p_pat, item_lower):
                    palm_oil_found.append(item)
                    ing_details["is_palm_oil"] = True
                    break

            # Track hydrogenated fats
            for h_pat in FATS_AND_OILS["hydrogenated_fat"]:
                if re.search(h_pat, item_lower):
                    hydrogenated_fats_found.append(item)
                    ing_details["is_hydrogenated_fat"] = True
                    break

            # Track refined vegetable oils
            for r_pat in FATS_AND_OILS["refined_vegetable_oils"]:
                if re.search(r_pat, item_lower):
                    refined_oils_found.append(item)
                    break

            # Track non-salt sodium
            for n_pat, n_desc in NON_SALT_SODIUM:
                if re.search(n_pat, item_lower):
                    hidden_sodium_found.append({"ingredient": item, "description": n_desc})
                    ing_details["is_hidden_sodium"] = True
                    break

            # Track refined grains
            for rg_pat, rg_desc in REFINED_GRAINS:
                if re.search(rg_pat, item_lower):
                    refined_grains_found.append(item)
                    ing_details["is_refined_grain"] = True
                    break

            # Track wholesome ingredients
            for w_pat, w_desc in WHOLESOME_INGREDIENTS:
                if re.search(w_pat, item_lower):
                    positive_wholesome_found.append(w_desc)
                    ing_details["is_wholesome"] = True
                    break

            # Track INS additives
            ins_matches = extract_ins_numbers(item)
            if ins_matches:
                for code_str, ins_info in ins_matches:
                    ins_additives_found.append({
                        "code": code_str,
                        "name": ins_info["name"],
                        "category": ins_info["category"],
                        "purpose": ins_info["purpose"],
                        "health_relevance": ins_info["health_relevance"],
                        "concerns": ins_info["concerns"],
                        "safety_status": ins_info["safety_status"]
                    })
                    ing_details["ins_details"] = ins_info

            # Map into display category
            cat_name = ing_details.get("display_category", "Other Additives")
            if cat_name in categories_map:
                categories_map[cat_name].append(ing_details)
            else:
                categories_map["Other Additives"].append(ing_details)

            structured_ingredients.append(ing_details)

        # -------------------------------------------------------------------
        # Build Misleading Names & Hidden Ingredient Alerts
        # -------------------------------------------------------------------
        hidden_insights: List[Dict[str, Any]] = []

        # 1. Sugar Splitting Detection
        unique_sugars = list(dict.fromkeys(hidden_sugars_found))
        if len(unique_sugars) >= 2 or (unique_sugars and maltodextrin_found):
            all_sugars_combined = unique_sugars + [m for m in maltodextrin_found if m not in unique_sugars]
            hidden_insights.append({
                "type": "MULTIPLE_ADDED_SUGARS",
                "severity": "HIGH",
                "title": f"Multiple Sugar Sources Detected ({len(all_sugars_combined)} forms)",
                "description": (
                    f"Label contains {len(all_sugars_combined)} distinct sources of added sweeteners: "
                    f"{', '.join(all_sugars_combined)}. Food manufacturers often split added sugar across multiple names "
                    "(e.g., sugar, liquid glucose, invert syrup, maltodextrin) so that no single sweetener appears as the #1 ingredient by weight."
                ),
                "items": all_sugars_combined,
                "action_advice": "Check the total added sugar grams in the nutrition facts panel rather than relying solely on the ingredient sequence."
            })
        elif len(unique_sugars) == 1:
            hidden_insights.append({
                "type": "ADDED_SUGAR_PRESENT",
                "severity": "MODERATE",
                "title": f"Added Sweetener Declared ({unique_sugars[0]})",
                "description": f"Contains added sugar ({unique_sugars[0]}). Limit overall daily intake per ICMR guidelines (<25g/day).",
                "items": unique_sugars,
                "action_advice": "Ensure portion moderation if managing weight, insulin, or blood glucose."
            })

        # Maltodextrin specific callout
        if maltodextrin_found and not any(h["type"] == "MULTIPLE_ADDED_SUGARS" for h in hidden_insights):
            hidden_insights.append({
                "type": "MALTODEXTRIN_GLYCEMIC_ALERT",
                "severity": "MODERATE",
                "title": "Maltodextrin Present (High Glycemic Index)",
                "description": (
                    "Maltodextrin is a hydrolyzed starch additive with an exceptionally high glycemic index (GI 85–105), "
                    "higher than standard table sugar (GI ~65). It rapidly hydrolyzes into glucose in the small intestine, "
                    "even though it may not always be classified under 'Sugars' on older label formats."
                ),
                "items": maltodextrin_found,
                "action_advice": "Individuals with diabetes or insulin resistance should factor maltodextrin into their total glycemic load."
            })

        # 2. Hydrogenated Fat / Trans Fat Precursor Alert
        if hydrogenated_fats_found:
            hidden_insights.append({
                "type": "HYDROGENATED_FAT_WARNING",
                "severity": "CRITICAL",
                "title": "Industrial Hydrogenated Fat Detected",
                "description": (
                    f"Contains {', '.join(hydrogenated_fats_found)}. Hydrogenation creates industrial trans-fatty acids, "
                    "which raise atherogenic LDL cholesterol and lower protective HDL cholesterol. "
                    "The World Health Organization (WHO) and ICMR strongly recommend eliminating industrial trans fat intake."
                ),
                "items": hydrogenated_fats_found,
                "action_advice": "Prefer alternatives prepared with non-hydrogenated cold-pressed or single-source unhardened oils."
            })

        # 3. Palm Oil Predominance Alert
        if palm_oil_found:
            hidden_insights.append({
                "type": "PALM_OIL_ALERT",
                "severity": "MODERATE",
                "title": "Palm Oil / Palmolein Present",
                "description": (
                    f"Contains {', '.join(palm_oil_found)}. Palm oil is heavily used in commercial packaged foods "
                    "due to its low cost and high thermal stability. However, it is approximately 50% saturated fat "
                    "(primarily palmitic acid), which can elevate serum LDL cholesterol when consumed in frequent quantities."
                ),
                "items": palm_oil_found,
                "action_advice": "Moderation recommended, especially for individuals monitoring cardiovascular health or lipid profiles."
            })

        # 4. Hidden Sodium Sources Beyond Table Salt
        if len(hidden_sodium_found) >= 2 or (hidden_sodium_found and any("bicarbonate" in s["ingredient"].lower() or "msg" in s["ingredient"].lower() for s in hidden_sodium_found)):
            items_names = [s["ingredient"] for s in hidden_sodium_found]
            hidden_insights.append({
                "type": "HIDDEN_SODIUM_SOURCES",
                "severity": "HIGH",
                "title": f"Multiple Sodium-Containing Additives ({len(hidden_sodium_found)} sources)",
                "description": (
                    f"In addition to table salt, the product contains non-salt sodium compounds: {', '.join(items_names)}. "
                    "Chemical additives such as baking soda (sodium bicarbonate), MSG (monosodium glutamate), and sodium preservatives "
                    "substantially elevate total sodium content beyond what consumers perceive from salty taste alone."
                ),
                "items": items_names,
                "action_advice": "Check the declared sodium (mg) row on the nutrition table; do not estimate sodium from salt alone."
            })

        # 5. Profile-Specific Health Relevance
        profile_alerts = self._evaluate_profile_alignment(
            user_profile,
            hidden_sugars=hidden_sugars_found + maltodextrin_found,
            hidden_sodium=hidden_sodium_found,
            fats=palm_oil_found + hydrogenated_fats_found,
            refined_grains=refined_grains_found
        )

        # 6. Wholesome Ingredients & Product Benefits
        product_benefits: List[str] = []
        if positive_wholesome_found:
            product_benefits.extend(list(dict.fromkeys(positive_wholesome_found)))
        if not hydrogenated_fats_found:
            product_benefits.append("Zero industrial partially hydrogenated vegetable fats detected in ingredient list.")
        if not hidden_sugars_found and not maltodextrin_found:
            product_benefits.append("No added sugars, corn syrups, or high-glycemic maltodextrin listed in ingredients.")
        if not any(ins["category"] == "Synthetic Food Colour" for ins in ins_additives_found):
            product_benefits.append("No synthetic coal-tar / azo food colours (such as Tartrazine or Sunset Yellow) detected.")

        # 7. Curate Healthier Alternatives tailored to product and user budget
        alternatives = self._suggest_alternatives(
            raw_list=raw_list,
            has_sugars=bool(hidden_sugars_found or maltodextrin_found),
            has_palm_or_hydrogenated=bool(palm_oil_found or hydrogenated_fats_found),
            has_high_sodium=bool(hidden_sodium_found),
            user_profile=user_profile
        )

        # Remove empty categories from display map
        active_categories = {k: v for k, v in categories_map.items() if len(v) > 0}

        return {
            "total_ingredients": len(raw_list),
            "raw_ingredients": raw_list,
            "structured_ingredients": structured_ingredients,
            "categorized_ingredients": active_categories,
            "ins_additives": ins_additives_found,
            "hidden_insights": hidden_insights,
            "profile_alerts": profile_alerts,
            "product_benefits": product_benefits,
            "healthier_alternatives": alternatives,
            "processing_grade": self._assess_processing_grade(raw_list, ins_additives_found, palm_oil_found, hydrogenated_fats_found)
        }

    # -----------------------------------------------------------------------
    # Helper Inspection Methods
    # -----------------------------------------------------------------------

    def _inspect_single_ingredient(self, name: str, is_primary: bool = False) -> Dict[str, Any]:
        """Provides functional classification, purpose, and safety summary for an ingredient."""
        lower = name.lower()

        # Check INS Additives first
        ins_matches = extract_ins_numbers(name)
        if ins_matches:
            _, info = ins_matches[0]
            display_cat = "Other Additives"
            if "Colour" in info["category"]: display_cat = "Colours & Flavours"
            elif "Preservative" in info["category"] or "Antioxidant" in info["category"]: display_cat = "Preservatives & Antioxidants"
            elif "Emulsifier" in info["category"] or "Thickener" in info["category"] or "Stabilizer" in info["category"]: display_cat = "Emulsifiers & Stabilizers"
            elif "Sweetener" in info["category"]: display_cat = "Added Sugars & Sweeteners"
            elif "Leavening" in info["category"] or "Flavour Enhancer" in info["category"]: display_cat = "Leavening & Sodium Additives"

            return {
                "name": name,
                "clean_name": info["name"],
                "category": info["category"],
                "display_category": display_cat,
                "purpose": info["purpose"],
                "health_relevance": info["health_relevance"],
                "concerns": info["concerns"],
                "safety_status": info["safety_status"],
                "is_primary": is_primary
            }

        # Check Sugars
        for pat in SUGAR_SYNONYMS:
            if re.search(pat, lower):
                return {
                    "name": name,
                    "clean_name": name,
                    "category": "Added Sugar / Sweetener",
                    "display_category": "Added Sugars & Sweeteners",
                    "purpose": "Provides sweet taste, moisture retention, browning (Maillard reaction), and bulk.",
                    "health_relevance": "Fast-absorbing simple carbohydrate. Supplies empty calories with high glycemic response.",
                    "concerns": "Excessive intake linked to insulin resistance, hepatic fat accumulation, dental caries, and obesity.",
                    "safety_status": "Generally Recognized as Safe (Consume in Strict Moderation)",
                    "is_primary": is_primary
                }

        # Check Maltodextrin
        for pat in MALTODEXTRIN_PATTERNS:
            if re.search(pat, lower):
                return {
                    "name": name,
                    "clean_name": "Maltodextrin",
                    "category": "Refined Carbohydrate / Bulking Agent",
                    "display_category": "Added Sugars & Sweeteners",
                    "purpose": "Acts as filler, texturizer, spray-drying carrier, and shelf-life extender.",
                    "health_relevance": "Very high glycemic index (GI 85–105). Absorbed as rapidly as pure glucose.",
                    "concerns": "Causes rapid postprandial glucose and insulin spikes; easily overlooked by consumers.",
                    "safety_status": "Generally Permitted Food Additive",
                    "is_primary": is_primary
                }

        # Check Fats and Oils
        for p_pat in FATS_AND_OILS["palm_oil"]:
            if re.search(p_pat, lower):
                return {
                    "name": name,
                    "clean_name": "Palmolein / Palm Oil",
                    "category": "Refined Vegetable Oil (High Saturated Fat)",
                    "display_category": "Fats & Oils",
                    "purpose": "Provides inexpensive fat with high oxidative stability and semi-solid room temperature texture.",
                    "health_relevance": "Contains ~50% saturated fatty acids (palmitic acid). Moderately elevates LDL cholesterol.",
                    "concerns": "High consumption in packaged foods contributes substantially to saturated fat load.",
                    "safety_status": "Permitted Edible Oil (Limit Intake)",
                    "is_primary": is_primary
                }

        for h_pat in FATS_AND_OILS["hydrogenated_fat"]:
            if re.search(h_pat, lower):
                return {
                    "name": name,
                    "clean_name": "Hydrogenated Vegetable Fat / Vanaspati",
                    "category": "Industrial Hydrogenated Fat (Trans Fat Risk)",
                    "display_category": "Fats & Oils",
                    "purpose": "Hardens liquid oils to create solid shortening with very long shelf-life.",
                    "health_relevance": "Contains harmful trans fatty acids. Lowers HDL and elevates LDL cholesterol.",
                    "concerns": "Strongly linked to coronary artery disease and systemic inflammation.",
                    "safety_status": "To Be Avoided / Severely Limited",
                    "is_primary": is_primary
                }

        for r_pat in FATS_AND_OILS["refined_vegetable_oils"]:
            if re.search(r_pat, lower):
                return {
                    "name": name,
                    "clean_name": name,
                    "category": "Refined Vegetable Cooking Oil",
                    "display_category": "Fats & Oils",
                    "purpose": "Provides cooking medium, texture, and mouthfeel in snack processing.",
                    "health_relevance": "High energy density (9 kcal/g); high in Omega-6 polyunsaturated fatty acids.",
                    "concerns": "Imbalanced Omega-6 to Omega-3 ratio when consumed in large quantities in processed snacks.",
                    "safety_status": "Standard Food Ingredient",
                    "is_primary": is_primary
                }

        # Check Grains
        for rg_pat, rg_desc in REFINED_GRAINS:
            if re.search(rg_pat, lower):
                return {
                    "name": name,
                    "clean_name": name,
                    "category": "Refined Grain / Starch",
                    "display_category": "Refined Grains & Starches",
                    "purpose": "Provides main structural bulk, light crumb, and soft texture.",
                    "health_relevance": "Stripped of cereal germ and bran; lacks dietary fiber and natural micronutrients.",
                    "concerns": "High glycemic index; rapid conversion to glucose in the bloodstream.",
                    "safety_status": "Standard Food Ingredient",
                    "is_primary": is_primary
                }

        for w_pat, w_desc in WHOLESOME_INGREDIENTS:
            if re.search(w_pat, lower):
                return {
                    "name": name,
                    "clean_name": name,
                    "category": "Wholesome / Whole Grain Ingredient",
                    "display_category": "Primary Ingredients",
                    "purpose": "Nutritional grain/seed providing natural complex carbs, fiber, and micronutrients.",
                    "health_relevance": "Wholesome, nutrient-dense food component.",
                    "concerns": "None; highly desirable for metabolic wellness.",
                    "safety_status": "Wholesome & Recommended",
                    "is_primary": is_primary
                }

        # Check Salt
        if re.search(r"\bsalt\b|\biodised\s*salt\b|\bvacuum\s*evaporated\b", lower):
            return {
                "name": name,
                "clean_name": "Iodised Table Salt",
                "category": "Seasoning & Sodium Source",
                "display_category": "Leavening & Sodium Additives",
                "purpose": "Enhances palatability, acts as natural preservative, and supplies essential iodine.",
                "health_relevance": "Table salt is ~40% sodium by weight. Daily recommendation is under 5g salt (2,000 mg sodium).",
                "concerns": "Excess sodium intake is the primary dietary risk factor for hypertension and stroke.",
                "safety_status": "Standard Dietary Mineral (Monitor Intake)",
                "is_primary": is_primary
            }

        # Default fallback
        display_cat = "Primary Ingredients" if is_primary else "Other Additives"
        return {
            "name": name,
            "clean_name": name,
            "category": "Food Ingredient" if is_primary else "Specialty Additive / Seasoning",
            "display_category": display_cat,
            "purpose": "Contributes to taste, texture, appearance, or stability of the finished product.",
            "health_relevance": "Permitted ingredient used within standard food manufacturing practices.",
            "concerns": "None known in standard dietary usage.",
            "safety_status": "Permitted Food Ingredient",
            "is_primary": is_primary
        }

    def _evaluate_profile_alignment(
        self,
        profile: Optional[Dict[str, Any]],
        hidden_sugars: List[str],
        hidden_sodium: List[Dict[str, str]],
        fats: List[str],
        refined_grains: List[str]
    ) -> List[Dict[str, Any]]:
        """Evaluates ingredients against specific clinical health profile risks."""
        alerts: List[Dict[str, Any]] = []
        if not profile:
            return alerts

        restrictions = [str(r).lower() for r in (profile.get("dietary_restrictions") or profile.get("food_restrictions") or [])]
        goal = str(profile.get("dietary_goal") or profile.get("nutrition_goal") or "").lower()
        allergies = [str(a).lower() for a in (profile.get("allergies") or profile.get("food_allergies") or [])]

        is_hypertensive = any("hypertens" in r for r in restrictions) or "hypertens" in goal
        is_diabetic = any("diabet" in r for r in restrictions) or "diabet" in goal
        is_pcos = "pcos" in goal or any("pcos" in r for r in restrictions)
        is_celiac = any("gluten" in r or "celiac" in r for r in restrictions)
        is_weight_loss = "weight_loss" in goal

        if is_hypertensive and hidden_sodium:
            alerts.append({
                "profile": "Hypertension / DASH Diet",
                "severity": "HIGH",
                "message": (
                    f"⚠️ Caution for Hypertension: Product contains {len(hidden_sodium)} non-salt sodium contributors "
                    f"({', '.join([s['ingredient'] for s in hidden_sodium])}) in addition to table salt. "
                    "Non-salt sodium additives directly raise blood pressure."
                )
            })

        if (is_diabetic or is_pcos) and hidden_sugars:
            alerts.append({
                "profile": "Diabetes & PCOS / Insulin Sensitivity",
                "severity": "HIGH",
                "message": (
                    f"⚠️ Glycemic Spike Risk: Detected {len(hidden_sugars)} rapid-acting sweetener/sugar sources "
                    f"({', '.join(hidden_sugars)}). Rapidly elevates postprandial glucose and worsens insulin resistance."
                )
            })

        if is_celiac and refined_grains:
            alerts.append({
                "profile": "Gluten-Free / Celiac",
                "severity": "CRITICAL",
                "message": "🚫 Contains Wheat/Maida: Not suitable for celiac disease or gluten intolerance."
            })

        if is_weight_loss and fats:
            alerts.append({
                "profile": "Weight Management",
                "severity": "MODERATE",
                "message": f"Contains calorie-dense added fats ({', '.join(fats)}). Check portion size carefully."
            })

        return alerts

    def _assess_processing_grade(
        self,
        raw_list: List[str],
        ins_additives: List[Dict],
        palm_oil: List[str],
        hydrogenated_fats: List[str]
    ) -> Dict[str, Any]:
        """Classifies product according to the internationally recognized NOVA food processing framework."""
        # Ultra-processed markers: cosmetic additives (flavor enhancers, synthetic colors, high intensity sweeteners, emulsifiers like PGPR, DATEM, hydrogenated fats)
        nova_markers = []
        if hydrogenated_fats:
            nova_markers.append("Hydrogenated vegetable fats")
        if palm_oil:
            nova_markers.append("Fractionated / refined palm oil")
        for ins in ins_additives:
            cat = ins.get("category", "")
            if any(term in cat for term in ["Flavour Enhancer", "Synthetic Food Colour", "Artificial Sweetener", "Emulsifier"]):
                nova_markers.append(f"{ins['name']} ({ins['category']})")

        marker_count = len(nova_markers)
        if marker_count >= 2 or hydrogenated_fats:
            return {
                "nova_group": 4,
                "classification": "Ultra-Processed Food (UPF)",
                "description": "Industrial formulation containing substances not commonly used in culinary preparations (emulsifiers, flavour enhancers, or cosmetic additives).",
                "color": "#F43F5E",
                "badge_class": "nova-4"
            }
        elif marker_count == 1 or len(raw_list) > 6:
            return {
                "nova_group": 3,
                "classification": "Processed Food",
                "description": "Manufactured food made by adding salt, oil, or preservatives to whole ingredients.",
                "color": "#F59E0B",
                "badge_class": "nova-3"
            }
        else:
            return {
                "nova_group": 1,
                "classification": "Minimally Processed Food",
                "description": "Whole food modified by simple processes like roasting, drying, or boiling without industrial chemical additives.",
                "color": "#10B981",
                "badge_class": "nova-1"
            }

    def _suggest_alternatives(
        self,
        raw_list: List[str],
        has_sugars: bool,
        has_palm_or_hydrogenated: bool,
        has_high_sodium: bool,
        user_profile: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Recommends culturally authentic, wholesome, budget-friendly Indian snack and food alternatives."""
        all_text = " ".join(raw_list).lower()
        budget = (user_profile or {}).get("budget", "medium").lower()

        # Heuristic detection of product category
        is_noodle_or_pasta = any(k in all_text for k in ["noodle", "pasta", "seasoning", "tastemaker", "hydrolysed"])
        is_biscuit_or_cookie = any(k in all_text for k in ["biscuit", "cookie", "raising agent", "maida", "dough", "shortbread"])
        is_chips_or_namkeen = any(k in all_text for k in ["potato", "namkeen", "bhujia", "sev", "chips", "crisps", "fry"])
        is_beverage = any(k in all_text for k in ["carbonated", "syrup", "juice", "beverage", "drink", "cola"])

        alternatives = []

        if is_noodle_or_pasta:
            alternatives.append({
                "name": "Millet / Whole Wheat Vegetable Upma",
                "price_tier": "Low (₹15–25/portion)",
                "calories": "180 kcal",
                "fat_reduction": "-70% Fat",
                "why_better": "Made from whole foxtail or barnyard millet with fresh vegetables. Zero palm oil, zero synthetic flavor enhancers (MSG), and 4x more dietary fiber.",
                "suitable_for": ["Diabetic Friendly", "Heart Healthy", "High Fiber"]
            })
            alternatives.append({
                "name": "Spiced Steamed Moong Dal Poha",
                "price_tier": "Low (₹15–20/portion)",
                "calories": "210 kcal",
                "fat_reduction": "-60% Fat",
                "why_better": "Lightly tempered with mustard seeds, curry leaves, and roasted peanuts. Provides natural plant protein without deep-fried noodle cakes.",
                "suitable_for": ["Budget Friendly", "Weight Management"]
            })

        elif is_biscuit_or_cookie:
            alternatives.append({
                "name": "Roasted Ragi (Finger Millet) & Sesame Cookies",
                "price_tier": "Medium (₹30–40/portion)",
                "calories": "140 kcal",
                "fat_reduction": "-40% Saturated Fat",
                "why_better": "Rich in natural dietary calcium and iron. Sweetened with small amounts of unrefined jaggery instead of invert syrup and artificial vanilla.",
                "suitable_for": ["Calcium Rich", "Zero Palm Oil", "PCOS Friendly"]
            })
            alternatives.append({
                "name": "Roasted Peanuts with Jaggery (Chana-Gud)",
                "price_tier": "Low (₹10–15/portion)",
                "calories": "160 kcal",
                "fat_reduction": "Heart-Healthy Monounsaturated Fats",
                "why_better": "Classic traditional Indian snack combining roasted legumes with natural minerals. Zero emulsifiers, zero preservatives, and rich natural protein.",
                "suitable_for": ["Budget Friendly", "Natural Protein", "Zero Additives"]
            })

        elif is_chips_or_namkeen:
            alternatives.append({
                "name": "Dry-Roasted Phool Makhana (Fox Nuts)",
                "price_tier": "Medium (₹25–35/portion)",
                "calories": "110 kcal",
                "fat_reduction": "-80% Fat",
                "why_better": "Air-roasted lotus seeds tossed with a drop of ghee and rock salt. Exceptionally light, low in sodium, zero trans fat, and rich in potassium.",
                "suitable_for": ["Hypertension Safe", "Low Calorie", "Diabetic Friendly"]
            })
            alternatives.append({
                "name": "Roasted Spiced Chana (Bengal Gram)",
                "price_tier": "Low (₹10–15/portion)",
                "calories": "140 kcal",
                "fat_reduction": "-75% Fat vs Fried Sev",
                "why_better": "Crispy roasted whole black chickpeas with skin. Packed with 7g plant protein and 6g dietary fiber per serving.",
                "suitable_for": ["Budget Friendly", "High Protein", "Low GI"]
            })

        elif is_beverage:
            alternatives.append({
                "name": "Spiced Buttermilk (Masala Chaas)",
                "price_tier": "Low (₹10–15/portion)",
                "calories": "45 kcal",
                "fat_reduction": "-90% Calories vs Cola",
                "why_better": "Naturally fermented cultured yogurt drink with roasted cumin, ginger, and mint. Delivers natural gut probiotics with zero added sugar.",
                "suitable_for": ["Digestive Health", "Zero Added Sugar", "PCOS Safe"]
            })
            alternatives.append({
                "name": "Fresh Tender Coconut Water",
                "price_tier": "Medium (₹40–50/portion)",
                "calories": "40 kcal",
                "fat_reduction": "Zero Fat",
                "why_better": "Natural isotonic electrolyte beverage rich in potassium and magnesium. Completely free from caramel colour (INS 150d) and artificial sweeteners.",
                "suitable_for": ["Natural Electrolytes", "Hypertension Friendly"]
            })

        else:
            # General Wholesome Indian Snack Alternatives
            alternatives.append({
                "name": "Spiced Roasted Chana & Makhana Medley",
                "price_tier": "Low-to-Medium (₹20/portion)",
                "calories": "130 kcal",
                "fat_reduction": "-70% Fat vs Packaged Namkeen",
                "why_better": "Crispy air-roasted whole pulses and water lily seeds seasoned with cumin and dry mango powder. Unprocessed, high in natural protein and dietary fiber.",
                "suitable_for": ["General Wellness", "Budget Friendly", "Clean Eating"]
            })
            alternatives.append({
                "name": "Fresh Seasonal Fruit Bowl with Chaat Masala",
                "price_tier": "Low (₹15–25/portion)",
                "calories": "90 kcal",
                "fat_reduction": "Zero Added Oil",
                "why_better": "Whole fresh guava, papaya, or apple. Delivers intact dietary fiber, living enzymes, and Vitamin C without preservatives.",
                "suitable_for": ["Heart Healthy", "High Fiber", "Weight Loss"]
            })

        return alternatives

    def _empty_analysis(self) -> Dict[str, Any]:
        """Fallback empty analysis structure."""
        return {
            "total_ingredients": 0,
            "raw_ingredients": [],
            "structured_ingredients": [],
            "categorized_ingredients": {},
            "ins_additives": [],
            "hidden_insights": [],
            "profile_alerts": [],
            "product_benefits": [],
            "healthier_alternatives": [],
            "processing_grade": {
                "nova_group": 1,
                "classification": "Unspecified",
                "description": "No ingredients detected.",
                "color": "#94A3B8",
                "badge_class": "nova-none"
            }
        }


# ---------------------------------------------------------------------------
# 6. Popular Indian Packaged Food Knowledge Base (Manual Fallback)
# ---------------------------------------------------------------------------

POPULAR_PACKAGED_FOODS: List[Dict[str, Any]] = [
    {
        "product_name": "Maggi 2-Minute Masala Noodles",
        "brand": "Nestlé",
        "category": "Instant Noodles",
        "aliases": ["maggi", "maggi noodles", "nestle maggi", "masala noodles", "2 minute noodles"],
        "serving_size": "70",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "1",
        "ingredients": "Refined Wheat Flour (Maida) (68%), Palm Oil, Iodised Salt, Wheat Gluten, Mineral (Calcium Carbonate), Thickeners (INS 508, INS 412), Acidity Regulators (INS 501(i), INS 500(i)), Humectant (INS 451(i)). TASTEMAKER: Hydrolysed Groundnut Protein, Mixed Spices (Noodles Masala 23.6%) [Dehydrated Onion, Coriander Powder, Red Chilli Powder, Turmeric Powder, Dehydrated Garlic, Cumin, Aniseed, Black Pepper, Fenugreek, Ginger, Clove, Nutmeg, Cardamom], Sugar, Edible Starch, Iodised Salt, Palm Oil, Thickener (INS 508), Flavour Enhancers (INS 627, INS 631), Colour (INS 150d), Acidity Regulator (INS 330).",
        "nutrition": {
            "calories": {"value": 427, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 8.0, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 63.5, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 2.2, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 1.5, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 3.6, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 15.7, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 6.8, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 1040, "unit": "mg", "ocr_extracted": False},
            "salt": {"value": 2.6, "unit": "g", "ocr_extracted": False},
            "calcium": {"value": 120, "unit": "mg", "ocr_extracted": False},
            "iron": {"value": 3.5, "unit": "mg", "ocr_extracted": False}
        },
        "allergens": {
            "declared": ["wheat", "gluten", "peanuts"],
            "detected_from_ingredients": ["wheat", "gluten", "peanuts"],
            "may_contain": ["milk", "soy", "mustard"],
            "all_identified_allergens": ["wheat", "gluten", "peanuts", "milk", "soy", "mustard"]
        }
    },
    {
        "product_name": "Britannia Good Day Butter Cookies",
        "brand": "Britannia",
        "category": "Cookies & Biscuits",
        "aliases": ["good day", "good day butter", "britannia good day", "good day biscuits"],
        "serving_size": "30",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "4",
        "ingredients": "Refined Wheat Flour (Maida) (58%), Sugar, Edible Vegetable Oil (Palmolein), Butter (2%), Invert Sugar Syrup, Liquid Glucose, Raising Agents [INS 503(ii), INS 500(ii)], Iodised Salt, Emulsifiers [INS 322 from Soy, INS 471], Dough Conditioner (INS 223), Nature Identical and Artificial Flavouring Substances (Butter, Vanilla).",
        "nutrition": {
            "calories": {"value": 490, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 7.0, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 67.0, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 22.5, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 21.0, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 1.5, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 22.0, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 10.5, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 320, "unit": "mg", "ocr_extracted": False},
            "salt": {"value": 0.8, "unit": "g", "ocr_extracted": False}
        },
        "allergens": {
            "declared": ["wheat", "milk", "soy"],
            "detected_from_ingredients": ["wheat", "milk", "soy"],
            "may_contain": ["tree nuts"],
            "all_identified_allergens": ["wheat", "milk", "soy", "tree nuts"]
        }
    },
    {
        "product_name": "Parle-G Original Glucose Biscuits",
        "brand": "Parle",
        "category": "Biscuits",
        "aliases": ["parle g", "parle-g", "parleg", "glucose biscuits"],
        "serving_size": "25",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "4",
        "ingredients": "Refined Wheat Flour (Maida) (67%), Sugar, Edible Vegetable Oil (Palm Oil), Invert Sugar Syrup, Raising Agents [INS 503(ii), INS 500(ii)], Milk Solids, Iodised Salt, Emulsifier (INS 322), Dough Conditioner (INS 223), Artificial Flavouring Substances (Vanilla).",
        "nutrition": {
            "calories": {"value": 454, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 6.5, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 78.2, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 26.5, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 25.5, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 1.2, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 13.0, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 6.2, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 280, "unit": "mg", "ocr_extracted": False}
        },
        "allergens": {
            "declared": ["wheat", "milk", "soy"],
            "detected_from_ingredients": ["wheat", "milk", "soy"],
            "may_contain": [],
            "all_identified_allergens": ["wheat", "milk", "soy"]
        }
    },
    {
        "product_name": "Lay's Classic Salted Potato Chips",
        "brand": "Lay's",
        "category": "Potato Chips & Crisps",
        "aliases": ["lays", "lays salted", "lay's", "lays chips", "potato chips"],
        "serving_size": "30",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "2",
        "ingredients": "Potato (63%), Edible Vegetable Oil (Palmolein, Rice Bran Oil), Iodised Salt (1.5%).",
        "nutrition": {
            "calories": {"value": 544, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 6.7, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 52.3, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 1.0, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 3.4, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 34.2, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 15.0, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.1, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 560, "unit": "mg", "ocr_extracted": False},
            "salt": {"value": 1.4, "unit": "g", "ocr_extracted": False}
        },
        "allergens": {
            "declared": [],
            "detected_from_ingredients": [],
            "may_contain": ["milk"],
            "all_identified_allergens": ["milk"]
        }
    },
    {
        "product_name": "Kurkure Masala Munch",
        "brand": "Kurkure",
        "category": "Extruded Namkeen Snack",
        "aliases": ["kurkure", "kurkure masala", "masala munch"],
        "serving_size": "30",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "3",
        "ingredients": "Rice Meal (42.8%), Edible Vegetable Oil (Palmolein), Corn Meal (19.8%), Gram Meal (3.3%), Spices and Condiments [Onion Powder, Chilli Powder, Amchur, Coriander, Black Pepper, Garlic Powder, Turmeric, Fenugreek, Cumin], Iodised Salt, Sugar, Tartaric Acid (INS 334), Citric Acid (INS 330).",
        "nutrition": {
            "calories": {"value": 558, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 5.8, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 56.6, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 3.0, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 1.5, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 1.8, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 34.5, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 16.0, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 890, "unit": "mg", "ocr_extracted": False},
            "salt": {"value": 2.2, "unit": "g", "ocr_extracted": False}
        },
        "allergens": {
            "declared": [],
            "detected_from_ingredients": [],
            "may_contain": ["milk", "soy", "wheat"],
            "all_identified_allergens": ["milk", "soy", "wheat"]
        }
    },
    {
        "product_name": "Haldiram's Aloo Bhujia",
        "brand": "Haldiram's",
        "category": "Traditional Savoury Namkeen",
        "aliases": ["aloo bhujia", "haldiram aloo bhujia", "haldirams aloo bhujia", "bhujia"],
        "serving_size": "35",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "4",
        "ingredients": "Potatoes (44%), Edible Vegetable Oil (Cottonseed, Corn & Palmolein Oil), Gram Pulse Flour (Tepary Bean & Chickpea Flour) (28%), Starch, Spices and Condiments (Chilli Powder, Cumin, Black Pepper, Ginger, Cardamom, Clove, Nutmeg), Iodised Salt, Mango Powder, Citric Acid (INS 330).",
        "nutrition": {
            "calories": {"value": 586, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 9.4, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 41.5, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 1.8, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 4.8, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 42.5, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 17.5, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 780, "unit": "mg", "ocr_extracted": False},
            "salt": {"value": 1.95, "unit": "g", "ocr_extracted": False}
        },
        "allergens": {
            "declared": [],
            "detected_from_ingredients": [],
            "may_contain": ["peanuts", "tree nuts", "soy", "gluten"],
            "all_identified_allergens": ["peanuts", "tree nuts", "soy", "gluten"]
        }
    },
    {
        "product_name": "Oreo Original Chocolate Sandwich Cookies",
        "brand": "Cadbury / Mondelez",
        "category": "Cream Biscuits & Cookies",
        "aliases": ["oreo", "oreo biscuits", "oreo cookies", "cadbury oreo"],
        "serving_size": "29",
        "serving_unit": "g",
        "nutrition_basis": "per 100g",
        "servings_per_container": "4",
        "ingredients": "Refined Wheat Flour (Maida), Sugar, Edible Fractionated Vegetable Fat (Palm Oil), Invert Sugar Syrup, Cocoa Solids (3.8%), Liquid Glucose, Raising Agents [INS 500(ii), INS 503(ii)], Iodised Salt, Emulsifier (INS 322 from Soy), Natural Flavouring Substances (Vanilla).",
        "nutrition": {
            "calories": {"value": 482, "unit": "kcal", "ocr_extracted": False},
            "protein": {"value": 5.2, "unit": "g", "ocr_extracted": False},
            "total_carbohydrates": {"value": 71.0, "unit": "g", "ocr_extracted": False},
            "total_sugars": {"value": 38.0, "unit": "g", "ocr_extracted": False},
            "added_sugars": {"value": 36.5, "unit": "g", "ocr_extracted": False},
            "dietary_fibre": {"value": 2.1, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 19.5, "unit": "g", "ocr_extracted": False},
            "saturated_fat": {"value": 9.5, "unit": "g", "ocr_extracted": False},
            "trans_fat": {"value": 0.0, "unit": "g", "ocr_extracted": False},
            "sodium": {"value": 390, "unit": "mg", "ocr_extracted": False}
        },
        "allergens": {
            "declared": ["wheat", "soy"],
            "detected_from_ingredients": ["wheat", "soy"],
            "may_contain": ["milk"],
            "all_identified_allergens": ["wheat", "soy", "milk"]
        }
    }
]


def lookup_packaged_product(query: str, user_profile: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Looks up a packaged product by name or keyword in the verified knowledge base.
    Returns complete structured OCR-compatible response with Smart Ingredient Intelligence.
    """
    if not query or not query.strip():
        return None

    q = query.lower().strip()
    q_tokens = set(re.findall(r"\w+", q))

    best_match = None
    best_score = 0

    for prod in POPULAR_PACKAGED_FOODS:
        # Check direct alias matches
        for al in prod.get("aliases", []):
            if al == q:
                best_match = prod
                best_score = 100
                break
            if al in q or q in al:
                score = 50 + len(al)
                if score > best_score:
                    best_score = score
                    best_match = prod

        # Token overlap matching
        prod_tokens = set(re.findall(r"\w+", prod["product_name"].lower()))
        overlap = len(q_tokens & prod_tokens)
        if overlap > 0:
            score = overlap * 20
            if score > best_score:
                best_score = score
                best_match = prod

    if not best_match or best_score < 20:
        return None

    # Run ingredient intelligence on matched product
    analyzer = get_ingredient_analyzer()
    ing_intel = analyzer.analyze(best_match["ingredients"], user_profile=user_profile)

    return {
        "ocr_text": f"{best_match['brand']} {best_match['product_name']}\nNutrition Facts: {best_match['nutrition_basis']}\nIngredients: {best_match['ingredients']}",
        "readability": {
            "mean_confidence": 0.99,
            "min_confidence": 0.95,
            "readable_bbox_count": 25,
            "quality": "GOOD"
        },
        "product": {
            "product_name": best_match["product_name"],
            "brand": best_match["brand"],
            "ocr_extracted": False
        },
        "serving": {
            "serving_size": best_match["serving_size"],
            "serving_unit": best_match["serving_unit"],
            "nutrition_basis": best_match["nutrition_basis"],
            "servings_per_container": best_match["servings_per_container"],
            "ocr_extracted": False
        },
        "nutrition": best_match["nutrition"],
        "ingredients": best_match["ingredients"],
        "allergens": best_match["allergens"],
        "missing_fields": [],
        "extraction_note": "Retrieved from NutriSense AI Verified Packaged Food Knowledge Base.",
        "disclaimer": (
            "Nutritional information is based on manufacturer declared specifications for educational awareness. "
            "Always verify against the physical package label. Not clinical advice."
        ),
        "ingredient_intelligence": ing_intel,
        "is_catalog_lookup": True
    }


# Singleton instance
_analyzer_instance = None

def get_ingredient_analyzer() -> SmartIngredientAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SmartIngredientAnalyzer()
    return _analyzer_instance

