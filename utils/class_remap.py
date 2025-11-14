# =========================================================
# utilss/class_remap.py
# =========================================================

import random
import streamlit as st

# =========================================================
# ✅ ORIGINAL TACO CLASS DEFINITIONS
# =========================================================
TACO_CLASS_NAMES = {
    0: 'Aluminium foil', 1: 'Battery', 2: 'Aluminium blister pack', 3: 'Carded blister pack',
    4: 'Other plastic bottle', 5: 'Clear plastic bottle', 6: 'Glass bottle', 7: 'Plastic bottle cap',
    8: 'Metal bottle cap', 9: 'Broken glass', 10: 'Food Can', 11: 'Aerosol', 12: 'Drink can',
    13: 'Toilet tube', 14: 'Other carton', 15: 'Egg carton', 16: 'Drink carton',
    17: 'Corrugated carton', 18: 'Meal carton', 19: 'Pizza box', 20: 'Paper cup',
    21: 'Disposable plastic cup', 22: 'Foam cup', 23: 'Glass cup', 24: 'Other plastic cup',
    25: 'Food waste', 26: 'Glass jar', 27: 'Plastic lid', 28: 'Metal lid', 29: 'Other plastic',
    30: 'Magazine paper', 31: 'Tissues', 32: 'Wrapping paper', 33: 'Normal paper', 34: 'Paper bag',
    35: 'Plastified paper bag', 36: 'Plastic film', 37: 'Six pack rings', 38: 'Garbage bag',
    39: 'Other plastic wrapper', 40: 'Single-use carrier bag', 41: 'Polypropylene bag',
    42: 'Crisp packet', 43: 'Spread tub', 44: 'Tupperware', 45: 'Disposable food container',
    46: 'Foam food container', 47: 'Other plastic container', 48: 'Plastic glooves',
    49: 'Plastic utensils', 50: 'Pop tab', 51: 'Rope & strings', 52: 'Scrap metal',
    53: 'Shoe', 54: 'Squeezable tube', 55: 'Plastic straw', 56: 'Paper straw',
    57: 'Styrofoam piece', 58: 'Unlabeled litter', 59: 'Cigarette',
}

# =========================================================
# 🔹 STEP 1: Map fine-grained TACO classes → 6 base categories
# =========================================================
TACO_TO_6CLASS = {
    "Paper": [
        "Toilet tube", "Other carton", "Egg carton", "Drink carton", "Corrugated carton",
        "Meal carton", "Pizza box", "Paper cup", "Magazine paper", "Tissues", "Wrapping paper",
        "Normal paper", "Paper bag", "Plastified paper bag", "Paper straw"
    ],
    "Glass": [
        "Glass bottle", "Broken glass", "Glass cup", "Glass jar"
    ],
    "Metal": [
        "Aluminium foil", "Aluminium blister pack", "Metal bottle cap", "Food Can", "Aerosol",
        "Drink can", "Metal lid", "Scrap metal", "Pop tab"
    ],
    "Plastic": [
        "Other plastic bottle", "Clear plastic bottle", "Plastic bottle cap", "Plastic lid",
        "Other plastic", "Plastic film", "Six pack rings", "Other plastic wrapper",
        "Single-use carrier bag", "Polypropylene bag", "Crisp packet", "Spread tub",
        "Tupperware", "Disposable food container", "Foam food container",
        "Other plastic container", "Plastic glooves", "Plastic utensils",
        "Squeezable tube", "Plastic straw", "Styrofoam piece"
    ],
    "Organic": [
        "Food waste"
    ],
    "Trash": [
        "Battery", "Garbage bag", "Unlabeled litter", "Cigarette", "Shoe", "Rope & strings"
    ]
}

# Reverse mapping for quick lookup
CLASS_REMAP_6 = {}
for broad, fine_list in TACO_TO_6CLASS.items():
    for fine in fine_list:
        CLASS_REMAP_6[fine] = broad

# =========================================================
# 🔹 STEP 2: Merge 6 base → 3 final recycle categories
# =========================================================
RECYCLE_MAP = {
    "Dry recycle": ["Paper", "Glass", "Metal", "Plastic"],
    "Organic": ["Organic"],
    "Trash": ["Trash"]
}
# Reverse lookup for fine → 3-class
CLASS_REMAP_3 = {}
for recycle_class, base_classes in RECYCLE_MAP.items():
    for base_class in base_classes:
        for fine, broad in CLASS_REMAP_6.items():
            if broad == base_class:
                CLASS_REMAP_3[fine] = recycle_class


@st.cache_data
def get_color_map(_=None):
    six_classes = ["Paper", "Glass", "Metal", "Plastic", "Organic", "Trash"]
    return {name: [random.randint(0, 255) for _ in range(3)] for name in six_classes}

@st.cache_data
def get_color_recycle_map(_=None):
    recycle_classes = ["Dry recycle", "Organic", "Trash"]
    return {name: [random.randint(0, 255) for _ in range(3)] for name in recycle_classes}