"""
generate_symptom_dataset.py
===========================
Generates a 8,000-row first-person symptom-description dataset for the 10
skin-condition classes used by the Bio_ClinicalBERT fine-tuning pipeline.

Design principles (agreed upon Sep 24 2026):
  - Two-tier vocabulary: a global shared pool + per-class discriminators,
    with a 3-level confusion-tier system that enforces overlapping language
    between clinically similar classes.
  - 7 slot categories (location, appearance, texture, sensation, evolution,
    trigger, border_shape) instead of the original 4.
  - 4 sentence difficulty levels (Easy 20% / Medium 40% / Hard 30% /
    Ambiguous 10%) per class, so the model cannot win by memorizing a single
    keyword -- it must learn symptom COMBINATIONS.
  - 65 sentence templates across 7 style categories (declarative, compound,
    multi-sentence narrative, casual/informal, fragment, concern-driven,
    trigger-contextual) with variable slot count and slot ordering.
  - 800 unique sentences per class => 8,000 total rows.
  - Outputs: real_symptoms_dataset.csv (text, label columns).

Deliberately EXCLUDED from all vocabulary:
  - Treatment / management information
  - Cause / risk factors / contagion
  - Prevalence / epidemiology statistics
  - Diagnostic history or named-after facts
"""

import csv
import random

random.seed(42)

TARGET_PER_CLASS = 800
MAX_ATTEMPTS = 200_000  # per class; prevents infinite loops

# ============================================================================
# 1. GLOBAL SHARED VOCABULARY
#    These phrases appear across ALL classes and must NOT be used as
#    class-discriminating signals. They represent the baseline language
#    any patient might use regardless of their condition.
# ============================================================================

GLOBAL = {
    "location": [
        "on my arm", "on my leg", "on my back", "on my chest",
        "on my shoulder", "on my neck", "on my face", "on my scalp",
        "on my hands", "on my feet", "on my wrist", "on my forearm",
        "near my knee", "near my elbow",
    ],
    "appearance": [
        "a red patch", "a raised bump", "a scaly area", "a rough patch",
        "a discolored spot", "an inflamed patch", "a dry patch",
        "a skin lesion", "a reddish mark",
    ],
    "texture": [
        "dry and rough to the touch", "rough and flaky",
        "flaky and thickened", "dry and scaly",
        "rough-feeling in that spot",
    ],
    "sensation": [
        "mild itching", "some mild discomfort",
        "mild irritation", "no real pain",
        "occasional mild itching", "no significant pain",
    ],
    "evolution": [
        "it's been there for a while", "it showed up recently",
        "it hasn't changed much", "it came on gradually",
        "it's been there for several weeks",
    ],
    "trigger": [
        "when the weather changes", "when I sweat",
        "when my clothes rub against it",
    ],
    "border_shape": [
        "with somewhat defined edges", "with irregular edges",
        "with fuzzy borders",
    ],
}

# ============================================================================
# 2. PER-CLASS VOCABULARY
#    Each class has:
#      - "discriminators": the clinical features that distinguish it from
#        its confusion-tier neighbors. These are protected keywords.
#      - Per-slot shared vocabulary that overlaps with confusion-tier
#        partners (Tier 1 ~60-70%, Tier 2 ~40-50%, Tier 3 ~20-30%).
#
#    Confusion tier map:
#      Tier 1 (High overlap):
#        Eczema <-> Atopic Dermatitis
#        Seborrheic Keratoses <-> Benign Keratosis-like Lesions
#      Tier 2 (Moderate overlap):
#        Eczema <-> Psoriasis Lichen Planus
#        Melanocytic Nevi <-> Melanoma
#        Basal Cell Carcinoma <-> Melanoma
#      Tier 3 (Low overlap):
#        Psoriasis Lichen Planus <-> Fungal Infections
#        Viral Infections <-> Seborrheic Keratoses
# ============================================================================

VOCAB = {

    # ─── ECZEMA ──────────────────────────────────────────────────────────────
    # Tier 1 neighbor: Atopic Dermatitis
    # Tier 2 neighbor: Psoriasis Lichen Planus
    "Eczema": {
        "discriminators": [
            "patches that ooze clear fluid when I scratch",
            "skin that weeps and crusts over",
            "blistering skin that bursts and leaves raw, wet patches",
            "skin that breaks open and oozes when it flares",
            "patches with blurry, ill-defined edges that spread gradually",
            "skin irritation triggered by a certain soap or fabric",
            "raw, weeping patches that crust over as they heal",
        ],
        "location": [
            "on the inside of my elbows", "behind my knees",
            "on my eyelids", "on my hands", "on my wrists",
            "on the back of my hands", "on my fingers",
        ],
        "appearance": [
            "red to brownish-gray patches", "small fluid-filled bumps",
            "cracked, dry skin that stings", "raw, inflamed skin",
            "patches that ooze and crust", "red, inflamed areas",
            "scaly red patches",                # Tier 2 shared with Psoriasis
            "itchy red patches",                # shared
        ],
        "texture": [
            "rough and leathery feeling", "feels crusty and dry",
            "the skin there is thickened from years of scratching",
            "dry and cracked to the touch",
            "flaky and rough",                  # shared with Psoriasis T2
        ],
        "sensation": [
            "intense itching that gets worse at night",
            "itching so bad it keeps me awake",
            "a burning feeling when I sweat",
            "severe itching I can barely control",
            "itchy and irritated",              # Tier 1 shared with AD
            "constant urge to scratch",         # Tier 1 shared with AD
        ],
        "evolution": [
            "clears up for a few weeks and then flares back up",
            "keeps coming back every winter when the air gets dry",
            "spreads a bit more every time I scratch it",
            "has been flaring on and off for months",
            "flares and then settles down",     # Tier 1 shared with AD
        ],
        "trigger": [
            "whenever I use a harsh soap or detergent",
            "when I wear synthetic fabrics",
            "in cold, dry weather",             # Tier 1 shared with AD
            "when I sweat",                     # global
        ],
        "border_shape": [
            "with blurry, ill-defined edges",
            "with poorly defined borders that fade into normal skin",
            "with no sharp edge, just gradually redder skin",
        ],
    },

    # ─── ATOPIC DERMATITIS ───────────────────────────────────────────────────
    # Tier 1 neighbor: Eczema
    "Atopic Dermatitis": {
        "discriminators": [
            "symmetrical red patches on both sides of my body",
            "dry, itchy patches that have been recurring since childhood",
            "small rough bumps covering my upper arms and thighs",
            "skin prone to frequent infections from all the scratching",
            "an itch that triggers before any visible rash appears",
            "a hereditary skin condition -- my sibling has it too",
        ],
        "location": [
            "on the folds of my elbows and knees", "around my eyes",
            "on my neck", "on my cheeks", "on my scalp",
            "on my wrists", "on the back of my hands",
        ],
        "appearance": [
            "dry, red inflamed patches", "scaly, irritated skin",
            "skin that leaks clear fluid after scratching",
            "thickened, leathery patches",
            "red, inflamed areas",              # Tier 1 shared with Eczema
            "cracked, dry skin",                # Tier 1 shared with Eczema
        ],
        "texture": [
            "rough and thickened from scratching",
            "feels dry and scaly to the touch",
            "the skin there is leathery and thick",
            "dry and cracked to the touch",     # Tier 1 shared with Eczema
        ],
        "sensation": [
            "intense itching that wakes me up at night",
            "constant urge to scratch that never really goes away",
            "itchy and irritated",              # Tier 1 shared with Eczema
            "a burning, stinging feeling after I sweat",
            "itching so bad it keeps me awake", # Tier 1 shared with Eczema
        ],
        "evolution": [
            "has been a recurring problem since I was young",
            "flares and then settles down",     # Tier 1 shared with Eczema
            "gets noticeably worse in cold, dry weather",
            "keeps returning in the same spots year after year",
            "has come and gone since childhood",
        ],
        "trigger": [
            "in cold, dry weather",             # Tier 1 shared with Eczema
            "when I'm stressed",
            "when I sweat during exercise",
            "when I use scented products",
        ],
        "border_shape": [
            "with indistinct, gradually fading edges",
            "with blurry borders that blend into surrounding skin",
            "with poorly defined edges",
        ],
    },

    # ─── PSORIASIS LICHEN PLANUS ─────────────────────────────────────────────
    # Tier 2 neighbor: Eczema
    # Tier 3 neighbor: Fungal Infections
    "Psoriasis Lichen Planus": {
        "discriminators": [
            "thick silvery-white scales that flake off in sheets",
            "sharply defined plaques with very clear, distinct edges",
            "small flat-topped shiny purplish bumps",
            "bumps with a fine white lacy pattern on the surface",
            "pinpoint bleeding spots when the scale is removed",
            "patches on the extensor surface -- the outside of my elbows and knees",
            "plaques appearing exactly where my skin was scratched or injured",
        ],
        "location": [
            "on my elbows and knees", "on my scalp", "on my lower back",
            "on my shins", "on my wrists", "inside my mouth",
            "on my ankles",
            "on my arms",                       # Tier 2 shared with Eczema
        ],
        "appearance": [
            "thick red raised plaques", "red patches with silvery-white scales",
            "sharply defined, scaly plaques", "dry, scaly skin",
            "scaly red patches",                # Tier 2 shared with Eczema
            "red, ring-shaped scaly patches",   # Tier 3 shared with Fungal
        ],
        "texture": [
            "covered in thick silvery scales",
            "flaky when I run my hand over it",
            "feels rough and raised",
            "thick and plaque-like",
            "flaky and rough",                  # Tier 2 shared with Eczema
        ],
        "sensation": [
            "itching that can get severe",
            "a stinging sensation where the skin cracks at the edges",
            "occasional mild soreness",
            "itchy and irritated",              # Tier 2 shared with Eczema
        ],
        "evolution": [
            "has lasted for months without going away",
            "keeps flaking off no matter what I do",
            "comes back in the same spots every few months",
            "showed up as small dots and then grew into larger plaques",
            "has persisted for a long time without clearing",
        ],
        "trigger": [
            "after I had a stressful period at work",
            "after a skin injury or scratch",
            "in dry weather",
        ],
        "border_shape": [
            "with very sharply defined, distinct edges",
            "with clear, well-demarcated borders",
            "with precise, punched-out looking borders",
        ],
    },

    # ─── FUNGAL INFECTIONS ───────────────────────────────────────────────────
    # Tier 3 neighbor: Psoriasis Lichen Planus
    "Fungal Infections": {
        "discriminators": [
            "a ring-shaped rash with a raised scaly outer border and clearer skin in the middle",
            "athlete's foot causing cracked, peeling, white skin between my toes",
            "a thick, yellowed, crumbly nail that's lifting off the nail bed",
            "small circular bald patches on my scalp",
            "a rash that keeps expanding outward in a ring shape",
            "a reddish-brown patch that gets worse in hot, humid weather",
        ],
        "location": [
            "between my toes", "on the sole of my foot",
            "on my groin and inner thigh", "on my scalp",
            "on my torso", "on my fingernail", "on my toenail",
            "on my upper back and chest",
        ],
        "appearance": [
            "a ring-shaped rash with a raised border",
            "cracked, peeling skin", "white, scaly patches",
            "a reddish-brown rash", "a circular scaly rash",
            "red, ring-shaped scaly patches",   # Tier 3 shared with Psoriasis
        ],
        "texture": [
            "soft and macerated where skin touches skin",
            "crumbly and brittle on the nail",
            "peeling and flaking",
            "rough around the border, softer in the center",
        ],
        "sensation": [
            "intense itching and burning",
            "a stinging feeling especially after sweating",
            "mild itching that worsens in humid weather",
            "burning between my toes",
        ],
        "evolution": [
            "keeps expanding outward from the center",
            "spread from one foot to the other",
            "has been getting worse in hot, humid conditions",
            "hasn't cleared up even after several weeks",
        ],
        "trigger": [
            "in hot and humid weather",
            "when my feet stay damp in sweaty shoes",
            "when I walk barefoot in public areas",
        ],
        "border_shape": [
            "with a clearly raised, scaly outer edge",
            "with a well-defined ring-shaped border",
            "with a distinct circular outer rim",
        ],
    },

    # ─── VIRAL INFECTIONS ────────────────────────────────────────────────────
    # Tier 3 neighbor: Seborrheic Keratoses
    "Viral Infections": {
        "discriminators": [
            "a rough, cauliflower-like bump with tiny black dots on the surface",
            "a hard lump with visible black specks that look like seeds",
            "a smooth, pearly bump with a small dimple in the center",
            "a thin, thread-like skin tag growing on my eyelid",
            "a flat-topped warty bump that keeps multiplying",
            "a plantar wart on my heel that hurts when I walk",
        ],
        "location": [
            "on my finger", "on the sole of my foot", "on my face",
            "around my eyes", "on my hands", "on the back of my knuckles",
            "on my armpit", "on my neck",
        ],
        "appearance": [
            "a small rough bump", "a raised, warty growth",
            "a cluster of small flesh-colored bumps",
            "a hard, rough-surfaced growth",
            "a raised, well-defined bump",      # Tier 3 shared with Seb. Kera.
        ],
        "texture": [
            "rough like sandpaper on the surface",
            "hard and firm to the touch",
            "rough and irregular",
            "smooth with a central dimple",
        ],
        "sensation": [
            "no pain at all, just an odd texture",
            "mild discomfort when I press on it",
            "occasional itching around the bump",
            "painful when I put weight on it",
        ],
        "evolution": [
            "keeps spreading into a small cluster",
            "grew slowly over several months",
            "showed up suddenly and multiplied into several bumps",
            "has stayed about the same size for weeks",
        ],
        "trigger": [
            "after I cut myself shaving near the area",
            "the cluster grew after I picked at it",
        ],
        "border_shape": [
            "with a rough, uneven surface and irregular edges",
            "with a clearly defined circular border",
            "with distinct, punched-out looking edges",
        ],
    },

    # ─── MELANOCYTIC NEVI ────────────────────────────────────────────────────
    # Tier 2 neighbor: Melanoma
    "Melanocytic Nevi": {
        "discriminators": [
            "a perfectly round, evenly colored brown spot that hasn't changed at all",
            "a symmetric mole with a smooth, even border",
            "a single-colored mole that looks exactly the same as it did years ago",
            "a soft, dome-shaped mole that's been stable for as long as I can remember",
            "a small, uniform dark brown spot with no irregular areas",
        ],
        "location": [
            "on my back", "on my arm", "on my shoulder",
            "on my chest", "on my leg", "on my face",
            "on my stomach",
        ],
        "appearance": [
            "a small, round, evenly colored brown spot",
            "a flat, tan-colored mark", "a smooth, symmetric mole",
            "a slightly raised, uniform-colored bump",
            "a dark brown spot",                # Tier 2 shared with Melanoma
            "a pigmented spot",                 # Tier 2 shared with Melanoma
        ],
        "texture": [
            "smooth and flat against my skin",
            "slightly raised but soft",
            "even and smooth to the touch",
        ],
        "sensation": [
            "no itching or pain at all", "no discomfort whatsoever",
            "nothing unusual, it's just always been there",
        ],
        "evolution": [
            "has looked exactly the same for years",
            "hasn't changed in size, shape, or color",
            "has been there since I was a kid and never changed",
            "stays exactly the same",
            "has been completely stable",
        ],
        "trigger": [
            "no triggers -- it's been the same in all seasons",
        ],
        "border_shape": [
            "with smooth, perfectly regular borders",
            "with an even, clearly defined edge",
            "with symmetric, clean borders",
        ],
    },

    # ─── MELANOMA ────────────────────────────────────────────────────────────
    # Tier 2 neighbors: Melanocytic Nevi, Basal Cell Carcinoma
    "Melanoma": {
        "discriminators": [
            "a mole with an asymmetrical, uneven shape -- one half doesn't match the other",
            "a spot with jagged, notched, irregular borders",
            "a mole with multiple colors -- brown, black, red, and even some pale areas",
            "a mole that has been visibly growing and changing over the past few months",
            "a mole that started bleeding on its own without any injury",
            "a spot that's larger than a pencil eraser and keeps getting bigger",
        ],
        "location": [
            "on my back", "on my arm", "on my leg",
            "on my chest", "on my shoulder", "on my scalp",
            "on my face",
        ],
        "appearance": [
            "a mole with an uneven, asymmetrical shape",
            "a spot with irregular, notched borders",
            "a mole with several colors mixed in",
            "a dark spot that's been changing",
            "a dark brown spot",                # Tier 2 shared with Nevi
            "a pigmented spot",                 # Tier 2 shared with Nevi
            "a growth that bleeds",             # Tier 2 shared with BCC
        ],
        "texture": [
            "slightly raised and uneven",
            "partly flat and partly raised within the same spot",
            "feels different from my other moles",
        ],
        "sensation": [
            "itching and occasional soreness around it",
            "no real pain, but it bleeds a little sometimes",
            "a strange new sensation I never noticed before",
        ],
        "evolution": [
            "has been changing shape and color over the last few months",
            "started as a small mole and has grown noticeably bigger",
            "has started bleeding and crusting on its own",
            "looks completely different compared to a year ago",
        ],
        "trigger": [
            "it started changing after a lot of sun exposure",
        ],
        "border_shape": [
            "with irregular, notched, uneven borders",
            "with jagged edges that look ragged",
            "with an asymmetric, non-circular outline",
        ],
    },

    # ─── BASAL CELL CARCINOMA ────────────────────────────────────────────────
    # Tier 2 neighbor: Melanoma
    "Basal Cell Carcinoma": {
        "discriminators": [
            "a shiny, pearly bump with tiny visible blood vessels running through it",
            "an open sore that bleeds, scabs over, and then reopens a few weeks later",
            "a waxy, scar-like flat patch with an indistinct border",
            "a bump that keeps bleeding even after a very minor bump or scrape",
            "a pink growth with a slightly sunken, crater-like center",
        ],
        "location": [
            "on my nose", "on my ear", "on my forehead", "on my cheek",
            "on my neck", "on my shoulder", "on my scalp",
        ],
        "appearance": [
            "a small, shiny, pearly bump",
            "an open sore that won't fully heal",
            "a flat, reddish, scaly patch",
            "a waxy, scar-like area",
            "a pink bump with a sunken center",
            "a growth that bleeds",             # Tier 2 shared with Melanoma
        ],
        "texture": [
            "smooth and shiny on the surface",
            "waxy and scar-like",
            "the center feels soft and tends to crust over",
        ],
        "sensation": [
            "no real pain, just occasional bleeding after a minor bump",
            "mild irritation but nothing painful",
            "no discomfort, it just won't go away",
        ],
        "evolution": [
            "keeps bleeding, scabbing over, and then reopening",
            "has been slowly growing over several months",
            "hasn't healed in over two months despite repeated scabbing",
            "the scab keeps falling off and the raw area underneath comes back",
        ],
        "trigger": [
            "it started on an area that gets a lot of sun",
        ],
        "border_shape": [
            "with a rolled, slightly raised outer edge",
            "with translucent, blurry borders",
            "with an indistinct, difficult-to-define edge",
        ],
    },

    # ─── SEBORRHEIC KERATOSES ────────────────────────────────────────────────
    # Tier 1 neighbor: Benign Keratosis-like Lesions
    # Tier 3 neighbor: Viral Infections
    "Seborrheic Keratoses": {
        "discriminators": [
            "a waxy, 'stuck-on' looking growth that appears pasted onto the skin surface",
            "a brown to dark brown raised plaque with a greasy, waxy feel",
            "a growth that looks like it could be lifted off with a fingernail",
            "a warty, barnacle-like bump with a rough, pitted surface",
            "a tan to dark brown growth that looks thicker and greasier than a normal mole",
        ],
        "location": [
            "on my chest", "on my back", "on my face",
            "on my scalp", "on my shoulder", "on my neck",
            "on my trunk",
        ],
        "appearance": [
            "a waxy, stuck-on looking growth",
            "a brown to dark brown raised patch",
            "a round, well-defined growth",
            "a scaly, wart-like bump",
            "a raised, well-defined bump",      # Tier 3 shared with Viral
            "a rough-surfaced growth",          # Tier 1 shared with BKL
        ],
        "texture": [
            "waxy and slightly greasy to the touch",
            "feels like it's sitting on top of the skin rather than growing from it",
            "rough and bumpy on the surface",
            "rough and flaky",                  # Tier 1 shared with BKL
        ],
        "sensation": [
            "no pain, just occasional mild itching if clothing rubs it",
            "mild irritation when it catches on something",
            "no discomfort unless I scratch it",
        ],
        "evolution": [
            "has been slowly growing thicker over several years",
            "stayed the same size for a long time but got darker recently",
            "grew alongside several similar spots in the same area",
        ],
        "trigger": [
            "no particular trigger, it just appeared gradually",
        ],
        "border_shape": [
            "with sharp, well-defined, stuck-on looking edges",
            "with a clearly defined, elevated border",
            "with distinct, punched-out looking edges",
        ],
    },

    # ─── BENIGN KERATOSIS-LIKE LESIONS ───────────────────────────────────────
    # Tier 1 neighbor: Seborrheic Keratoses
    "Benign Keratosis-like Lesions": {
        "discriminators": [
            "a rough, sandpaper-like patch that feels much rougher than it looks",
            "a dry, crusty spot on a sun-damaged area that keeps peeling off and returning",
            "a flat, pink-red scaly patch with a gritty texture on my sun-exposed skin",
            "a slightly raised rough patch that bleeds lightly when I scratch it",
            "a scaly spot that's tender when I rub it firmly",
        ],
        "location": [
            "on my face", "on my ears", "on the back of my hands",
            "on my forearms", "on my scalp", "on my lower lip",
            "on my bald scalp",
        ],
        "appearance": [
            "a rough, sandpaper-like patch",
            "a small, dry, crusty spot",
            "a flat, slightly rough patch darker than the surrounding skin",
            "a scaly, reddish-pink patch",
            "a rough-surfaced growth",          # Tier 1 shared with SK
        ],
        "texture": [
            "rough and gritty like sandpaper",
            "feels dry and crusty when I run my finger over it",
            "rough and flaky",                  # Tier 1 shared with SK
            "feels tender when pressed",
        ],
        "sensation": [
            "mild stinging or burning feeling",
            "occasional itching",
            "tenderness when I press on it",
            "no pain, just a rough texture I can feel",
        ],
        "evolution": [
            "keeps coming back in the same spot after it peels off",
            "has been there for months without healing completely",
            "seems to be slowly getting larger",
            "the crust falls off and then regrows",
        ],
        "trigger": [
            "on areas that get a lot of sun exposure",
        ],
        "border_shape": [
            "with ill-defined, gradual borders",
            "with a somewhat irregular, flat border",
            "with blurry edges that fade into normal skin",
        ],
    },
}

# ============================================================================
# 3. SENTENCE TEMPLATES
#    65 templates across 7 style categories.
#    Each template references a subset of slots (2-6) with variable ordering.
#    Variables used: {appearance} {location} {texture} {sensation}
#                   {evolution} {trigger} {border_shape}
#    Also special: {Location_cap} = location with capital letter stripped of "on"
# ============================================================================

TEMPLATES = {

    # --- Category 1: Simple declarative (9 templates, 2-3 slots) ---
    "declarative": [
        "I have {appearance} {location}.",
        "I've got {appearance} {location}.",
        "There's {appearance} {location}.",
        "I'm noticing {appearance} {location}.",
        "I have {appearance} {location}, and {texture}.",
        "There's {appearance} {location}, and {sensation}.",
        "I have {sensation}, and there's {appearance} {location}.",
        "{appearance} showed up {location} a while back.",
        "I've noticed {appearance} {location} lately.",
    ],

    # --- Category 2: Compound descriptive (12 templates, 3-4 slots) ---
    "compound": [
        "I have {appearance} {location} with {texture}.",
        "I've had {appearance} {location} for a while, with {sensation}.",
        "I noticed {appearance} {location}, along with {sensation}.",
        "There's {appearance} {location} -- {texture} -- and {sensation}.",
        "The skin {location} shows {appearance}, and I've been dealing with {sensation}.",
        "I developed {appearance} {location} that comes with {sensation}.",
        "There's {appearance} {location} {border_shape}, and {texture}.",
        "I have {appearance} {location} {border_shape}.",
        "I've had {appearance} {location} that {evolution}.",
        "There's {appearance} {location} -- {texture} -- that {evolution}.",
        "I have {sensation}, along with {appearance} {location} {border_shape}.",
        "Lately I've had {appearance} {location} that comes with {sensation}.",
    ],

    # --- Category 3: Multi-sentence narrative (14 templates, 4-6 slots) ---
    "narrative": [
        "{appearance} {location}. {evolution}. {sensation}.",
        "I've had {appearance} {location} for some time. {texture}. {sensation}.",
        "I noticed {appearance} {location}. {texture}. {evolution}.",
        "There's {appearance} {location} {border_shape}. {texture}. {sensation}.",
        "The skin {location} has changed. There's {appearance} now. {texture}. {evolution}.",
        "I have {appearance} {location}. {texture}. {evolution}. {sensation}.",
        "Something's going on {location}. It's {appearance}. {texture}. {sensation}.",
        "I've had {sensation} for a while now. There's {appearance} {location} {border_shape}.",
        "It started as {appearance} {location}. Now {texture}. {evolution}.",
        "{appearance} appeared {location} a while ago. {border_shape}. {texture}. {evolution}.",
        "My skin {location} has been off. {appearance}. {texture}. {sensation}.",
        "There's a patch {location}. It looks like {appearance}. {texture}. {evolution}.",
        "I've noticed {sensation} and there's {appearance} {location}. {border_shape}. {evolution}.",
        "Something showed up {location}. It's {appearance} {border_shape}. {texture}. {sensation}.",
    ],

    # --- Category 4: Casual / informal (12 templates, 2-4 slots) ---
    "casual": [
        "so I've got {appearance} {location}, with {sensation}",
        "got {appearance} {location} and {evolution}",
        "there's {appearance} {location}, {texture}, not sure what it is",
        "my skin {location} is doing something weird -- {appearance}",
        "I have {appearance} {location} and honestly {evolution}",
        "so {location} there's {appearance}, and {sensation}",
        "{appearance} showed up {location} and {evolution}",
        "weird {appearance} {location}, {texture}, no clue what it is",
        "got {appearance} {location}, {texture}",
        "something {location} looks like {appearance} and {evolution}",
        "I've got {appearance} {location} -- {texture} -- and {sensation}",
        "{appearance} {location}, {sensation}, been there ages",
    ],

    # --- Category 5: Fragment / shorthand (7 templates, 2 slots) ---
    "fragment": [
        "{appearance} {location}. {sensation}.",
        "{appearance} {location}, {sensation}.",
        "{location}: {appearance}. {evolution}.",
        "{appearance}, {location}, {texture}.",
        "{sensation} -- {appearance} {location}.",
        "{appearance} {location}. {texture}.",
        "{appearance} {location}. {border_shape}.",
    ],

    # --- Category 6: Concern-driven (6 templates, 3-5 slots) ---
    "concern": [
        "I'm worried about {appearance} {location} because {evolution}.",
        "I'm not sure what {appearance} {location} is -- {texture} and {sensation}.",
        "I'm concerned about {appearance} {location} that {evolution} -- {sensation}.",
        "Something doesn't look right {location}: {appearance} {border_shape}. {texture}.",
        "I've been watching {appearance} {location} and {evolution}. {sensation}.",
        "Should I be worried about {appearance} {location}? {texture}. {evolution}.",
    ],

    # --- Category 7: Trigger-contextual (5 templates, 3-5 slots) ---
    "trigger_context": [
        "{trigger}, I get {appearance} {location}.",
        "{trigger}, {appearance} {location} gets worse.",
        "I notice {sensation} {trigger}. There's {appearance} {location}.",
        "{appearance} {location} flares {trigger}.",
        "{trigger}, {sensation} and {appearance} {location} appear.",
    ],
}

# Flatten templates into a list of (category, template) pairs for weighted sampling
ALL_TEMPLATES = []
CATEGORY_WEIGHTS = {
    "declarative": 1.0,
    "compound": 1.2,
    "narrative": 1.2,
    "casual": 1.0,
    "fragment": 0.7,
    "concern": 0.8,
    "trigger_context": 0.6,
}
for cat, tmpls in TEMPLATES.items():
    weight = CATEGORY_WEIGHTS[cat]
    for t in tmpls:
        ALL_TEMPLATES.append((cat, t, weight))

# ============================================================================
# 4. DIFFICULTY-LEVEL CONFIGURATION
#    Controls the mix of discriminator vs. shared vocabulary per sentence.
# ============================================================================

# Probability of pulling a phrase from discriminators (vs. class-specific pool)
DIFFICULTY_CONFIG = {
    "easy":      {"discriminator_prob": 0.80, "share_global_prob": 0.05, "fraction": 0.20},
    "medium":    {"discriminator_prob": 0.45, "share_global_prob": 0.20, "fraction": 0.40},
    "hard":      {"discriminator_prob": 0.20, "share_global_prob": 0.40, "fraction": 0.30},
    "ambiguous": {"discriminator_prob": 0.00, "share_global_prob": 1.00, "fraction": 0.10},
}

# ============================================================================
# 5. GENERATOR
# ============================================================================

def pick_phrase(slot, class_name, discriminator_prob, share_global_prob):
    """
    For a given slot category, picks a phrase using the difficulty-driven
    weighted-random selection:
      - With probability `discriminator_prob` (ONLY for the `appearance` slot):
        draw from the class's discriminator list.
      - With probability `share_global_prob`: draw from the global shared pool.
      - Otherwise: draw from the class-specific per-slot vocabulary.

    Discriminators are always full clinical presentation phrases (appearance-
    type text). Injecting them into non-appearance slots (e.g. location,
    sensation) produces grammatically broken sentences -- so they are gated
    to the appearance slot only.
    """
    class_vocab = VOCAB[class_name]
    roll = random.random()

    # Only inject a discriminator when filling the appearance slot
    if slot == "appearance" and roll < discriminator_prob and class_vocab.get("discriminators"):
        return random.choice(class_vocab["discriminators"])

    if roll < discriminator_prob + share_global_prob and slot in GLOBAL:
        candidates = GLOBAL[slot]
        if candidates:
            return random.choice(candidates)

    # Class-specific per-slot vocabulary
    if slot in class_vocab and class_vocab[slot]:
        return random.choice(class_vocab[slot])

    # Ultimate fallback: global pool
    if slot in GLOBAL and GLOBAL[slot]:
        return random.choice(GLOBAL[slot])

    return ""


def fill_template(template, class_name, discriminator_prob, share_global_prob):
    """
    Fills all {slot} placeholders in a template with phrases drawn according
    to the current difficulty configuration.
    Returns None if any required slot produces an empty string.
    """
    import re
    slots_needed = re.findall(r"\{(\w+)\}", template)
    filled = template

    for slot in slots_needed:
        phrase = pick_phrase(slot, class_name, discriminator_prob, share_global_prob)
        if not phrase:
            return None
        filled = filled.replace("{" + slot + "}", phrase, 1)

    # Capitalize first character
    return filled[0].upper() + filled[1:] if filled else None


def generate_class_sentences(class_name, target_n):
    """
    Generates `target_n` unique sentences for a given class, distributing
    across difficulty levels as specified in DIFFICULTY_CONFIG.
    """
    difficulty_targets = {}
    remaining = target_n
    for i, (level, cfg) in enumerate(DIFFICULTY_CONFIG.items()):
        if i == len(DIFFICULTY_CONFIG) - 1:
            difficulty_targets[level] = remaining
        else:
            n = int(cfg["fraction"] * target_n)
            difficulty_targets[level] = n
            remaining -= n

    sentences = []
    seen = set()

    for level, count in difficulty_targets.items():
        cfg = DIFFICULTY_CONFIG[level]
        d_prob = cfg["discriminator_prob"]
        g_prob = cfg["share_global_prob"]
        attempts = 0
        level_sentences = []

        while len(level_sentences) < count and attempts < MAX_ATTEMPTS:
            attempts += 1
            _, template, _ = random.choice(ALL_TEMPLATES)
            sentence = fill_template(template, class_name, d_prob, g_prob)
            if sentence and sentence not in seen:
                seen.add(sentence)
                level_sentences.append(sentence)

        if len(level_sentences) < count:
            print(f"  WARNING [{class_name}][{level}]: only generated "
                  f"{len(level_sentences)}/{count} unique sentences.")

        sentences.extend(level_sentences)

    return sentences



def main():
    rows = []
    for class_name in VOCAB:
        sentences = generate_class_sentences(class_name, TARGET_PER_CLASS)
        print(f"  {class_name}: {len(sentences)} sentences")
        for s in sentences:
            rows.append((s, class_name))

    random.shuffle(rows)

    out_path = "real_symptoms_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print("\nSample rows per class:")
    per_class = {}
    for text, label in rows:
        per_class.setdefault(label, []).append(text)
    for cls in sorted(per_class):
        sample = random.choice(per_class[cls])
        print(f"  [{cls}] {sample[:120]}")


if __name__ == "__main__":
    main()
