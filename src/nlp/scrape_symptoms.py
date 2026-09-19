"""
File: src/nlp/scrape_symptoms.py
Purpose: Scrapes real-world medical symptom descriptions for all 10 skin disease classes
         from multiple reputable medical sources (Wikipedia, MedlinePlus, NHS, Healthline, WebMD).
Why we need it: The NLP branch must be reliable for text-only diagnosis (no image). 
                Replacing synthetic/augmented data with genuinely diverse real-world 
                sentences across multiple sources dramatically improves the embedding 
                model's semantic understanding of how each disease is described.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import os
import re
import time

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_DIR  = "data/nlp"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "real_symptoms.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; DiseaseDiagnosisBot/1.0; academic research)'
}

# Per-class multi-source URL mapping.
# Each class gets URLs from up to 5 different medical sources to ensure
# vocabulary diversity (clinical Wikipedia vs. patient-facing NHS/Healthline).
CLASS_SOURCES = {
    "eczema": [
        "https://en.wikipedia.org/wiki/Eczema",
        "https://en.wikipedia.org/wiki/Dermatitis",
        "https://medlineplus.gov/eczema.html",
        "https://www.nhs.uk/conditions/atopic-eczema/",
        "https://www.healthline.com/health/eczema",
    ],
    "melanoma": [
        "https://en.wikipedia.org/wiki/Melanoma",
        "https://medlineplus.gov/melanoma.html",
        "https://www.nhs.uk/conditions/melanoma-skin-cancer/",
        "https://www.healthline.com/health/melanoma",
        "https://www.webmd.com/melanoma-skin-cancer/what-is-melanoma",
    ],
    "atopic_dermatitis": [
        "https://en.wikipedia.org/wiki/Atopic_dermatitis",
        "https://medlineplus.gov/atopicdermatitis.html",
        "https://www.nhs.uk/conditions/atopic-eczema/symptoms/",
        "https://www.healthline.com/health/atopic-dermatitis",
        "https://www.webmd.com/skin-problems-and-treatments/eczema/atopic-dermatitis-overview",
    ],
    "basal_cell_carcinoma": [
        "https://en.wikipedia.org/wiki/Basal-cell_carcinoma",
        "https://medlineplus.gov/ency/article/000824.htm",
        "https://www.nhs.uk/conditions/basal-cell-skin-cancer/",
        "https://www.healthline.com/health/basal-cell-carcinoma",
        "https://www.webmd.com/melanoma-skin-cancer/basal-cell-carcinoma",
    ],
    "melanocytic_nevi": [
        "https://en.wikipedia.org/wiki/Melanocytic_nevus",
        "https://medlineplus.gov/ency/article/000826.htm",
        "https://www.healthline.com/health/common-mole",
        "https://www.webmd.com/skin-problems-and-treatments/moles-freckles-skin-tags",
        "https://en.wikipedia.org/wiki/Nevus",
        # Targeted additions for the mole/nevus vocabulary gap
        "https://en.wikipedia.org/wiki/Blue_nevus",
        "https://en.wikipedia.org/wiki/Spitz_nevus",
    ],
    "benign_keratosis": [
        "https://en.wikipedia.org/wiki/Seborrheic_keratosis",
        "https://medlineplus.gov/ency/article/000884.htm",
        "https://www.nhs.uk/conditions/seborrhoeic-keratosis/",
        "https://www.healthline.com/health/seborrheic-keratosis",
        "https://www.webmd.com/skin-problems-and-treatments/seborrheic-keratoses",
        # Targeted additions — related benign keratosis-family conditions
        "https://my.clevelandclinic.org/health/diseases/21721-seborrheic-keratosis",
        "https://en.wikipedia.org/wiki/Actinic_keratosis",
        "https://en.wikipedia.org/wiki/Keratosis_pilaris",
        "https://en.wikipedia.org/wiki/Solar_lentigo",
    ],
    "psoriasis_lichen_planus": [
        "https://en.wikipedia.org/wiki/Psoriasis",
        "https://en.wikipedia.org/wiki/Lichen_planus",
        "https://medlineplus.gov/psoriasis.html",
        "https://www.nhs.uk/conditions/psoriasis/",
        "https://www.healthline.com/health/psoriasis",
    ],
    "seborrheic_keratoses": [
        "https://en.wikipedia.org/wiki/Seborrheic_keratosis",
        "https://medlineplus.gov/ency/article/000884.htm",
        "https://www.healthline.com/health/seborrheic-keratosis",
        "https://www.webmd.com/skin-problems-and-treatments/seborrheic-keratoses",
        "https://en.wikipedia.org/wiki/Acanthosis",
        # Targeted additions for seborrheic keratosis related conditions
        "https://my.clevelandclinic.org/health/diseases/21721-seborrheic-keratosis",
        "https://en.wikipedia.org/wiki/Stucco_keratosis",
        "https://en.wikipedia.org/wiki/Dermatosis_papulosa_nigra",
    ],
    "fungal_infections": [
        "https://en.wikipedia.org/wiki/Dermatophytosis",
        "https://en.wikipedia.org/wiki/Tinea_corporis",
        "https://en.wikipedia.org/wiki/Athlete%27s_foot",
        "https://medlineplus.gov/fungalinfections.html",
        "https://www.nhs.uk/conditions/ringworm/",
        "https://www.healthline.com/health/ringworm",
    ],
    "viral_infections": [
        "https://en.wikipedia.org/wiki/Molluscum_contagiosum",
        "https://en.wikipedia.org/wiki/Wart",
        "https://medlineplus.gov/warts.html",
        "https://www.nhs.uk/conditions/warts-and-verrucas/",
        "https://www.nhs.uk/conditions/molluscum-contagiosum/",
        "https://www.healthline.com/health/molluscum-contagiosum",
        # Targeted additions for the viral wart / HPV vocabulary
        "https://en.wikipedia.org/wiki/Human_papillomavirus_infection",
        "https://en.wikipedia.org/wiki/Condyloma",
    ],
}

TARGET_PER_CLASS = 450
MIN_SENTENCE_LENGTH = 30  # Filter out fragments that are too short to be meaningful

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def clean_text(text: str) -> str:
    """
    Strips Wikipedia citation markers, HTML artifacts, and excess whitespace.
    Why we need it: Raw scraped text contains noise like [1], [edit], \xa0 etc.
                    that pollutes the sentence encoder's vocabulary.
    """
    text = re.sub(r'\[\d+\]', '', text)         # Wikipedia citations [1], [23]
    text = re.sub(r'\[edit\]', '', text)        # Wikipedia section edit links
    text = re.sub(r'\s+', ' ', text)            # Multiple spaces/newlines → single space
    text = text.replace('\xa0', ' ').strip()    # Non-breaking spaces
    return text


def fetch_page_sentences(url: str) -> list:
    """
    Fetches a single URL, extracts all meaningful paragraph text, and splits into sentences.
    Why we need it: Each source describes the same disease differently — Wikipedia is clinical,
                    NHS is patient-facing, Healthline is conversational. Pulling from all 
                    sources gives the model diverse vocabulary for the same condition.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove navigation, footer, script, and style tags — we only want article body
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        paragraphs = soup.find_all('p')
        raw_text = ' '.join(p.get_text() for p in paragraphs)
        cleaned = clean_text(raw_text)

        sentences = sent_tokenize(cleaned)
        # Filter: must be long enough to carry medical meaning, must not be just a URL
        valid = [s.strip() for s in sentences
                 if len(s.strip()) >= MIN_SENTENCE_LENGTH and 'http' not in s]
        return valid

    except Exception as e:
        print(f"    [!] Failed to fetch {url}: {e}")
        return []


def scrape_all_classes() -> pd.DataFrame:
    """
    Iterates over all 10 classes and their source URLs, accumulating up to TARGET_PER_CLASS 
    real sentences per class, stopping early once the target is reached.
    Why we need it: Guarantees balanced, real-world data across all classes without 
                    wasting time scraping more than needed.
    """
    all_data = []

    for label, urls in CLASS_SOURCES.items():
        print(f"\n  [{label}] Scraping up to {TARGET_PER_CLASS} sentences...")
        class_sentences = []
        seen = set()  # Deduplicate across sources

        for url in urls:
            if len(class_sentences) >= TARGET_PER_CLASS:
                break  # Already have enough — skip remaining sources

            print(f"    Fetching: {url}")
            sentences = fetch_page_sentences(url)

            for sentence in sentences:
                if sentence not in seen and len(class_sentences) < TARGET_PER_CLASS:
                    seen.add(sentence)
                    class_sentences.append(sentence)

            time.sleep(1)  # Polite delay between requests to avoid rate limiting

        print(f"    Collected: {len(class_sentences)} sentences")
        for s in class_sentences:
            all_data.append({'text': s, 'label': label})

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting multi-source medical symptom scraping...")
    print(f"Target: {TARGET_PER_CLASS} real sentences per class across {len(CLASS_SOURCES)} classes\n")

    df = scrape_all_classes()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} total sentences to {OUTPUT_PATH}")
    print("\nFinal class distribution:")
    print(df['label'].value_counts())
