import requests
from bs4 import BeautifulSoup
import json
import os

# Map our 10 specific classes to DermNet NZ URLs
DISEASE_URLS = {
    'Eczema': 'https://dermnetnz.org/topics/atopic-dermatitis', # General eczema
    'Atopic Dermatitis': 'https://dermnetnz.org/topics/atopic-dermatitis',
    'Melanoma': 'https://dermnetnz.org/topics/melanoma',
    'Basal Cell Carcinoma': 'https://dermnetnz.org/topics/basal-cell-carcinoma',
    'Melanocytic Nevi': 'https://dermnetnz.org/topics/melanocytic-naevus',
    'Benign Keratosis-like Lesions': 'https://dermnetnz.org/topics/seborrhoeic-keratosis', # BKL typically includes SK
    'Psoriasis Lichen Planus': 'https://dermnetnz.org/topics/psoriasis',
    'Seborrheic Keratoses': 'https://dermnetnz.org/topics/seborrhoeic-keratosis',
    'Fungal Infections': 'https://dermnetnz.org/topics/tinea-corporis',
    'Viral Infections': 'https://dermnetnz.org/topics/viral-wart'
}

# The headers we care about (pure clinical truth)
TARGET_HEADERS = [
    'What are the clinical features',
    'clinical features',
    'symptoms',
    'what does it look like',
    'physical examination'
]

def clean_text(text):
    return ' '.join(text.split()).strip()

def scrape_disease(url):
    print(f"Scraping: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        extracted_text = []
        
        # DermNet articles are usually inside a div with class 'articleText' or 'bodyText'
        # Let's iterate through headers
        for h2 in soup.find_all(['h2', 'h3']):
            header_text = clean_text(h2.get_text().lower())
            
            # Check if this header is about symptoms
            if any(target in header_text for target in TARGET_HEADERS):
                # Grab all siblings until the next header
                for sibling in h2.find_next_siblings():
                    if sibling.name in ['h2', 'h3']:
                        break
                    
                    # Grab paragraphs and list items
                    if sibling.name == 'p':
                        txt = clean_text(sibling.get_text())
                        if txt:
                            extracted_text.append(txt)
                    elif sibling.name == 'ul':
                        for li in sibling.find_all('li'):
                            txt = clean_text(li.get_text())
                            if txt:
                                extracted_text.append(txt)
                                
        return " ".join(extracted_text)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

if __name__ == '__main__':
    dataset = {}
    
    for disease, url in DISEASE_URLS.items():
        text = scrape_disease(url)
        dataset[disease] = text
        print(f"Extracted {len(text)} characters for {disease}\n")
        
    output_path = os.path.join(os.path.dirname(__file__), 'core_clinical_truth.json')
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Saved core truth to {output_path}")
