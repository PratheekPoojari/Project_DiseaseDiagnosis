"""
File: src/nlp/augment_data.py
Purpose: Augments the scraped medical dataset with two complementary strategies:
         1. Synonym replacement to expand all classes to 600 samples.
         2. Conversational paraphrasing to bridge the gap between formal encyclopedic 
            training language and how real users actually describe their symptoms.
Why we need it: Ensures balanced classes for the TF-IDF + SVM model to achieve 87%+ accuracy.
"""

import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
import os

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

TARGET_SAMPLES_PER_CLASS = 600
INPUT_PATH  = "data/nlp/real_symptoms.csv"
OUTPUT_PATH = "data/nlp/augmented_symptoms.csv"

CLINICAL_TO_CASUAL = {
    "pruritus": "itching", "erythema": "redness", "edema": "swelling",
    "lesion": "spot", "macule": "flat spot", "papule": "bump",
    "vesicle": "blister", "pustule": "pus-filled bump", "exudate": "oozing",
    "desquamation": "peeling", "alopecia": "hair loss", "urticaria": "hives",
    "hyperhidrosis": "excessive sweating", "purpura": "purple spots",
    "ecchymosis": "bruising", "neoplasm": "growth", "benign": "harmless",
    "malignant": "cancerous", "dermatitis": "skin irritation", "asymptomatic": "no symptoms",
    "exacerbation": "flare-up", "idiopathic": "unknown cause", "chronic": "long-lasting",
    "acute": "sudden", "topical": "on the skin", "systemic": "whole body",
    "localized": "in one area", "diffuse": "widespread", "congenital": "present at birth",
    "acquired": "developed over time"
}

def clinical_to_conversational(text):
    words = text.split()
    new_words = [CLINICAL_TO_CASUAL.get(word.lower(), word) for word in words]
    return " ".join(new_words)

CONVERSATIONAL_PREFIXES = [
    "I noticed ", "My skin has ", "I have this ", "There is a ",
    "I woke up with ", "Recently developed ", "I've been experiencing ",
    "My doctor said it might be ", "It looks like ", "Feels like "
]

def add_conversational_prefix(text):
    return random.choice(CONVERSATIONAL_PREFIXES) + text.lower()

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(words, n):
    words = words.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in nltk.corpus.stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def augment_dataset(df):
    print(f"Augmenting data to reach {TARGET_SAMPLES_PER_CLASS} samples per class...")
    augmented_data = []
    classes = df['label'].unique()

    for cls in classes:
        class_df = df[df['label'] == cls]
        count = len(class_df)
        print(f"  Class: {cls} (Original: {count})")
        original_sentences = class_df['text'].tolist()

        for text in original_sentences:
            augmented_data.append({'text': text, 'label': cls})

        conv_sentences_to_add = min(len(original_sentences), 200)
        for sentence in random.sample(original_sentences, conv_sentences_to_add):
            paraphrased = clinical_to_conversational(sentence)
            if random.random() < 0.5:
                paraphrased = add_conversational_prefix(paraphrased)
            augmented_data.append({'text': paraphrased, 'label': cls})

        current_count = count + conv_sentences_to_add
        if current_count < TARGET_SAMPLES_PER_CLASS:
            needed = TARGET_SAMPLES_PER_CLASS - current_count
            for _ in range(needed):
                base_sentence = random.choice(original_sentences)
                new_sentence = synonym_replacement(base_sentence, n=random.randint(1, 3))
                augmented_data.append({'text': new_sentence, 'label': cls})

    result_df = pd.DataFrame(augmented_data)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return result_df

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        exit(1)
    df = pd.read_csv(INPUT_PATH)
    augmented_df = augment_dataset(df)
    augmented_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(augmented_df)} augmented sentences to {OUTPUT_PATH}")
