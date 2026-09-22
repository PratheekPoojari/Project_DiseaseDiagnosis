import json
import random
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import os

BASE_DIR = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_DIR, 'core_clinical_truth.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'expanded_symptoms_dataset.csv')

# Target sentences per class
TARGET = 200

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
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.isalnum()]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: # only replace up to n words
            break

    sentence = ' '.join(new_words)
    return sentence

def augment_sentence(sentence):
    words = word_tokenize(sentence)
    # Replace roughly 30% of the words with synonyms
    n = max(1, int(len(words) * 0.3))
    return synonym_replacement(words, n)

if __name__ == '__main__':
    with open(INPUT_PATH, 'r') as f:
        core_truth = json.load(f)
        
    dataset_records = []
    
    for disease, text in core_truth.items():
        print(f"Augmenting: {disease}")
        # Split the core truth paragraph into individual sentences
        base_sentences = sent_tokenize(text)
        
        # If the web scraper failed to get anything, put a dummy sentence so it doesn't crash
        if not base_sentences:
            base_sentences = [f"{disease} presents with skin lesions and rash."]
            
        # Add the original sentences first
        for s in base_sentences:
            dataset_records.append({'text': s, 'label': disease})
            
        # Loop until we hit our target (200 variations per class)
        while len([r for r in dataset_records if r['label'] == disease]) < TARGET:
            base_s = random.choice(base_sentences)
            new_s = augment_sentence(base_s)
            dataset_records.append({'text': new_s, 'label': disease})
            
    df = pd.DataFrame(dataset_records)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Created local NLP dataset with {len(df)} records at {OUTPUT_PATH}")
    print("Zero API calls used.")
