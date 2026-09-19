"""
File: src/nlp/train_nlp_model.py
"""
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/nlp/augmented_symptoms.csv"
MODEL_DIR = "models/nlp"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_tfidf_model.pkl")

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    df.dropna(subset=['text', 'label'], inplace=True)
    return df['text'], df['label']

def build_and_train_model(X_train, y_train):
    print("Building TF-IDF and SVM Pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
        ('svm', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42), cv=3))
    ])
    print("Training the SVM model...")
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating Model on Test Data...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions))

def save_model(model, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nModel successfully saved to {output_path}")

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    
    model = build_and_train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)
