# NLP Branch Synopsis: Symptom Text Classification

## 1. What We Did: The Data Engineering Journey
Our goal was to build an NLP model capable of reading a patient's raw symptom description and classifying it into one of 10 skin disease classes. 

We ran extensive data engineering experiments:
1. **Targeted Scraping:** We built a multi-source web scraper to pull real medical sentences from Wikipedia, MedlinePlus, NHS, WebMD, and Healthline, yielding ~3,500 highly relevant sentences.
2. **The "Big Data" Trap:** We attempted to aggressively scrape the internet for 32,000+ sentences to feed a dense semantic embedding model. This caused accuracy to collapse to **48%**. We proved the **"Garbage In, Garbage Out"** principle: scraping massive amounts of data pulled in noisy, irrelevant text (epidemiology, history, doctor biographies) which destroyed the model's ability to learn actual symptoms.
3. **Data Augmentation:** We reverted to our clean 3,500-sentence dataset and used **NLTK Synonym Replacement** and **Conversational Paraphrasing** (mapping clinical words like "erythema" to "redness") to perfectly balance every class to 600 samples.

## 2. What We Ended Up With: Final Architecture
Because our dataset is highly targeted but relatively small (6,150 sentences total), dense embedding models (like SentenceTransformers) struggled to find decision boundaries. 

We pivoted to a sparse data architecture, which is mathematically optimized for this exact scenario:
*   **Vectorization:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency). This algorithm ignores common noise words (like "the", "doctor", "patient") and assigns high mathematical weights to words that strongly predict a specific class (like "melanocyte", "fungal", "oozing").
*   **Algorithm:** `LinearSVC` (Support Vector Machine).
*   **Calibration:** `CalibratedClassifierCV` wrapped around the SVM.

## 3. The Output: Probabilities vs. Hard Labels
The model does **not** return a hard label (e.g., it does not just say "Melanoma"). Instead, it returns a **Probability Dictionary**.

**Example Output:**
```json
{
  "melanoma": 0.85,
  "melanocytic_nevi": 0.12,
  "basal_cell_carcinoma": 0.02,
  "eczema": 0.01,
  ... (remaining classes close to 0.0)
}
```

**Why this is critical:** 
A standard SVM only outputs a geometric "distance" from a mathematical boundary. By wrapping our SVM in `CalibratedClassifierCV`, we force it to convert that distance into true percentage-based probabilities (Platt Scaling). 
We **must** have probabilities for the upcoming **Fusion Layer**. When the CNN processes the skin image, it will also output probabilities. The Fusion Layer will literally do mathematical weighting (e.g., `(NLP_prob * 0.4) + (CNN_prob * 0.6)`) to make the final, combined prediction. If the NLP model just returned the word "Melanoma", the Fusion Layer couldn't do math on it.

## 4. Final Performance Metrics
**Overall Accuracy: 81.06%**

| Disease Class | Precision | Recall | F1-Score | Status |
|---------------|-----------|--------|----------|--------|
| Fungal Infections | 90% | 93% | 91% | ✅ Excellent |
| Melanocytic Nevi (Moles) | 91% | 88% | 90% | ✅ Excellent |
| Psoriasis / Lichen Planus | 89% | 88% | 89% | ✅ Excellent |
| Basal Cell Carcinoma | 90% | 85% | 88% | ✅ Excellent |
| Viral Infections | 89% | 88% | 88% | ✅ Excellent |
| Melanoma (Cancer) | 87% | 87% | 87% | ✅ Excellent |
| Eczema | 78% | 86% | 82% | ✅ Strong |
| Atopic Dermatitis | 84% | 78% | 81% | ✅ Strong |
| Seborrheic Keratoses | 56% | 62% | 59% | ⚠️ Confused with BK |
| Benign Keratosis (BK) | 57% | 54% | 55% | ⚠️ Confused with SK |

*Note on the limitation:* The only classes dragging the accuracy down to 81% are Benign Keratosis and Seborrheic Keratosis. Medically, Seborrheic Keratosis is literally a sub-type of benign keratosis. The model is getting them confused because they share identical symptom vocabularies, but this is a clinically safe confusion (a harmless growth being confused for another harmless growth). The critical cancer classes (Melanoma, BCC) are scoring near 90%.
