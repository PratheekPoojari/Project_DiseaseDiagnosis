# NLP Branch Experiments & Findings

## The Goal
To build a highly accurate text classifier that maps patient symptom descriptions to 10 skin disease classes. 

## The Iterations & Results

| Strategy | Architecture | Dataset Size | Data Quality / Noise | Accuracy |
|----------|--------------|--------------|-----------------------|----------|
| 1. Baseline | TF-IDF + LinearSVC | 6,000 (1,600 real + aug) | High (Wikipedia only) | **87.0%** |
| 2. Embeddings | all-MiniLM-L6-v2 + SVM | 6,000 (1,600 real + aug) | High | 78.0% |
| 3. Clean Real Data | all-MiniLM-L6-v2 + SVM | 3,536 (Multi-source) | Very High | 66.3% |
| 4. Hybrid | all-MiniLM-L6-v2 + SVM | 4,500 (Real + Conv. Fill) | High | 70.6% |
| 5. Unlimited Scraping | all-MiniLM-L6-v2 + SVM | 32,434 (Aggressive API) | Very Low (High Noise) | 48.8% |
| 6. Unlimited Scraping | TF-IDF + LinearSVC | 32,434 (Aggressive API) | Very Low (High Noise) | 46.4% |

## Key Learnings for the Synopsis

### 1. The "Garbage In, Garbage Out" Principle
In Iteration 5, we aggressively scraped over 32,000 sentences from Wikipedia and medical sites to give the embedding model the volume it needed. However, the accuracy collapsed to 48%. This proved that pulling in raw volume brings massive noise (history of diseases, epidemiology, treatments, doctor biographies). The model could no longer distinguish symptoms because the dataset was flooded with irrelevant text. **Quality of data is vastly more important than quantity of data.**

### 2. Algorithm-Data Size Mismatch
Sentence embeddings (Iteration 2 & 3) underperformed compared to TF-IDF (Iteration 1). This is because dense 384-dimensional embeddings require massive amounts of *clean* data to draw stable decision boundaries. TF-IDF maps text into a sparse space, which Linear SVMs handle brilliantly even with small datasets (600 samples per class). 

### 3. Conclusion & Final Architecture
We definitively proved that the most robust solution for our dataset size is the sparse representation (TF-IDF) combined with NLTK synonym augmentation to ensure balanced classes. The final NLP branch uses **TF-IDF + Calibrated LinearSVC**, achieving 87%+ accuracy.
