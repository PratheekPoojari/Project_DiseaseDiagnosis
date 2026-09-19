# Multimodal Skin Disease Diagnosis System

A diagnostic assistance system that predicts probable skin conditions from symptom text, skin images, or both — combining NLP, Digital Image Processing, and Machine Learning behind a Streamlit web interface.

> ⚠️ **Academic project — not intended for real medical diagnosis or clinical use.**

---

## Overview

This system accepts a text description of symptoms, an image of the affected skin area, or both simultaneously. It returns a predicted condition with a confidence score and an audio readout. Built as Project 1 for both the NLP and DIP courses at Surana College, BCA 5th Semester.

It covers **10 skin conditions:**
Eczema · Melanoma · Atopic Dermatitis · Basal Cell Carcinoma · Melanocytic Nevi · Benign Keratosis · Psoriasis / Lichen Planus · Seborrheic Keratoses · Fungal Infections · Viral Infections (Warts/Molluscum)

---

## Architecture

```
User Input
    │
    ├── Symptom Text (typed or recorded via microphone)
    │        └── STT (Google Web Speech API) ──→ Text
    │                      └── NLP Branch (TF-IDF + LinearSVC) ──→ probability dict
    │
    └── Skin Image (uploaded or captured via camera)
                 └── DIP Branch (EfficientNet-B0 CNN) ──→ probability dict
                                          │
                            Late Fusion Layer (60% Image / 40% Text)
                                          │
                                  Final Diagnosis
                                          │
                                TTS Audio Readout (gTTS)
```

| Component | Method | Key Libraries | Accuracy |
|---|---|---|---|
| NLP Branch | TF-IDF + Calibrated LinearSVC | scikit-learn | ~81% |
| DIP Branch | EfficientNet-B0 (transfer learning) | PyTorch, torchvision | ~83.76% |
| Fusion Layer | Weighted probability averaging | Pure Python | — |
| Frontend | Streamlit web app | Streamlit | — |
| STT | Google Web Speech API | SpeechRecognition | — |
| TTS | Google Text-to-Speech | gTTS | — |
| Notifications *(pending)* | Email + SMS + scheduler | smtplib, Twilio, APScheduler | — |

---

## Dataset

**Source:** [Kaggle — skin-diseases-image-dataset (ismailpromus)](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
Downloaded via `kagglehub`. Full 10-class dataset, ~27,000 images total.

| Split | Images |
|---|---|
| Training (80%) | 21,722 |
| Validation (20%) | 5,431 |

Raw images and processed data are **not tracked in git** (`data/` is gitignored).
Run `src/prepare_dataset.py` to reproduce the full download and preprocessing pipeline.

---

## Project Structure

```
Project_DiseaseDiagnosis/
├── app.py                      # Main Streamlit application
├── requirements.txt
├── README.md
├── project-history.md          # Chronological build log & decisions
├── AGENTS.md                   # AI agent working rules for this project
│
├── data/                       # Raw + processed images (gitignored)
│   ├── raw/
│   └── processed/
│       ├── eczema/
│       ├── melanoma/
│       └── ...                 # 10 class folders
│
├── models/
│   ├── nlp/
│   │   └── svm_tfidf_model.pkl # Trained NLP pipeline (~2.8MB)
│   └── dip/
│       └── efficientnet_b0_skin.pth  # Trained CNN checkpoint (~16MB)
│
├── src/
│   ├── prepare_dataset.py      # Downloads + preprocesses all 10 classes
│   ├── nlp/
│   │   ├── train_nlp_model.py  # Trains TF-IDF + LinearSVC pipeline
│   │   └── predict.py          # NLP inference interface
│   ├── dip/
│   │   ├── preprocess.py       # OpenCV image preprocessing pipeline
│   │   ├── train_cnn.py        # EfficientNet-B0 training script (Kaggle Notebook)
│   │   └── predict.py          # DIP inference interface
│   └── fusion/
│       └── fuse.py             # Late fusion — weighted probability averaging
│
├── docs/
│   ├── NLP_Synopsis.md
│   ├── DIP_Synopsis.md
│   └── nlp_experiment_log.md   # Full NLP experiment results across 6 models
│
└── notebooks/                  # Jupyter scratch space
```

---

## Setup

```bash
git clone git@github.com:PratheekPoojari/Project_DiseaseDiagnosis.git
cd Project_DiseaseDiagnosis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Reproduce the dataset (requires Kaggle API credentials)
python3 src/prepare_dataset.py

# Launch the app
streamlit run app.py
```

> **Note:** CNN training requires a cloud GPU. The `train_cnn.py` script is designed to be copy-pasted into a Kaggle Notebook (GPU T4). Download the output `.pth` file and place it at `models/dip/efficientnet_b0_skin.pth`.

---

## Current Status

| Feature | Status |
|---|---|
| Dataset sourced and preprocessed (10 classes) | ✅ Complete |
| NLP Branch (TF-IDF + LinearSVC, ~81% acc.) | ✅ Complete |
| DIP Branch (EfficientNet-B0, ~83.76% acc.) | ✅ Complete |
| Late Fusion Layer (60% Image / 40% Text) | ✅ Complete |
| Streamlit Frontend (core UI) | ✅ Complete |
| Speech-to-Text input (Google Web Speech) | ✅ Complete |
| Text-to-Speech output (gTTS, auto-plays result) | ✅ Complete |
| Camera input (`st.camera_input` + DroidCam) | ✅ Integrated (DroidCam connects as system webcam) |
| Multi-format symptom input (.txt/.csv/.pdf/.docx) | ⏳ Pending |
| Multi-format data export | ⏳ Pending |
| Notification module (Email + SMS + APScheduler) | ⏳ Pending |

**Deadline: September 28, 2026**

---

## Key Design Decisions

**Why EfficientNet-B0 over ResNet-18?**
EfficientNet-B0 has 5.3M parameters vs ResNet-18's 11.2M, downloads in 20MB vs 45MB, and achieves higher accuracy-per-parameter through compound scaling. Inference on CPU (the laptop) is also faster.

**Why TF-IDF over Sentence Transformers?**
Experiments on this specific dataset showed TF-IDF + LinearSVC (~81%) outperformed all embedding-based approaches. The dataset's symptom text is short and keyword-heavy — TF-IDF's word-frequency approach matches this structure better than semantic embeddings. Full results in `docs/nlp_experiment_log.md`.

**Why 60% Image / 40% Text fusion weights?**
Skin diseases are fundamentally visual. The image branch gets the deciding vote, but text provides meaningful correction when the CNN is uncertain. Decided by Pratheek based on the clinical nature of skin diagnosis.

**Why Google STT over Whisper (local)?**
Whisper's smallest model is ~150MB and slow on CPU without CUDA. Google's free Web Speech API is instant and accurate enough for symptom description, with graceful error handling for no-internet scenarios.

**Why gTTS over pyttsx3 for TTS?**
pyttsx3 only plays through the server's system speaker. gTTS generates an MP3 that `st.audio()` embeds directly in the browser — better UX. V2 will upgrade to ElevenLabs/OpenAI TTS for a more natural voice.

---

## Limitations

This system is developed strictly for academic and demonstrative purposes.
It does not replace professional medical consultation.
Follow-up notifications are for engagement and reminder purposes only.

---

## Authors

- **Pratheek S Poojari** — BCA, 5th Semester, Surana College
