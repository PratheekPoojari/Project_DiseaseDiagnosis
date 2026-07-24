# Multimodal Disease Diagnosis System

A diagnostic assistance system that predicts probable skin conditions from
symptom text, skin images, or both — combining NLP, Digital Image Processing,
and Machine Learning.

> Academic project — not intended for real medical diagnosis.

## Overview

This system accepts either a text description of symptoms, an image of the
affected skin area, or both, and returns a predicted condition with a
confidence score. It's scoped to three conditions: **Eczema**, **Psoriasis**,
and **Fungal Infections**.

## Architecture

| Component | Function | Key Libraries |
|---|---|---|
| NLP Branch | Classifies symptom text | scikit-learn, NLTK/spaCy |
| DIP Branch | Classifies skin images via CNN | OpenCV, PyTorch/TensorFlow |
| Fusion Layer | Combines both predictions | scikit-learn, NumPy |
| Frontend | User interface | Streamlit |
| Notification Module | SMS/email follow-up reminders | smtplib, Twilio, APScheduler |

## Dataset

Source: [Kaggle — skin-diseases-image-dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
(downloaded via `kagglehub`)

Filtered from the original 10 classes down to 3, matching this project's scope:

- Eczema — 1,677 images
- Psoriasis — 2,055 images
- Fungal Infections — 1,702 images

Raw images are not tracked in git (`data/` is gitignored). See
`src/download_data.py` to reproduce the download.

## Project Structure

```
Project_DiseaseDiagnosis/
├── data/ # raw + processed images (gitignored)
├── src/
│ ├── nlp/ # NLP branch
│ ├── dip/ # DIP branch
│ ├── fusion/ # fusion layer
│ └── download_data.py
├── models/ # trained model artifacts (gitignored, except .gitkeep)
├── notebooks/ # exploratory notebooks
└── requirements.txt
```

## Setup

```bash
git clone "git@github.com:PratheekPoojari/Project_DiseaseDiagnosis.git"
cd Project_DiseaseDiagnosis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/download_data.py
```

## Status

🚧 In active development — Disease Diagnosis is Project 1 of a semester
coursework series. Current stage: dataset sourced, preprocessing pipeline
completed.

## Limitations

This system is developed strictly for academic and demonstrative purposes.
It is not intended for actual medical diagnosis and does not replace
professional medical consultation. Follow-up notifications are for
engagement/reminder purposes only.

## Authors

- Pratheek S Poojari — BCA, 5th Semester
- Vivek Kumar Ghosh — BCA, 5th Semeter 
