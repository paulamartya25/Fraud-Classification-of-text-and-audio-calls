# 🛡️ Multilingual Fraud Detection System

![MINOR_6_4](https://github.com/user-attachments/assets/44a2b3af-7312-4336-bd15-79546468f37c)
![MINOR_6_3](https://github.com/user-attachments/assets/71152e9e-dced-485d-87ec-ebefef9b4307)
![MINOR_6_1](https://github.com/user-attachments/assets/99c7650d-f9e9-4727-9a91-7ce06437ca95)

### Real-time fraud classification across English, Hindi & Telugu — SMS text and audio call recordings

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Live Demo →** [fraudclassification-qlkoaargqguepk7ah6zaas.streamlit.app](https://fraudclassification-qlkoaargqguepk7ah6zaas.streamlit.app/)

---

## 📌 Overview

This project addresses a critical gap in fraud detection: **most systems only work for English, and only for text.** This system handles both **SMS messages** and **phone call audio recordings** across three languages — English, Hindi, and Telugu — making it practical for the linguistically diverse Indian market.

The system was deployed as a production-grade Streamlit web application, serving **15+ customer service professionals** in a real-world environment.

---

## 📊 Model Performance

### English Models (Improved)

| Modality | Model | Accuracy | Recall | F1-Score | ROC-AUC |
|----------|-------|----------|--------|----------|---------|
| SMS | TF-IDF + Logistic Regression (SMOTE) | **95%** | **90%** | 0.9826 ± 0.0348 | 1.0 |
| Call Audio | LSTM (128 units, optimized) | **95%** | **93%** | — | — |

### Hindi & Telugu Models

| Language | Modality | Model | Performance |
|----------|----------|-------|-------------|
| Hindi | SMS | TF-IDF + Naive Bayes | High precision on low-resource data |
| Hindi | Call | TF-IDF + Logistic Regression | Handles linguistic variations |
| Telugu | SMS | TF-IDF + Naive Bayes | Low-resource NLP pipeline |
| Telugu | Call | LSTM (H5) | Audio-to-text + classification |

> 📈 English SMS improved from 84.7% → **95% accuracy** and 83.75% → **90% recall** using SMOTE balancing and optimal thresholding.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                         │
│          (Text Input / Audio File Upload)                    │
└──────────┬───────────────────────────────┬───────────────────┘
           │ SMS Text                      │ Audio Call
           ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────────┐
│  Language Detection  │      │  Google Speech-to-Text API    │
│  (English/Hindi/Telugu)     │  (Real-time Transcription)    │
└──────────┬───────────┘      └──────────────┬────────────────┘
           │                                 │ Transcript
           ▼                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                Language-Specific Routing                     │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │   English   │  │    Hindi    │  │       Telugu         │ │
│  │ LSTM (SMS)  │  │  TF-IDF +  │  │  TF-IDF + NB / LR   │ │
│  │ LSTM (Call) │  │  NB / LR   │  │  LSTM (Call)         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬───────────┘ │
└─────────┼────────────────┼──────────────────── ┼────────────┘
          └────────────────┴────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Fraud / Normal + %    │
              │  Confidence Score      │
              └────────────────────────┘
```

---

## 🔍 Key Features

- **Multimodal** — processes both SMS text and audio call recordings
- **Multilingual** — English, Hindi, Telugu with language-specific model routing
- **Production-deployed** — live on Streamlit Cloud, used by real professionals
- **SMOTE-balanced training** — prevents class imbalance bias in fraud detection
- **Optimized decision thresholds** — SMS: 0.30, Call: 0.35 (tuned for recall)
- **Cross-validated** — 5-fold CV ensures generalization (F1 = 0.9826 ± 0.0348)
- **Duplicate detection** — hash-based deduplication for event integrity

---

## 🧠 Technical Approach

### Why different models per language?

| Language | Challenge | Solution |
|----------|-----------|----------|
| English | Variable-length sequences, contextual patterns | LSTM captures long-range dependencies |
| Hindi | Code-switching, script variation, limited labeled data | TF-IDF robust for morphologically rich text |
| Telugu | Very low-resource, limited training data | TF-IDF + simple classifiers avoid overfitting |

### Why SMOTE + Lower Thresholds?

Fraud datasets are inherently imbalanced (few fraud cases vs. many legitimate ones). Without correction, models learn to predict "Normal" always. SMOTE synthetically oversamples fraud cases, and lowering the decision threshold from 0.5 → 0.3 prioritizes **recall** (catching fraud) over precision — the correct trade-off for fraud detection.

---

## 📁 Repository Structure

```
├── app.py                          # Main Streamlit application
├── utils.py                        # Preprocessing & helper functions
├── requirements.txt                # Python dependencies
│
├── # English Models
├── msg_model.h5                    # LSTM model — English SMS
├── msg_tokenizer.pkl               # Tokenizer — English SMS
├── call_model.h5                   # LSTM model — English Call
├── call_tokenizer.pkl              # Tokenizer — English Call
│
├── # Hindi Models
├── hindi_fraud_model.pkl           # Hindi SMS classifier
├── hindi_vectorizer.pkl            # Hindi TF-IDF vectorizer
├── hindi_fraud_classifier.pkl      # Hindi call classifier
│
├── # Telugu Models
├── telugu_fraud_classifier.pkl     # Telugu SMS classifier
├── telugu_call_classifier.h5       # Telugu call LSTM model
│
├── # Datasets
├── hindi_sms_dataset.csv
├── hindi_call_records_dataset.csv
├── telugu_sms_dataset.csv
└── telugu_call_dataset.csv
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/paulamartya25/Fraud-Classification-of-text-and-audio-calls.git
cd Fraud-Classification-of-text-and-audio-calls
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install imbalanced-learn  # For SMOTE-based retraining
```

### 3. Run the app locally
```bash
streamlit run app.py
```

---

## 💻 Usage Examples

### SMS Classification (English)
```python
import joblib

sms_model = joblib.load('english_sms_model_improved.pkl')
sms_vectorizer = joblib.load('english_sms_vectorizer_improved.pkl')

text = "Congratulations! You won £1000. Click here now"
text_vec = sms_vectorizer.transform([text])
probability = sms_model.predict_proba(text_vec)[0][1]

# Optimal threshold: 0.30 (tuned for recall)
prediction = "Fraud" if probability >= 0.3 else "Normal"
print(f"Prediction: {prediction} | Confidence: {probability:.2%}")
```

### Call Audio Classification (English)
```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

call_model = load_model('english_call_model_improved.h5')
with open('english_call_tokenizer_improved.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

transcript = "Hello you have inherited property worth 5 million"
seq = tokenizer.texts_to_sequences([transcript])
padded = pad_sequences(seq, maxlen=22, padding='post')
probability = call_model.predict(padded)[0][0]

# Optimal threshold: 0.35
prediction = "Fraud" if probability >= 0.35 else "Normal"
print(f"Prediction: {prediction} | Confidence: {probability:.2%}")
```

---

## 📈 Model Improvements (v1 → v2)

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| Decision threshold | 0.50 | 0.30 (SMS), 0.35 (Call) | +50% fraud detection |
| Class balancing | None | SMOTE oversampling | Eliminates class bias |
| LSTM units | 64 | 128 | Better learning capacity |
| Learning rate | 0.001 | 0.0005 | More stable convergence |
| Training epochs | 15 | 30 | Full convergence |
| Validation | None | 5-fold cross-validation | Prevents overfitting |

---

## 🔄 Retraining

To retrain models on new data:
```bash
python train_improved_models.py
# Automatically applies: SMOTE, class weights, optimal thresholds, cross-validation
# Runtime: ~5 minutes
```

**Retraining schedule:** Every quarter, or when production accuracy drops >5%.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | TensorFlow / Keras (LSTM) |
| Classical ML | Scikit-learn (Naive Bayes, Logistic Regression) |
| Text Features | TF-IDF Vectorization |
| Class Balancing | imbalanced-learn (SMOTE) |
| Audio Transcription | Google Speech-to-Text API |
| Web Interface | Streamlit |
| Model Serialization | Pickle, Joblib, HDF5 (.h5) |

---

## 📋 Monitoring in Production

Track these metrics quarterly:

| Metric | Target | Action if below |
|--------|--------|-----------------|
| Accuracy | ≥ 90% | Retrain on new data |
| Recall (Fraud) | ≥ 85% | Lower decision threshold |
| F1-Score | ≥ 0.90 | Check class distribution |
| False Positive Rate | ≤ 10% | Raise decision threshold |

---

## 🔮 Future Work

- [ ] Add support for more Indian languages (Tamil, Kannada, Bengali)
- [ ] Transformer-based models (IndicBERT, mBERT) for better multilingual performance
- [ ] Real-time audio streaming (replace batch file upload)
- [ ] Explainability layer (LIME/SHAP for prediction reasoning)
- [ ] Active learning pipeline for continuous improvement with new fraud patterns

---

## 👤 Author

**Amartya Paul**  
B.Tech — Data Science & Artificial Intelligence  
IIIT Naya Raipur (2022–2026)  
📧 amartya11221@gmail.com  
🔗 [LinkedIn]() · [GitHub](https://github.com/paulamartya25)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
