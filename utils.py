import os
import joblib
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import speech_recognition as sr

# Try to import TensorFlow (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# -------------------- Constants --------------------

DATA_PATHS = {
    "hindi_sms": "hindi_sms_dataset.csv",
    "telugu_sms": "telugu_sms_dataset.csv",
    "hindi_call": "hindi_call_records_dataset.csv",
    "telugu_call": "telugu_call_dataset.csv",
}

MODEL_PATHS = {
    "hindi_sms": "hindi_fraud_classifier.pkl",
    "telugu_sms": "telugu_fraud_classifier.pkl",
    "hindi_call": {
        "model": "hindi_fraud_model.pkl",
        "vectorizer": "hindi_vectorizer.pkl"
    },
    "telugu_call": {
        "model": "telugu_fraud_classifier.pkl",
        "tokenizer": "tokenizer.pkl"
    }
}

CALL_MODEL_PATH_EN = "call_model.h5"
MSG_MODEL_PATH_EN = "msg_model.h5"
CALL_TOKENIZER_PATH_EN = "call_tokenizer.pkl"
MSG_TOKENIZER_PATH_EN = "msg_tokenizer.pkl"

# -------------------- Load English Models --------------------
def load_sms_dataset(language):
    key = language.lower()
    path = DATA_PATHS.get(key)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset for '{language}' not found at: {path}")
    return pd.read_csv(path)

def load_english_models():
    call_model = load_model(CALL_MODEL_PATH_EN)
    msg_model = load_model(MSG_MODEL_PATH_EN)

    with open(CALL_TOKENIZER_PATH_EN, 'rb') as f:
        call_tokenizer = pickle.load(f)
    with open(MSG_TOKENIZER_PATH_EN, 'rb') as f:
        msg_tokenizer = pickle.load(f)

    return call_model, call_tokenizer, msg_model, msg_tokenizer

# -------------------- English Classification --------------------

def classify_english_message(text, msg_model, msg_tokenizer):
    input_seq = msg_tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_seq, maxlen=200, padding="post")
    prediction = msg_model.predict(input_padded)[0][0]
    return "Fraud Message" if prediction >= 0.5 else "Normal Message"

def classify_english_audio(audio_path, call_model, call_tokenizer):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data)

        input_seq = call_tokenizer.texts_to_sequences([transcript])
        input_padded = pad_sequences(input_seq, maxlen=200, padding="post")
        prediction = call_model.predict(input_padded)[0][0]
        return "Fraud Call" if prediction >= 0.5 else "Normal Call"
    except Exception as e:
        return f"Error processing audio: {e}"

# -------------------- SMS Classification --------------------

#def load_sms_dataset(language):
#    key = f"{language}_sms"
#    df = pd.read_csv(DATA_PATHS[key])
 #   assert 'Text' in df.columns and 'Label' in df.columns, "Dataset must contain 'Text' and 'Label' columns"
  #  return df


def train_or_load_sms_model(df, language):
    filename = MODEL_PATHS[f"{language}"]
    if os.path.exists(filename):
        return joblib.load(filename)

    X = df['Text']
    y = df['Label'].copy()
    
    # Normalize labels: 'fraud' or 1 = 1, 'real' or 0 = 0
    y = y.apply(lambda x: 1 if (x == 'fraud' or x == 1) else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])
    model.fit(X_train, y_train)
    print(f"{language.title()} SMS Model:\n", classification_report(y_test, model.predict(X_test)))
    joblib.dump(model, filename)
    return model

# -------------------- Hindi Call Classification --------------------

def train_or_load_hindi_call_model():
    model_file = MODEL_PATHS['hindi_call']['model']
    vec_file = MODEL_PATHS['hindi_call']['vectorizer']

    if os.path.exists(model_file) and os.path.exists(vec_file):
        return joblib.load(model_file), joblib.load(vec_file)

    df = pd.read_csv(DATA_PATHS['hindi_call'])
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Hindi Call Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print(classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vec_file)
    return model, vectorizer

def predict_hindi_call_from_audio(audio_path):
    model, vectorizer = train_or_load_hindi_call_model()
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="hi-IN")
        features = vectorizer.transform([text])
        prediction = model.predict(features)[0]
        return "Fraud" if prediction == 1 else "Real"
    except Exception as e:
        return f"Speech recognition error: {e}"

# -------------------- Telugu Call Classification --------------------

def train_or_load_telugu_call_model():
    model_file = MODEL_PATHS['telugu_call']['model']
    tokenizer_file = MODEL_PATHS['telugu_call']['tokenizer']

    df = pd.read_csv(DATA_PATHS['telugu_call'])
    df['label'] = df['label'].map({'fraud': 1, 'real': 0})
    X = df['transcript'].astype(str).values
    y = df['label'].values

    if os.path.exists(tokenizer_file):
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)

    sequences = tokenizer.texts_to_sequences(X)
    max_len = max(len(seq) for seq in sequences)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding='post')
    X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = Sequential([
            Embedding(input_dim=10000, output_dim=64, input_length=max_len),
            LSTM(64),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
        model.save(model_file)

    return model, tokenizer, max_len

def predict_telugu_call(text):
    model, tokenizer, max_len = train_or_load_telugu_call_model()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)[0][0]
    label = "Fraud" if prediction >= 0.5 else "Real"
    return f"Prediction: {label} (Confidence: {prediction:.4f})"
