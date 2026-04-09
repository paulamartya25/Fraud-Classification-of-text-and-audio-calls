#!/usr/bin/env python3
"""Train improved Hindi and Telugu Call models with optimization techniques"""
import os
import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Try to import TensorFlow (optional - for Telugu LSTM)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available. Telugu LSTM model will be skipped.")

print("=" * 80)
print("TRAINING IMPROVED HINDI AND TELUGU CALL MODELS")
print("=" * 80)

# ==================== HINDI CALL MODEL (Improved LogisticRegression) ====================

def train_improved_hindi_call_model():
    """Train improved Hindi Call model with TF-IDF + LogisticRegression + SMOTE"""
    
    print(f"\n{'-' * 80}")
    print(f"Training Improved HINDI Call Model")
    print(f"{'-' * 80}")
    
    # Load dataset
    df = pd.read_csv('hindi_call_records_dataset.csv')
    print(f"[INFO] Loaded Hindi call dataset: {df.shape[0]} samples")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Check label format and normalize
    X = df['text'].astype(str)
    y = df['label'].copy()
    
    # Normalize labels (handle both string and numeric)
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if (x == 'fraud' or x == '1') else 0)
    else:
        y = y.apply(lambda x: 1 if x == 1 else 0)
    
    print(f"[INFO] Label distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize text
    print("[INFO] Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Apply SMOTE for class balancing
    print("[INFO] Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)
    
    # Train LogisticRegression with optimized parameters
    print("[INFO] Training LogisticRegression model...")
    model = LogisticRegression(
        max_iter=1000,
        C=0.1,  # Regularization strength (lower = stronger regularization)
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_smote, y_train_smote)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n[RESULTS] HINDI Call Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Detailed classification report
    print(f"\n[DETAILED REPORT]")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Save model and vectorizer
    joblib.dump(model, 'hindi_call_model_improved.pkl')
    joblib.dump(vectorizer, 'hindi_call_vectorizer_improved.pkl')
    print(f"[SUCCESS] Hindi call model saved: hindi_call_model_improved.pkl")
    print(f"[SUCCESS] Hindi call vectorizer saved: hindi_call_vectorizer_improved.pkl")
    
    return model, vectorizer, accuracy, precision, recall, f1

# ==================== TELUGU CALL MODEL (Improved LSTM) ====================

def train_improved_telugu_call_model():
    """Train improved Telugu Call model with LSTM + optimized hyperparameters"""
    
    if not TENSORFLOW_AVAILABLE:
        print(f"\n{'-' * 80}")
        print(f"[SKIP] Telugu LSTM Model - TensorFlow not available")
        print(f"{'-' * 80}")
        return None, None, None, 0, 0, 0, 0
    
    print(f"\n{'-' * 80}")
    print(f"Training Improved TELUGU Call Model")
    print(f"{'-' * 80}")
    
    # Load dataset
    df = pd.read_csv('telugu_call_dataset.csv')
    print(f"[INFO] Loaded Telugu call dataset: {df.shape[0]} samples")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # Prepare data
    X = df['transcript'].astype(str).values
    y = df['label'].copy()
    
    # Normalize labels
    if y.dtype == 'object':
        y = y.apply(lambda x: 1 if (x == 'fraud' or x == '1') else 0)
    else:
        y = y.apply(lambda x: 1 if x == 1 else 0)
    
    y = y.values
    
    print(f"[INFO] Label distribution: {np.bincount(y)}")
    
    # Tokenize and pad sequences
    print("[INFO] Tokenizing text...")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    
    sequences = tokenizer.texts_to_sequences(X)
    max_len = max(len(seq) for seq in sequences)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    print(f"[INFO] Max sequence length: {max_len}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build improved LSTM model
    print("[INFO] Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Train model with optimized hyperparameters
    print("[INFO] Training model (30 epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=0
    )
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n[RESULTS] TELUGU Call Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Detailed classification report
    print(f"\n[DETAILED REPORT]")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Save model and tokenizer
    model.save('telugu_call_model_improved.h5')
    with open('telugu_call_tokenizer_improved.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"[SUCCESS] Telugu call model saved: telugu_call_model_improved.h5")
    print(f"[SUCCESS] Telugu call tokenizer saved: telugu_call_tokenizer_improved.pkl")
    print(f"[INFO] Max sequence length saved: {max_len}")
    
    return model, tokenizer, max_len, accuracy, precision, recall, f1

# ==================== TRAIN BOTH MODELS ====================

# Train Hindi Call Model
h_model, h_vec, h_acc, h_prec, h_rec, h_f1 = train_improved_hindi_call_model()

# Train Telugu Call Model
t_model, t_tok, t_max_len, t_acc, t_prec, t_rec, t_f1 = train_improved_telugu_call_model()

# Summary table
print(f"\n{'=' * 80}")
print("IMPROVED CALL MODELS SUMMARY")
print(f"{'=' * 80}")
print(f"\n{'Language':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print(f"{'-' * 63}")
print(f"{'Hindi':<15} {h_acc:<12.4f} {h_prec:<12.4f} {h_rec:<12.4f} {h_f1:<12.4f}")
if TENSORFLOW_AVAILABLE:
    print(f"{'Telugu':<15} {t_acc:<12.4f} {t_prec:<12.4f} {t_rec:<12.4f} {t_f1:<12.4f}")
print(f"\n{'=' * 80}")
print("[SUCCESS] All improved call models trained and saved!")
print("=" * 80)
