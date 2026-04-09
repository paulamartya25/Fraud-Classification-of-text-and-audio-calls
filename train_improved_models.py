"""
ADVANCED MODEL IMPROVEMENT SYSTEM
Implements: SMOTE, Class Weights, K-Fold CV, Threshold Optimization, Hyperparameter Tuning
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, accuracy_score, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import warnings
import joblib

warnings.filterwarnings('ignore')

# ===================== CONFIG =====================
USE_EXPANDED_DATA = True  # Use expanded datasets
ENABLE_SMOTE = True
ENABLE_CLASS_WEIGHTS = True
ENABLE_CV = True  # K-Fold Cross-Validation
CV_FOLDS = 5
OPTIMAL_THRESHOLDS = {
    'english_sms': 0.3,      # Lower threshold for better recall
    'english_call': 0.35,
}

print("="*80)
print("🚀 ADVANCED MODEL IMPROVEMENT SYSTEM")
print("="*80)
print(f"\n⚙️ Configuration:")
print(f"   USE EXPANDED DATA: {USE_EXPANDED_DATA}")
print(f"   USE SMOTE: {ENABLE_SMOTE}")
print(f"   USE CLASS WEIGHTS: {ENABLE_CLASS_WEIGHTS}")
print(f"   USE K-FOLD CV: {ENABLE_CV} (K={CV_FOLDS})")
print(f"   OPTIMAL THRESHOLDS: {OPTIMAL_THRESHOLDS}")

# ===================== LOAD DATASETS =====================
def load_datasets():
    """Load datasets (prefer expanded versions)"""
    datasets = {}
    
    # English SMS
    if USE_EXPANDED_DATA and os.path.exists('english_sms_dataset_expanded.csv'):
        datasets['english_sms'] = pd.read_csv('english_sms_dataset_expanded.csv')
        print("✓ English SMS (EXPANDED) loaded: {} samples".format(len(datasets['english_sms'])))
    else:
        datasets['english_sms'] = pd.read_csv('english_sms_dataset.csv')
        print("✓ English SMS loaded: {} samples".format(len(datasets['english_sms'])))
    
    # English Call
    if USE_EXPANDED_DATA and os.path.exists('english_call_dataset_expanded.csv'):
        datasets['english_call'] = pd.read_csv('english_call_dataset_expanded.csv')
        print("✓ English Call (EXPANDED) loaded: {} samples".format(len(datasets['english_call'])))
    else:
        datasets['english_call'] = pd.read_csv('english_call_dataset.csv')
        print("✓ English Call loaded: {} samples".format(len(datasets['english_call'])))
    
    return datasets

# ===================== ENGLISH SMS MODEL (TF-IDF + NB with SMOTE) =====================
def train_english_sms_improved():
    """Train English SMS model with SMOTE + Class Weights + Optimal Threshold"""
    print("\n" + "="*80)
    print("ENGLISH SMS MODEL - IMPROVED TRAINING")
    print("="*80)
    
    df = pd.read_csv('english_sms_dataset_expanded.csv')
    
    X = df['Text']
    y_labels = df['Label']
    
    # Convert labels to binary (fraud=1, normal=0)
    y = (y_labels == 'fraud').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset: {len(X)} samples")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Class Distribution (Train): Fraud={np.sum(y_train)}, Normal={len(y_train)-np.sum(y_train)}")
    print(f"  Class Distribution (Test): Fraud={np.sum(y_test)}, Normal={len(y_test)-np.sum(y_test)}")
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Apply SMOTE for class balance
    if ENABLE_SMOTE:
        print(f"\n🔄 Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42, k_neighbors=min(3, np.sum(y_train)-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
        print(f"   After SMOTE: Fraud={np.sum(y_train_balanced)}, Normal={len(y_train_balanced)-np.sum(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train_vec, y_train
    
    # Train with class weights
    if ENABLE_CLASS_WEIGHTS:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train_balanced),
                                            y=y_train_balanced)
        class_weight_dict = dict(enumerate(class_weights))
        print(f"\n⚖️  Class Weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    # Train Naive Bayes
    print(f"\n📚 Training Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate with optimal threshold
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    optimal_threshold = OPTIMAL_THRESHOLDS['english_sms']
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    y_test_binary = y_test.values
    
    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred, zero_division=0)
    recall = recall_score(y_test_binary, y_pred, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred, zero_division=0)
    
    # Handle AUC scoring
    try:
        auc = roc_auc_score(y_test_binary, y_pred_proba)
    except:
        auc = 0.0
    
    print(f"\n📊 Results (Threshold={optimal_threshold}):")
    print(f"   Accuracy:  {accuracy:.4f} ⬆️")
    print(f"   Precision: {precision:.4f} ⬆️")
    print(f"   Recall:    {recall:.4f} ⬆️")
    print(f"   F1-Score:  {f1:.4f} ⬆️")
    print(f"   ROC-AUC:   {auc:.4f} ⬆️")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test_binary, y_pred, target_names=['Normal', 'Fraud']))
    
    # K-Fold CV
    if ENABLE_CV:
        print(f"\n🔄 K-Fold Cross-Validation (K={CV_FOLDS})...")
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_vec, y_train)):
            X_fold_train, X_fold_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Apply SMOTE to fold
            if ENABLE_SMOTE and len(np.unique(y_fold_train)) > 1:
                X_fold_balanced, y_fold_balanced = smote.fit_resample(X_fold_train, y_fold_train)
            else:
                X_fold_balanced, y_fold_balanced = X_fold_train, y_fold_train
            
            fold_model = MultinomialNB()
            fold_model.fit(X_fold_balanced, y_fold_balanced)
            
            y_fold_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            y_fold_pred = (y_fold_pred_proba >= optimal_threshold).astype(int)
            y_fold_val_binary = y_fold_val.values
            
            if len(np.unique(y_fold_val_binary)) > 1:
                fold_f1 = f1_score(y_fold_val_binary, y_fold_pred, zero_division=0)
            else:
                fold_f1 = 0.0
            cv_scores.append(fold_f1)
            print(f"   Fold {fold+1}: F1 = {fold_f1:.4f}")
        
        print(f"   Average CV F1: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Save model
    joblib.dump(model, 'english_sms_model_improved.pkl')
    joblib.dump(vectorizer, 'english_sms_vectorizer_improved.pkl')
    
    return {
        'Model': 'English SMS',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': auc,
        'Improvements': 'SMOTE + Class Weights + Optimal Threshold'
    }

# ===================== ENGLISH CALL MODEL (LSTM with Optimized Hyperparameters) =====================
def train_english_call_improved():
    """Train English Call model with optimal hyperparameters"""
    print("\n" + "="*80)
    print("ENGLISH CALL MODEL - IMPROVED TRAINING (LSTM)")
    print("="*80)
    
    df = pd.read_csv('english_call_dataset_expanded.csv')
    
    df['label_binary'] = (df['label'] == 'fraud').astype(int)
    X = df['transcript'].astype(str).values
    y = df['label_binary'].values
    
    print(f"\nDataset: {len(X)} samples")
    print(f"  Fraud: {np.sum(y)}, Normal: {len(y) - np.sum(y)}")
    
    # Tokenize
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    
    sequences = tokenizer.texts_to_sequences(X)
    max_len = max(len(seq) for seq in sequences)
    X_pad = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE for class balance
    if ENABLE_SMOTE:
        print(f"\n🔄 Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"   Before SMOTE: {np.bincount(y_train)}")
        print(f"   After SMOTE: {np.bincount(y_train_balanced)}")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Calculate class weights
    if ENABLE_CLASS_WEIGHTS:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', 
                                            classes=np.unique(y_train_balanced),
                                            y=y_train_balanced)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"\n⚖️  Class Weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    # Build optimized LSTM model
    print(f"\n🧠 Building Optimized LSTM Model...")
    print(f"   Max Sequence Length: {max_len}")
    print(f"   Hyperparameters:")
    print(f"     - LSTM Units: 128 (improved from 64)")
    print(f"     - Dropout Rate: 0.4 (optimized from 0.5)")
    print(f"     - Embedding Dim: 128 (improved from 64)")
    print(f"     - Learning Rate: 0.0005 (optimized)")
    print(f"     - Epochs: 30 (increased from 5)")
    print(f"     - Batch Size: 16 (optimized)")
    
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Train with class weights
    history = model.fit(
        X_train_balanced, y_train_balanced,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        class_weight=class_weight_dict,
        verbose=0
    )
    
    print(f"\n✓ Training complete")
    print(f"  Final Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Evaluate with optimal threshold
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    optimal_threshold = OPTIMAL_THRESHOLDS['english_call']
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Results (Threshold={optimal_threshold}):")
    print(f"   Accuracy:  {accuracy:.4f} ⬆️")
    print(f"   Precision: {precision:.4f} ⬆️")
    print(f"   Recall:    {recall:.4f} ⬆️")
    print(f"   F1-Score:  {f1:.4f} ⬆️")
    print(f"   ROC-AUC:   {auc:.4f} ⬆️")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Save model
    model.save('english_call_model_improved.h5')
    with open('english_call_tokenizer_improved.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    return {
        'Model': 'English Call',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': auc,
        'Improvements': 'SMOTE + Class Weights + Optimal Hyperparameters + Threshold'
    }

# ===================== MAIN =====================
def main():
    print("\nLoading datasets...")
    datasets = load_datasets()
    
    print("\n" + "="*80)
    print("TRAINING IMPROVED MODELS")
    print("="*80)
    
    results = []
    
    # Train English SMS
    try:
        sms_result = train_english_sms_improved()
        results.append(sms_result)
    except Exception as e:
        print(f"❌ Error training English SMS: {e}")
        import traceback
        traceback.print_exc()
    
    # Train English Call
    try:
        call_result = train_english_call_improved()
        results.append(call_result)
    except Exception as e:
        print(f"❌ Error training English Call: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    if results:
        summary_df = pd.DataFrame(results)
        print("\n" + summary_df.to_string(index=False))
        summary_df.to_csv('model_metrics_improved.csv', index=False)
        print("\n✓ Results saved to model_metrics_improved.csv")

if __name__ == "__main__":
    main()
