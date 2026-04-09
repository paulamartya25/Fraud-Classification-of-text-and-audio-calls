"""
Comprehensive Model Metrics Report
Evaluates all models: F1 Score, Precision, Recall, Accuracy, and more
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import warnings

warnings.filterwarnings('ignore')


# ===================== LOAD DATASETS =====================
def load_datasets():
    """Load all datasets"""
    datasets = {}
    
    try:
        datasets['hindi_sms'] = pd.read_csv('hindi_sms_dataset.csv')
        print("✓ Hindi SMS Dataset loaded")
    except:
        print("✗ Hindi SMS Dataset not found")
    
    try:
        datasets['telugu_sms'] = pd.read_csv('telugu_sms_dataset.csv')
        print("✓ Telugu SMS Dataset loaded")
    except:
        print("✗ Telugu SMS Dataset not found")
    
    try:
        datasets['hindi_call'] = pd.read_csv('hindi_call_records_dataset.csv')
        print("✓ Hindi Call Dataset loaded")
    except:
        print("✗ Hindi Call Dataset not found")
    
    try:
        datasets['telugu_call'] = pd.read_csv('telugu_call_dataset.csv')
        print("✓ Telugu Call Dataset loaded")
    except:
        print("✗ Telugu Call Dataset not found")
    
    try:
        datasets['english_sms'] = pd.read_csv('english_sms_dataset.csv')
        print("✓ English SMS Dataset loaded")
    except:
        print("✗ English SMS Dataset not found")
    
    try:
        datasets['english_call'] = pd.read_csv('english_call_dataset.csv')
        print("✓ English Call Dataset loaded")
    except:
        print("✗ English Call Dataset not found")
    
    return datasets


# ===================== ENGLISH MODELS =====================
def evaluate_english_sms_model(df=None):
    """Evaluate English SMS Model"""
    print("\n" + "="*80)
    print("ENGLISH SMS MODEL (msg_model.h5)")
    print("="*80)
    
    try:
        # Load model and tokenizer
        msg_model = load_model("msg_model.h5")
        with open("msg_tokenizer.pkl", 'rb') as f:
            msg_tokenizer = pickle.load(f)
        
        if df is None or df.empty:
            print("✗ English SMS dataset not available")
            return None
        
        # Prepare test data
        X = df['Text']
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tokenize and pad sequences
        test_sequences = msg_tokenizer.texts_to_sequences(X_test)
        test_padded = pad_sequences(test_sequences, maxlen=200, padding="post")
        
        # Make predictions
        y_pred_proba = msg_model.predict(test_padded, verbose=0)
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        
        # Convert labels to binary
        y_test_binary = (y_test == 'fraud').astype(int).values
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test_binary, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test_binary, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_test_binary, y_pred_proba)
        
        print(f"Model Type: LSTM + Embedding + Bidirectional + Dropout")
        print(f"Test Set Size: {len(X_test)}")
        print(f"\n{'Metric':<20} {'Score':>20}")
        print("-" * 42)
        print(f"{'Accuracy':<20} {accuracy:>20.4f}")
        print(f"{'Precision':<20} {precision:>20.4f}")
        print(f"{'Recall':<20} {recall:>20.4f}")
        print(f"{'F1-Score':<20} {f1:>20.4f}")
        print(f"{'ROC-AUC':<20} {auc:>20.4f}")
        
        print("\n" + "-" * 42)
        print("Classification Report:")
        print(classification_report(y_test_binary, y_pred, target_names=['normal', 'fraud']))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test_binary, y_pred)
        print(cm)
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'ROC-AUC': auc}
        
    except Exception as e:
        print(f"✗ Error evaluating English SMS model: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_english_call_model(df=None):
    """Evaluate English Call Model"""
    print("\n" + "="*80)
    print("ENGLISH CALL MODEL (call_model.h5)")
    print("="*80)
    
    try:
        # Load model and tokenizer
        call_model = load_model("call_model.h5")
        with open("call_tokenizer.pkl", 'rb') as f:
            call_tokenizer = pickle.load(f)
        
        if df is None or df.empty:
            print("✗ English Call dataset not available")
            return None
        
        # Prepare test data
        X = df['transcript']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Tokenize and pad sequences
        test_sequences = call_tokenizer.texts_to_sequences(X_test)
        test_padded = pad_sequences(test_sequences, maxlen=200, padding="post")
        
        # Make predictions
        y_pred_proba = call_model.predict(test_padded, verbose=0)
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        
        # Convert labels to binary
        y_test_binary = (y_test == 'fraud').astype(int).values
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test_binary, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test_binary, y_pred, average='binary', zero_division=0)
        auc = roc_auc_score(y_test_binary, y_pred_proba)
        
        print(f"Model Type: LSTM + Embedding + Bidirectional + Dropout")
        print(f"Test Set Size: {len(X_test)}")
        print(f"\n{'Metric':<20} {'Score':>20}")
        print("-" * 42)
        print(f"{'Accuracy':<20} {accuracy:>20.4f}")
        print(f"{'Precision':<20} {precision:>20.4f}")
        print(f"{'Recall':<20} {recall:>20.4f}")
        print(f"{'F1-Score':<20} {f1:>20.4f}")
        print(f"{'ROC-AUC':<20} {auc:>20.4f}")
        
        print("\n" + "-" * 42)
        print("Classification Report:")
        print(classification_report(y_test_binary, y_pred, target_names=['normal', 'fraud']))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test_binary, y_pred)
        print(cm)
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'ROC-AUC': auc}
        
    except Exception as e:
        print(f"✗ Error evaluating English Call model: {e}")
        import traceback
        traceback.print_exc()
        return None


# ===================== HINDI SMS MODEL =====================
def evaluate_hindi_sms_model(df):
    """Evaluate Hindi SMS Model"""
    print("\n" + "="*80)
    print("HINDI SMS MODEL (TF-IDF + Naive Bayes)")
    print("="*80)
    
    try:
        model = joblib.load("hindi_fraud_classifier.pkl")
        
        if df is None or df.empty:
            print("✗ Hindi SMS dataset not available")
            return None
        
        # Prepare test data
        X = df['Text']
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Model Type: TF-IDF Vectorizer + Naive Bayes")
        print(f"Test Set Size: {len(X_test)}")
        print(f"\n{'Metric':<20} {'Score':>20}")
        print("-" * 42)
        print(f"{'Accuracy':<20} {accuracy:>20.4f}")
        print(f"{'Precision':<20} {precision:>20.4f}")
        print(f"{'Recall':<20} {recall:>20.4f}")
        print(f"{'F1-Score':<20} {f1:>20.4f}")
        
        print("\n" + "-" * 42)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        
    except Exception as e:
        print(f"✗ Error evaluating Hindi SMS model: {e}")
        return None


# ===================== HINDI CALL MODEL =====================
def evaluate_hindi_call_model(df):
    """Evaluate Hindi Call Model"""
    print("\n" + "="*80)
    print("HINDI CALL MODEL (TF-IDF + Logistic Regression)")
    print("="*80)
    
    try:
        model = joblib.load("hindi_fraud_model.pkl")
        vectorizer = joblib.load("hindi_vectorizer.pkl")
        
        if df is None or df.empty:
            print("✗ Hindi Call dataset not available")
            return None
        
        # Prepare test data
        X = vectorizer.transform(df['text'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Model Type: TF-IDF Vectorizer + Logistic Regression")
        print(f"Test Set Size: {len(X_test)}")
        print(f"\n{'Metric':<20} {'Score':>20}")
        print("-" * 42)
        print(f"{'Accuracy':<20} {accuracy:>20.4f}")
        print(f"{'Precision':<20} {precision:>20.4f}")
        print(f"{'Recall':<20} {recall:>20.4f}")
        print(f"{'F1-Score':<20} {f1:>20.4f}")
        
        print("\n" + "-" * 42)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        
    except Exception as e:
        print(f"✗ Error evaluating Hindi Call model: {e}")
        return None


# ===================== TELUGU SMS MODEL =====================
def evaluate_telugu_sms_model(df):
    """Evaluate Telugu SMS Model"""
    print("\n" + "="*80)
    print("TELUGU SMS MODEL (TF-IDF + Naive Bayes)")
    print("="*80)
    
    try:
        model = joblib.load("telugu_fraud_classifier.pkl")
        
        if df is None or df.empty:
            print("✗ Telugu SMS dataset not available")
            return None
        
        # Prepare test data
        X = df['Text']
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Model Type: TF-IDF Vectorizer + Naive Bayes")
        print(f"Test Set Size: {len(X_test)}")
        print(f"\n{'Metric':<20} {'Score':>20}")
        print("-" * 42)
        print(f"{'Accuracy':<20} {accuracy:>20.4f}")
        print(f"{'Precision':<20} {precision:>20.4f}")
        print(f"{'Recall':<20} {recall:>20.4f}")
        print(f"{'F1-Score':<20} {f1:>20.4f}")
        
        print("\n" + "-" * 42)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        
    except Exception as e:
        print(f"✗ Error evaluating Telugu SMS model: {e}")
        return None


# ===================== TELUGU CALL MODEL =====================
def evaluate_telugu_call_model(df):
    """Evaluate Telugu Call Model (LSTM)"""
    print("\n" + "="*80)
    print("TELUGU CALL MODEL (LSTM + Embedding)")
    print("="*80)
    
    try:
        model = load_model("telugu_call_classifier.h5")
        with open("tokenizer.pkl", 'rb') as f:
            tokenizer = pickle.load(f)
        
        if df is None or df.empty:
            print("✗ Telugu Call dataset not available")
            return None
        
        # Prepare data
        df['label'] = df['label'].map({'fraud': 1, 'real': 0})
        X = df['transcript'].astype(str).values
        y = df['label'].values
        
        # Tokenize and pad
        sequences = tokenizer.texts_to_sequences(X)
        max_len = max(len(seq) for seq in sequences)
        X_pad = pad_sequences(sequences, maxlen=max_len, padding='post')
        
        X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"Model Type: LSTM + Embedding + Dropout")
        print(f"Test Set Size: {len(X_test)}")
        print(f"Max Sequence Length: {max_len}")
        print(f"\n{'Metric':<20} {'Score':>20}")
        print("-" * 42)
        print(f"{'Accuracy':<20} {accuracy:>20.4f}")
        print(f"{'Precision':<20} {precision:>20.4f}")
        print(f"{'Recall':<20} {recall:>20.4f}")
        print(f"{'F1-Score':<20} {f1:>20.4f}")
        print(f"{'ROC-AUC':<20} {auc:>20.4f}")
        
        print("\n" + "-" * 42)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'ROC-AUC': auc}
        
    except Exception as e:
        print(f"✗ Error evaluating Telugu Call model: {e}")
        return None


# ===================== SUMMARY REPORT =====================
def generate_summary_report(metrics_dict):
    """Generate a summary report of all models"""
    print("\n\n" + "="*80)
    print("SUMMARY REPORT - ALL MODELS")
    print("="*80)
    
    summary_df = pd.DataFrame(metrics_dict).T
    print(summary_df.to_string())
    
    print("\n" + "="*80)
    print("MODEL RANKINGS BY F1-SCORE")
    print("="*80)
    
    if 'F1' in summary_df.columns:
        ranking = summary_df['F1'].sort_values(ascending=False)
        for idx, (model, score) in enumerate(ranking.items(), 1):
            print(f"{idx}. {model:<40} F1-Score: {score:.4f}")
    
    return summary_df


# ===================== MAIN EXECUTION =====================
def main():
    print("\n" + "="*80)
    print("🔍 FRAUD DETECTION MODELS - COMPREHENSIVE METRICS REPORT")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = load_datasets()
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Evaluate all models
    print("\n\nEvaluating all models...\n")
    
    all_metrics['HINDI_SMS'] = evaluate_hindi_sms_model(datasets.get('hindi_sms'))
    all_metrics['TELUGU_SMS'] = evaluate_telugu_sms_model(datasets.get('telugu_sms'))
    all_metrics['HINDI_CALL'] = evaluate_hindi_call_model(datasets.get('hindi_call'))
    all_metrics['TELUGU_CALL'] = evaluate_telugu_call_model(datasets.get('telugu_call'))
    all_metrics['ENGLISH_SMS'] = evaluate_english_sms_model(datasets.get('english_sms'))
    all_metrics['ENGLISH_CALL'] = evaluate_english_call_model(datasets.get('english_call'))
    
    # Clean up None values
    all_metrics = {k: v for k, v in all_metrics.items() if v is not None}
    
    # Generate summary
    if all_metrics:
        summary_df = generate_summary_report(all_metrics)
        
        # Save to CSV
        summary_df.to_csv('model_metrics_summary.csv')
        print("\n✓ Summary saved to 'model_metrics_summary.csv'")
    
    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
