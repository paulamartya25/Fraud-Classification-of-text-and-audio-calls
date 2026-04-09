import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TRAINING CALL MODELS WITH SCIKIT-LEARN (No TensorFlow Required)")
print("=" * 80)

# ==================== ENGLISH CALL MODEL ====================
print("\n📱 Training English Call Model...")
try:
    eng_call_df = pd.read_csv('hindi_call_records_dataset.csv')
    
    # Prepare data
    X_train = eng_call_df['text'].astype(str).values
    # Convert labels: genuine=0, fraud=1
    y_train = (eng_call_df['label'].str.lower() == 'fraud').astype(int).values
    
    # Train TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X_train)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_tfidf)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    f1 = f1_score(y_train, y_pred, zero_division=0)
    
    print(f"  ✅ Accuracy:  {accuracy:.4f}")
    print(f"  ✅ Precision: {precision:.4f}")
    print(f"  ✅ Recall:    {recall:.4f}")
    print(f"  ✅ F1 Score:  {f1:.4f}")
    
    # Save model
    joblib.dump(model, 'english_call_model_sklearn.pkl')
    joblib.dump(vectorizer, 'english_call_vectorizer.pkl')
    print("  💾 Saved: english_call_model_sklearn.pkl, english_call_vectorizer.pkl")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# ==================== HINDI CALL MODEL ====================
print("\n📱 Training Hindi Call Model...")
try:
    hi_call_df = pd.read_csv('hindi_call_records_dataset.csv')
    
    # Prepare data
    X_train = hi_call_df['text'].astype(str).values
    # Convert labels: genuine=0, fraud=1
    y_train = (hi_call_df['label'].str.lower() == 'fraud').astype(int).values
    
    # Train TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X_train)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_tfidf)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    f1 = f1_score(y_train, y_pred, zero_division=0)
    
    print(f"  ✅ Accuracy:  {accuracy:.4f}")
    print(f"  ✅ Precision: {precision:.4f}")
    print(f"  ✅ Recall:    {recall:.4f}")
    print(f"  ✅ F1 Score:  {f1:.4f}")
    
    # Save model
    joblib.dump(model, 'hindi_call_model_sklearn.pkl')
    joblib.dump(vectorizer, 'hindi_call_vectorizer.pkl')
    print("  💾 Saved: hindi_call_model_sklearn.pkl, hindi_call_vectorizer.pkl")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# ==================== TELUGU CALL MODEL ====================
print("\n📱 Training Telugu Call Model...")
try:
    te_call_df = pd.read_csv('telugu_call_dataset.csv')
    
    # Prepare data - use 'transcript' column for Telugu
    X_train = te_call_df['transcript'].astype(str).values
    # Convert labels: real=0, fraud=1
    y_train = (te_call_df['label'].str.lower() == 'fraud').astype(int).values
    
    # Train TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X_train)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_tfidf)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, zero_division=0)
    recall = recall_score(y_train, y_pred, zero_division=0)
    f1 = f1_score(y_train, y_pred, zero_division=0)
    
    print(f"  ✅ Accuracy:  {accuracy:.4f}")
    print(f"  ✅ Precision: {precision:.4f}")
    print(f"  ✅ Recall:    {recall:.4f}")
    print(f"  ✅ F1 Score:  {f1:.4f}")
    
    # Save model
    joblib.dump(model, 'telugu_call_model_sklearn.pkl')
    joblib.dump(vectorizer, 'telugu_call_vectorizer.pkl')
    print("  💾 Saved: telugu_call_model_sklearn.pkl, telugu_call_vectorizer.pkl")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

print("\n" + "=" * 80)
print("✅ All Call Models Trained with scikit-learn (Ready for Streamlit Cloud)")
print("=" * 80)
