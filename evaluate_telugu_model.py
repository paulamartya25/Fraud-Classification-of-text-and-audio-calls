import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("=" * 80)
print("EVALUATING TELUGU CALL MODEL")
print("=" * 80)

# Load data
df = pd.read_csv('telugu_call_dataset.csv')
X = df['transcript'].astype(str).values
y = (df['label'].str.lower() == 'fraud').astype(int).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train fraud %: {(y_train.sum() / len(y_train) * 100):.1f}%")
print(f"Test fraud %: {(y_test.sum() / len(y_test) * 100):.1f}%")

# Train new model
print("\n🔄 Training new Telugu Call Model...")
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_train_pred = model.predict(X_train_tfidf)
y_test_pred = model.predict(X_test_tfidf)

print("\n📊 TRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred, zero_division=0):.4f}")
print(f"  F1 Score:  {f1_score(y_train, y_train_pred, zero_division=0):.4f}")

print("\n📊 TEST SET PERFORMANCE:")
print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_test_pred, zero_division=0):.4f}")

print("\n📋 CLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Real', 'Fraud']))

# Test individual predictions
print("\n🧪 SAMPLE PREDICTIONS (First 10 test samples):")
for i in range(min(10, len(X_test))):
    actual = 'FRAUD' if y_test[i] else 'REAL'
    pred = 'FRAUD' if y_test_pred[i] else 'REAL'
    prob = model.predict_proba(X_test_tfidf[i])[0][1]
    match = "✅" if actual == pred else "❌"
    print(f"  {match} Sample {i+1}: Expected {actual}, Predicted {pred} (fraud prob: {prob:.2%})")
    print(f"      Text: {X_test[i][:70]}...")
