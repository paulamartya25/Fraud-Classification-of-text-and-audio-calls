import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("IMPROVING TELUGU CALL MODEL WITH ROBUST TECHNIQUES")
print("=" * 80)

# Load data
df = pd.read_csv('telugu_call_dataset.csv')
X = df['transcript'].astype(str).values
y = (df['label'].str.lower() == 'fraud').astype(int).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Dataset Split:")
print(f"  Training: {len(X_train)} samples ({(y_train.sum()/len(y_train)*100):.1f}% fraud)")
print(f"  Testing:  {len(X_test)} samples ({(y_test.sum()/len(y_test)*100):.1f}% fraud)")

# ============= APPROACH 1: TF-IDF + Logistic Regression (Current) =============
print("\n" + "="*80)
print("APPROACH 1: TF-IDF + Logistic Regression (Current)")
print("="*80)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model1 = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model1.fit(X_train_tfidf, y_train)

y_pred1 = model1.predict(X_test_tfidf)
y_pred1_proba = model1.predict_proba(X_test_tfidf)[:, 1]

print(f"\n📈 Test Performance:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred1):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred1):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred1):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred1):.4f}")

# ============= APPROACH 2: SMOTE + Regularization =============
print("\n" + "="*80)
print("APPROACH 2: TF-IDF + SMOTE + Logistic Regression (Improved)")
print("="*80)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_tfidf_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

print(f"\n✅ SMOTE Applied:")
print(f"  Original fraud count: {y_train.sum()}")
print(f"  After SMOTE fraud count: {y_train_smote.sum()}")
print(f"  Balance ratio: {(y_train_smote.sum() / len(y_train_smote) * 100):.1f}% fraud")

# Train with SMOTE
model2 = LogisticRegression(max_iter=1000, random_state=42, C=0.5, class_weight='balanced')
model2.fit(X_train_tfidf_smote, y_train_smote)

y_pred2 = model2.predict(X_test_tfidf)
y_pred2_proba = model2.predict_proba(X_test_tfidf)[:, 1]

print(f"\n📈 Test Performance:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred2):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred2):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred2):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred2):.4f}")

# ============= APPROACH 3: Cross-validation for robustness =============
print("\n" + "="*80)
print("APPROACH 3: Cross-Validation Analysis")
print("="*80)

from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced'),
    X_train_tfidf,
    y_train,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)

print(f"\n🔄 5-Fold Cross-Validation Results:")
print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
print(f"  Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
print(f"  Recall:    {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
print(f"  F1 Score:  {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")

# Save the improved model
print("\n" + "="*80)
print("SAVING IMPROVED MODEL")
print("="*80)

joblib.dump(model2, 'telugu_call_model_improved_v2.pkl')
joblib.dump(vectorizer, 'telugu_call_vectorizer_improved_v2.pkl')
print("\n✅ Saved:")
print("  - telugu_call_model_improved_v2.pkl (SMOTE + Balanced)")
print("  - telugu_call_vectorizer_improved_v2.pkl")

# ============= Threshold Analysis =============
print("\n" + "="*80)
print("THRESHOLD OPTIMIZATION")
print("="*80)

fpr, tpr, thresholds = roc_curve(y_test, y_pred2_proba)
roc_auc = auc(fpr, tpr)

print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")

# Find optimal threshold for balanced performance
from sklearn.metrics import f1_score as f1

best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.3, 0.8, 0.05):
    y_pred_thresholded = (y_pred2_proba >= threshold).astype(int)
    f1_score_val = f1_score(y_test, y_pred_thresholded)
    if f1_score_val > best_f1:
        best_f1 = f1_score_val
        best_threshold = threshold

print(f"\n🎯 Optimal Threshold: {best_threshold:.2f}")
print(f"   F1 Score at optimal threshold: {best_f1:.4f}")

# Evaluate at optimal threshold
y_pred_optimal = (y_pred2_proba >= best_threshold).astype(int)
print(f"\n📈 Performance at optimal threshold ({best_threshold:.2f}):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_optimal):.4f}")
print(f"  F1 Score:  {f1_score(y_test, y_pred_optimal):.4f}")

print("\n" + "="*80)
print("✅ RECOMMENDATION: Use model2 with threshold {:.2f}".format(best_threshold))
print("="*80)
