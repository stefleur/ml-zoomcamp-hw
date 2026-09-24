"""Reproduce the answers for 2026 Homework 4."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split


HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "course_lead_scoring_2026.csv")
categorical = df.select_dtypes(include="object").columns.tolist()
numerical = df.select_dtypes(include="number").columns.drop("converted").tolist()

print("Rows:", len(df))
print("Missing values:", df.isna().sum().to_dict())
df[categorical] = df[categorical].fillna("NA")
df[numerical] = df[numerical].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)
print("Split sizes:", len(df_train), len(df_val), len(df_test))

print("\nQ1: Single-feature AUC (direction corrected)")
feature_auc = {}
for feature in numerical:
    auc = roc_auc_score(df_train.converted, df_train[feature])
    feature_auc[feature] = max(auc, 1 - auc)
    print(f"  {feature}: {feature_auc[feature]:.6f}")
print("Answer:", max(feature_auc, key=feature_auc.get))


def train_and_predict(train, validation, c=1.0):
    vectorizer = DictVectorizer()
    x_train = vectorizer.fit_transform(
        train.drop(columns="converted").to_dict(orient="records")
    )
    x_validation = vectorizer.transform(
        validation.drop(columns="converted").to_dict(orient="records")
    )
    model = LogisticRegression(solver="liblinear", C=c, max_iter=1000)
    model.fit(x_train, train.converted)
    return model.predict_proba(x_validation)[:, 1]


val_scores = train_and_predict(df_train, df_val)
val_auc = roc_auc_score(df_val.converted, val_scores)
print(f"\nQ2: Validation AUC = {val_auc:.6f} (rounded: {val_auc:.3f})")

thresholds = np.arange(101) / 100
actual = df_val.converted.to_numpy().astype(bool)
rows = []
for threshold in thresholds:
    predicted = val_scores >= threshold
    true_positive = np.sum(predicted & actual)
    false_positive = np.sum(predicted & ~actual)
    false_negative = np.sum(~predicted & actual)
    precision = true_positive / (true_positive + false_positive) if predicted.any() else 0.0
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    rows.append((threshold, precision, recall, f1))
metrics = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])

valid = metrics.loc[(metrics.precision > 0) | (metrics.recall > 0)].copy()
valid["difference"] = (valid.precision - valid.recall).abs()
intersection = valid.loc[valid.difference.idxmin()]
print(
    "Q3: Nearest precision/recall intersection: "
    f"threshold={intersection.threshold:.2f}, "
    f"precision={intersection.precision:.6f}, "
    f"recall={intersection.recall:.6f}"
)

best_f1 = metrics.loc[metrics.f1.idxmax()]
print(f"Q4: Maximum F1: threshold={best_f1.threshold:.2f}, F1={best_f1.f1:.6f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(metrics.threshold, metrics.precision, label="Precision")
ax.plot(metrics.threshold, metrics.recall, label="Recall")
ax.axvline(intersection.threshold, linestyle="--", color="gray", linewidth=1)
ax.set(xlabel="Threshold", ylabel="Score", title="Validation precision and recall")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "precision_recall.png", dpi=150)
plt.close(fig)


def cross_validation(c):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_index, val_index in kfold.split(df_full_train):
        train_fold = df_full_train.iloc[train_index]
        val_fold = df_full_train.iloc[val_index]
        predictions = train_and_predict(train_fold, val_fold, c=c)
        scores.append(roc_auc_score(val_fold.converted, predictions))
    return np.array(scores)


scores = cross_validation(1.0)
print("\nQ5: Fold AUCs:", [round(float(score), 6) for score in scores])
print(f"Q5: Standard deviation = {scores.std():.6f} (rounded: {scores.std():.3f})")

print("\nQ6: Cross-validation by C")
results = []
for c in [0.000001, 0.001, 1]:
    scores = cross_validation(c)
    results.append((c, round(scores.mean(), 3), round(scores.std(), 3)))
    print(f"  C={c:g}: mean={scores.mean():.6f} ({scores.mean():.3f}), std={scores.std():.6f} ({scores.std():.3f})")
winner = sorted(results, key=lambda result: (-result[1], result[2], result[0]))[0]
print(f"Answer: C={winner[0]:g}")
