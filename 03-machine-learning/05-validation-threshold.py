
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. Generate synthetic learner data
rng = np.random.default_rng(2026)
n = 600

df = pd.DataFrame({
    "accuracy": rng.integers(35, 96, size=n),
    "weekly_reviews": rng.integers(0, 8, size=n),
    "recent_mistakes": rng.integers(0, 16, size=n)
})

# Simulate the probability of needing review
z = (
    -0.06 * (df["accuracy"] - 65)
    - 0.30 * (df["weekly_reviews"] - 3)
    + 0.24 * (df["recent_mistakes"] - 7)
    + rng.normal(0, 1.4, size=n)
)

true_prob = 1 / (1 + np.exp(-z))

df["needs_review"] = rng.binomial(1, true_prob)

print("=== Dataset ===")
print(df.head())
print("Dataset shape:", df.shape)

# 2. Separate features and labels
X = df[["accuracy", "weekly_reviews", "recent_mistakes"]]
y = df["needs_review"]


# 3. Split into Train / Validation / Test

# Reserve 20% for final testing
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split development data into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev
)

print("\n=== Dataset Shapes ===")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# 4. Train the model

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 5. Evaluate thresholds on VALIDATION data

print("\n=== Validation Threshold Comparison ===")

val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [0.3, 0.5, 0.7]
validation_results = []

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    validation_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(f"\nThreshold: {threshold:.1f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")


# 6. Select the best threshold by Validation F1

best_result = max(
    validation_results,
    key=lambda result: result["f1"]
)

best_threshold = best_result["threshold"]

print("\n=== Selected Threshold ===")
print(f"Best Threshold: {best_threshold:.1f}")
print(f"Validation F1:  {best_result['f1']:.4f}")


# 7. Final evaluation on TEST data

print("\n=== Final Test Evaluation ===")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

print("\nConfusion Matrix:")
print(f"TN={tn}, FP={fp}")
print(f"FN={fn}, TP={tp}")

print(
    f"Accuracy:  "
    f"{accuracy_score(y_test, test_predictions):.4f}"
)
print(
    f"Precision: "
    f"{precision_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    "
    f"{recall_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(
    f"F1-score:  "
    f"{f1_score(y_test, test_predictions, zero_division=0):.4f}"
)

