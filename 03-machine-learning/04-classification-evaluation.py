
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
rng = np.random.default_rng(42)
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

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n=== Dataset Split ===")
print("Training:", X_train.shape)
print("Test:", X_test.shape)


# 4. Train Logistic Regression model

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\n=== Model Training ===")
print("Training completed!")

# 5. Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Get predicted probabilities for class 1
test_probabilities = model.predict_proba(X_test)[:, 1]

print("\n=== Sample Predictions ===")
for actual, predicted, prob in zip(
    y_test.iloc[:5],
    test_predictions[:5],
    test_probabilities[:5]
):
    print(
        f"Actual: {actual}, "
        f"Predicted: {predicted}, "
        f"P(Review=1): {prob:.2%}"
    )



# 6. Evaluate model performance

print("\n=== Classification Evaluation ===")

cm = confusion_matrix(y_test, test_predictions)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print("\nConfusion Matrix:")
print(cm)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1-score:          {f1:.4f}")



# 7. Classification Threshold Experiment

print("\n=== Threshold Comparison ===")

thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:

    # Convert predicted probabilities to classes
    threshold_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    # Calculate evaluation metrics
    accuracy_t = accuracy_score(
        y_test, threshold_predictions
    )

    precision_t = precision_score(
        y_test, threshold_predictions,
        zero_division=0
    )

    recall_t = recall_score(
        y_test, threshold_predictions
    )

    f1_t = f1_score(
        y_test, threshold_predictions
    )

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        threshold_predictions,
        labels=[0, 1]
    ).ravel()

    print(f"\n--- Threshold: {threshold:.1f} ---")
    print(f"Accuracy:  {accuracy_t:.4f}")
    print(f"Precision: {precision_t:.4f}")
    print(f"Recall:    {recall_t:.4f}")
    print(f"F1-score:  {f1_t:.4f}")

    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")
