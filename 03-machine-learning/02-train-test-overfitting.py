import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# 1. Generate synthetic Koreply learner data
rng = np.random.default_rng(42)
n = 600

df = pd.DataFrame({
    "study_minutes": rng.integers(10, 121, size=n),
    "weekly_reviews": rng.integers(0, 8, size=n),
    "accuracy": rng.integers(40, 96, size=n)
})

df["next_week_score"] = np.clip(
    10
    + 0.25 * df["study_minutes"]
    + 2.0 * df["weekly_reviews"]
    + 0.55 * df["accuracy"]
    + 8 * (
    (df["weekly_reviews"] >= 4)
        & (df["accuracy"] >= 75)
    ).astype(int)
    + rng.normal(0, 7, size = n),
    0,
    100
)

print(df.head())
print("Dataset shape:", df.shape)

# 3. Separate features and labels
x = df[["study_minutes", "weekly_reviews", "accuracy"]]
y = df["next_week_score"]

# 4. Split into Training / Validation / Test Sets - after apply Validation

# First split: 80% Development, 20% Test
x_dev, x_test, y_dev, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Second split: 75% Training, 25% Validation
# This split is applied ONLY to the development set.
x_train, x_val, y_train, y_val = train_test_split(
    x_dev,
    y_dev,
    test_size=0.25,
    random_state=42
)

print("\n=== Dataset Shapes ===")
print("Training:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)

# 5. Compare Three Models Using Validation Data

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree (depth=4)": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    ),

    "Decision Tree (unlimited)": DecisionTreeRegressor(
        random_state=42
    )
}

for name, model in models.items():

    # Train the model
    model.fit(x_train, y_train)

    # Predict on training and validation sets
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    # Calculate errors
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    print(f"\n{name}")
    print(f"Training MAE:   {train_mae:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"MAE Gap:        {val_mae - train_mae:.4f}")

# 6. 5-fold cross-validation

print("\n=== 5-Fold Cross-validation ===")

kf = KFold(
    n_splits = 5,
    shuffle=True,
    random_state=42
)

# show the number of samples in each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(x_dev), start=1):
    print(
        f"Fold {fold}: "
        f"Train={len(train_idx)}, "
        f"Validation={len(val_idx)}"
    )

# Compare models using cross-validation
for name, model in models.items():

    scores = cross_validate(
        model,
        x_dev,
        y_dev,
        cv=kf,
        scoring="neg_mean_absolute_error",
        return_train_score=True
    )

    train_maes = -scores["train_score"]
    val_maes = -scores["test_score"]

    print(f"\n{name}")
    print(f"Training MAEs: {train_maes.round(4)}")
    print(f"Validation MAEs: {val_maes.round(4)}")

    print(f"Mean Training MAE: {np.mean(train_maes):.4f}")
    print(f"Mean Validation MAE: {np.mean(val_maes):.4f}")
    print(f"Validation MAE Std: {np.std(val_maes):.4f}")


# 7. Train the Final Model

print("\n=== Final Model Training ===")

# Select the best model based on cross-validation
final_model = LinearRegression()

# Retrain using the entire development set
final_model.fit(x_dev, y_dev)

print("Final model trained on:", x_dev.shape[0], "samples")

# 8. Evaluate on the Test set

print("\n === Final Test Evaluation ===")

# Predict test data (never used for training)
test_predications = final_model.predict(x_test)

# Calculate final test error
test_mae = mean_absolute_error(y_test, test_predications)

print(f"Final Test MAE: {test_mae:.4f}")

# 9. Compare actual vs predicted scores

print("\n=== Sample Predictions ===")

comparison = pd.DataFrame({
    "Actual Score": y_test.to_numpy(),
    "Predicted Score": test_predications
})

print(comparison.head(10))

