import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample dataset: Korean language learners
data = {
    "study_minutes": [30, 60, 20, 45],
    "weekly_reviews": [3, 6, 2, 5],
    "accuracy": [65, 85, 50, 75],
    "next_week_score": [72, 91, 58, 83]
}

df = pd.DataFrame(data)

print("=== Dataset ===")
print(df)

# TODO: Separate features (X) and labels (y)

print("\n=== Features ===")
features = df[["study_minutes", "weekly_reviews", "accuracy"]]
print(features)

print("\n=== Labels ===")
labels = df["next_week_score"]
print(labels)

print("\n=== Shapes ===")
print("X shape:", features.shape)
print("y shape:", labels.shape)

# --------

# Step 1: Create a Linear Regression Model
model =  LinearRegression()

# Step 2: Train the model using features and labels
model.fit(features, labels)

print("Model training completed!")

new_learner = pd.DataFrame({
    "study_minutes": [35],
    "weekly_reviews": [4],
    "accuracy": [70],
})

prediction = model.predict(new_learner)
print("Predicted score:", prediction[0])

print("\n=== Model Parameters ===")

print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

for feature_name, weight in zip(features.columns, model.coef_):
    print(f"{feature_name}: {weight:.4f}")

# Training Error calculation

# Predictions on the SAME data used for training
train_predictions = model.predict(features)

mae = mean_absolute_error(labels, train_predictions)
mse = mean_squared_error(labels, train_predictions)

print("\n=== Training Evaluation ===")

for actual, predicted in zip(labels, train_predictions):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")

print(f"Training MAE: {mae:.6f}")
print(f"Training MSE: {mse:.6f}")