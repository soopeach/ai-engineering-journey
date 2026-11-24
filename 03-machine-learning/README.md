# Machine Learning Fundamentals

## Overview

This module documents my hands-on learning of machine learning fundamentals, including model training, evaluation, overfitting, cross-validation, and model selection.

The experiments use a synthetic dataset representing learners of **Koreply**, my AI-powered Korean language learning project.

## Learning Objectives

- Understand supervised and unsupervised learning.
- Distinguish regression from classification.
- Identify features (X) and targets (y).
- Train and evaluate machine learning models.
- Understand overfitting and generalisation.
- Apply train/validation/test splitting.
- Perform K-Fold cross-validation.
- Select and retrain a final model.

## 1. Features, Labels, and Linear Regression

**File:** `01-features-labels.py`

I created a simple dataset containing learner activity information.

**Input features:**
- Daily study time
- Weekly review frequency
- Current exercise accuracy

**Target:**
- Next week's test score

I trained a Linear Regression model using scikit-learn and generated predictions for a new learner.

I also examined the learned coefficients, intercept, and training errors using MAE and MSE.

### Key Learning

A model can achieve nearly zero training error without necessarily generalising well to unseen data.

## 2. Overfitting and Model Comparison

**File:** `02-train-test-overfitting.py`

I generated 600 synthetic learner records and compared three models:

- Linear Regression
- Decision Tree (max_depth=4)
- Decision Tree (unlimited depth)

The unrestricted Decision Tree achieved zero training MAE but substantially higher validation MAE, indicating strong overfitting.

## 3. Train, Validation, and Test

I divided the dataset into:

| Dataset | Samples | Purpose |
|---|---:|---|
| Training | 360 | Learn model parameters |
| Validation | 120 | Compare model performance |
| Test | 120 | Evaluate the final model |

I learned that validation data supports model selection and hyperparameter tuning, while the test set should be reserved for final evaluation.

## 4. Five-Fold Cross-Validation

I performed 5-Fold Cross-validation using the 480-sample development dataset.

Each fold used:
- 384 samples for training
- 96 samples for validation

### Results

| Model | Mean Train MAE | Mean Validation MAE | Validation Std |
|---|---:|---:|---:|
| Linear Regression | 5.7308 | 5.7799 | 0.3976 |
| Decision Tree (depth=4) | 5.8789 | 7.1064 | 0.2842 |
| Decision Tree (unlimited) | 0.0000 | 7.9658 | 0.8569 |

### Model Selection

Linear Regression achieved the lowest mean validation MAE.

Cross-validation provided a more robust comparison by evaluating each model across multiple data partitions rather than relying on a single validation split.

## 5. Final Model Training

After selecting Linear Regression, I trained a new model using all 480 development samples.

The remaining 120 test samples were used for evaluation.

**Final Test MAE:** Pending verification after correcting the final training split.

### Evaluation Limitation

The test split had already been examined in an earlier educational experiment. Therefore, this result is a demonstration of the evaluation workflow rather than a completely independent, unbiased final performance estimate.

## 6. Key Takeaways

1. Model training and prediction are different operations.
2. Training error alone cannot establish generalisation performance.
3. Overfitting occurs when a model fits training data too closely and performs poorly on unseen data.
4. Validation is used to select models and hyperparameters.
5. K-Fold cross-validation reduces dependence on a single data split.
6. Final model training can use all available development data after model selection.
7. Test data should remain separate from model development.

## 7. Connection to Koreply

These fundamentals will support future work on:

- Korean grammar correction datasets
- Text classification
- NLP and Transformer models
- Supervised fine-tuning of pretrained language models
- Model evaluation and generalisation
- Detecting overfitting in language models

## Technologies

Python · NumPy · Pandas · scikit-learn

**Dataset:** Synthetic educational data, not real user data.

---

## Day 2 — Classification and Logistic Regression

### Overview

On Day 2, I explored binary classification using Logistic Regression and learned how to evaluate classification models with different performance metrics.

The experiments use synthetic learner activity data inspired by Koreply, my AI-powered Korean language learning application.

### 1. Logistic Regression

**File:** `03-logistic-regression.py`

I trained a Logistic Regression model to predict whether a learner needs additional review.

**Features:**
- Exercise accuracy
- Weekly review frequency
- Number of recent mistakes

**Target:**
- `needs_review`: 0 (No) or 1 (Yes)

I learned the difference between:

- `predict()`: Returns predicted class labels.
- `predict_proba()`: Returns estimated class probabilities.

I also learned how the sigmoid function maps model scores to probabilities between 0 and 1.

### 2. Classification Evaluation

**File:** `04-classification-evaluation.py`

I learned to evaluate binary classification models using:

- Confusion Matrix (TP, TN, FP, FN)
- Accuracy
- Precision
- Recall
- F1-score

I then trained a Logistic Regression model on 600 synthetic learner records and evaluated it using a train/test split.

**Model evaluation results (initial dataset):**

| Metric | Value |
|---|---:|
| Accuracy | 0.6917 |
| Precision | 0.6714 |
| Recall | 0.7705 |
| F1-score | 0.7176 |

### 3. Classification Threshold Experiment

I compared different classification thresholds using the same trained model.

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.5957 | 0.9180 | 0.7226 |
| 0.5 | 0.6714 | 0.7705 | 0.7176 |
| 0.7 | 0.7419 | 0.3770 | 0.5000 |

**Key observation:**

Lowering the threshold increased recall but also increased false positives.

This experiment helped me understand the trade-off between identifying more positive cases and avoiding incorrect positive predictions.

### 4. Validation-Based Threshold Selection

**File:** `05-validation-threshold.py`

I generated a new synthetic dataset with 600 learner records and divided it into:

| Dataset | Samples | Purpose |
|---|---:|---|
| Training | 360 | Fit the model |
| Validation | 120 | Select the classification threshold |
| Test | 120 | Evaluate the selected model and threshold |

I compared thresholds using the validation set rather than selecting them based on test performance.

**Validation results:**

| Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0.3 | 0.5957 | 0.9492 | 0.7320 |
| 0.5 | 0.6923 | 0.7627 | 0.7258 |
| 0.7 | 0.7188 | 0.3898 | 0.5055 |

**Selected threshold:** `0.3`

It achieved the highest validation F1-score among the tested thresholds.

### 5. Final Test Evaluation

Using the selected threshold of 0.3, I evaluated the model on the held-out test dataset.

**Confusion Matrix:**

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 23 | 37 |
| Actual 1 | 2 | 58 |

**Final results:**

| Metric | Value |
|---|---:|
| Accuracy | 0.6750 |
| Precision | 0.6105 |
| Recall | 0.9667 |
| F1-score | 0.7484 |

The model identified 58 out of 60 learners who needed review, but also incorrectly flagged 37 learners who did not need additional review.

This demonstrates why classification performance should not be evaluated using accuracy alone.

### 6. Key Learnings

1. Logistic Regression is a classification algorithm despite its name.
2. Predicted probabilities and predicted classes are different outputs.
3. Classification thresholds determine how probabilities are converted into class labels.
4. Precision measures how reliable positive predictions are.
5. Recall measures how many actual positive cases are identified.
6. F1-score combines precision and recall using their harmonic mean.
7. Validation data should be used for model and threshold selection.
8. Test data should be reserved for final performance evaluation.

### 7. Connection to Koreply

These experiments establish foundational concepts for developing Korean grammar error detection features in Koreply.

Potential applications include:

- Detecting whether a Korean sentence contains a grammar error.
- Evaluating false positives and false negatives in error detection.
- Selecting decision thresholds based on product requirements.
- Measuring how reliably a model identifies errors.
- Preparing for NLP and pretrained language model fine-tuning.

### Limitations

All experiments use synthetic educational data rather than real Korean learner data.

The models operate on numerical learner activity features, not Korean text. Real grammar error detection will require text preprocessing, appropriately labelled language datasets, and task-specific evaluation.

**Technologies:** Python, NumPy, Pandas, scikit-learn