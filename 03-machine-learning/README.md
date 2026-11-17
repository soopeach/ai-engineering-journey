# Machine Learning Fundamentals — Day 1

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