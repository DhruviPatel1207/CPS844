# Wine Quality Analysis and Prediction

This project applies machine learning techniques to analyze and predict wine quality using the UCI Wine Quality dataset. Both classification and regression approaches are explored. The impact of feature selection using Principal Component Analysis (PCA) is also evaluated.

# Dataset

- Source: UCI Machine Learning Repository
- File: winequality-red.csv
- Total samples: 1,599
- Features: 11 physicochemical attributes (e.g., alcohol, pH, sulphates)
- Targets:
  - quality: integer score from 0 to 10
  - quality_class: binary label ('Good' if score ≥6, else 'Bad')

# Objectives

- Classify wines as Good or Bad based on physicochemical properties
- Predict the exact quality score using regression
- Evaluate the effect of PCA-based feature selection on model performance

# Tools and Libraries

- pandas, seaborn, matplotlib
- scikit-learn (models, preprocessing, metrics, PCA)

# Models Used

Classification:
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest Classifier
- Support Vector Classifier

Regression:
- Linear Regression
- Random Forest Regressor

# Workflow

1. Load and preprocess the data
2. Encode the binary classification target
3. Split into training and testing sets
4. Standardize features
5. Apply PCA for dimensionality reduction
6. Train and evaluate models on:
   - All features
   - PCA-selected features
7. Compare results and visualize correlation matrix

# Evaluation Metrics

Classification:
- Accuracy
- Classification report (Precision, Recall, F1)

Regression:
- Mean Squared Error (MSE)
- R² Score

# Results Summary

- Classification and regression models were evaluated with and without PCA.
- PCA reduced features to 5 principal components while retaining most variance.
- Comparison tables show performance changes due to feature selection.
- Correlation matrix was visualized to assess feature relationships.

# Usage

To run the analysis:
1. Make sure all required libraries are installed.
2. Run the Python script file in your environment.
3. Review the printed evaluation metrics and visualizations.


