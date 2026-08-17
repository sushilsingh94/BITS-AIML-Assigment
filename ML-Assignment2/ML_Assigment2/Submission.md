# Machine Learning - Assignment 2

**Name:** Sushil Kumar
**BITS ID:** 2025ac05637
**Programme:** M.Tech (AIML)
**Course:** Machine Learning
**Submission Date:** 18-Aug-2026

---

## 1. GitHub Repository Link

**Repository:** [https://github.com/sushilsingh94/BITS-AIML-Assigment/tree/main/ML-Assignment2/ML_Assigment2](https://github.com/sushilsingh94/BITS-AIML-Assigment/tree/main/ML-Assignment2/ML_Assigment2)

The repository contains:
- Complete source code (app.py, model/train_models.py)
- requirements.txt
- README.md
- test_data.csv

---

## 2. Live Streamlit App Link

**Streamlit App:** [https://bits-aiml-assigment-sushilsingh94.streamlit.app/](https://bits-aiml-assigment-sushilsingh94.streamlit.app/)

The app opens an interactive frontend with CSV upload, model selection dropdown, evaluation metrics display, and confusion matrix visualization.

---

## 3. Screenshot

*Screenshot of assignment execution on BITS Virtual Lab:*

![BITS Virtual Lab Screenshot](screenshots/bits_lab_screenshot.png)

*Screenshot of Streamlit App - Home Page:*

![Streamlit App Home](screenshots/streamlit_app_home.png)

*Screenshot of Streamlit App - Model Comparison:*

![Streamlit App Comparison](screenshots/streamlit_app_comparison.png)

---

## 4. GitHub README Content

### a. Problem Statement

The goal of this project is to classify breast tumors as **Malignant** or **Benign** based on features extracted from digitized images of fine needle aspirates (FNA) of breast masses. This is a binary classification problem where we implement and compare five different machine learning models on the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository.

The five models implemented are: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors, Gaussian Naive Bayes, and Random Forest. Each model is evaluated using six standard classification metrics — Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). A Streamlit web application is built to interactively explore model results, upload test data, and compare model performance visually.

### b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

**Total Instances:** 569

**Number of Features:** 30 (all numeric, continuous)

**Target Variable:** Diagnosis — Malignant (0) or Benign (1)

**Class Distribution:** 212 Malignant, 357 Benign

**Train/Test Split:** 80/20 stratified split (455 train, 114 test)

The dataset contains features computed from digitized images of fine needle aspirates of breast masses. For each cell nucleus in the image, ten real-valued properties are measured:

1. Radius (mean of distances from center to points on the perimeter)
2. Texture (standard deviation of gray-scale values)
3. Perimeter
4. Area
5. Smoothness (local variation in radius lengths)
6. Compactness (perimeter^2 / area - 1.0)
7. Concavity (severity of concave portions of the contour)
8. Concave Points (number of concave portions of the contour)
9. Symmetry
10. Fractal Dimension (coastline approximation - 1)

For each of these 10 properties, the mean, standard error, and "worst" (mean of the three largest values) are computed, giving a total of 30 features.

**Preprocessing:** All features were standardized using StandardScaler (zero mean, unit variance) before model training.

### c. GitHub Repository Link

**Repository:** [https://github.com/sushilsingh94/BITS-AIML-Assigment/tree/main/ML-Assignment2/ML_Assigment2](https://github.com/sushilsingh94/BITS-AIML-Assigment/tree/main/ML-Assignment2/ML_Assigment2)

The repository contains the following structure:

```
ML_Assigment2/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test data (114 samples)
├── train_data.csv          # Training data (455 samples)
├── model_results.csv       # Model evaluation results
└── model/
    ├── train_models.py         # Training script for all models
    ├── scaler.pkl              # Saved StandardScaler
    ├── logistic_regression.pkl # Saved Logistic Regression model
    ├── decision_tree.pkl       # Saved Decision Tree model
    ├── knn.pkl                 # Saved KNN model
    ├── naive_bayes.pkl         # Saved Naive Bayes model
    └── random_forest.pkl       # Saved Random Forest model
```

### d. Models Used

#### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9929 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

#### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed the best overall on this dataset. It achieved the highest Accuracy (0.9825), AUC (0.9954), Precision (0.9861), F1 (0.9861), and MCC (0.9623). This makes sense because the breast cancer dataset has linearly separable classes to a large extent — the features after standardization create a space where a linear decision boundary works really well. It only misclassified 2 out of 114 test samples. The high MCC of 0.9623 confirms that the model is performing well even considering class imbalance. |
| Decision Tree | Decision Tree had the lowest performance among all five models. Its Accuracy was 0.9211 and AUC was 0.9163, both the worst in the group. The model misclassified 9 out of 114 samples. Decision trees tend to overfit on training data and create sharp boundaries that don't generalize as well. Even with max_depth=5 to control overfitting, the rigid axis-aligned splits struggle to capture the smooth decision boundary that linear models handle better on this dataset. The MCC of 0.8341 is still decent but noticeably lower than the other models. |
| kNN | kNN achieved perfect Recall (1.0000), meaning it correctly identified every Benign case in the test set without missing a single one. Its Accuracy (0.9737) and F1 (0.9796) were the second best after Logistic Regression. The 3 misclassifications were all Malignant cases predicted as Benign, which explains why Precision (0.9600) is slightly lower. kNN works well here because the standardized features create clusters where similar cases are genuinely close together. With k=7, it smooths out noise while still capturing the local structure of the data. |
| Naive Bayes | Naive Bayes gave moderate results with an Accuracy of 0.9298 and equal Precision and Recall (0.9444). Its AUC (0.9868) was actually quite high, suggesting the probabilistic rankings are good even if the hard predictions have some errors. The assumption of feature independence (which Naive Bayes relies on) doesn't fully hold for this dataset since many features are correlated — for example, radius, perimeter, and area are inherently related. This explains why the hard classification metrics are somewhat lower compared to Logistic Regression and kNN, which don't make such assumptions. |
| Random Forest (Ensemble) | Random Forest performed solidly with an Accuracy of 0.9561 and the second-highest AUC (0.9929). As an ensemble of 150 decision trees, it overcomes the individual decision tree's tendency to overfit. The AUC being so close to Logistic Regression (0.9929 vs 0.9954) shows that the ensemble's probability estimates are quite reliable. However, its Accuracy and F1 are lower than Logistic Regression and kNN, likely because this dataset's decision boundary is well approximated by a hyperplane, which Logistic Regression captures directly while Random Forest approximates it with many axis-aligned splits. |
| Overall Winner for your dataset? | **Logistic Regression** is the best model for this dataset. It leads in 5 out of 6 metrics (all except Recall where kNN got a perfect 1.0). The breast cancer features, when standardized, are largely linearly separable, which plays directly to Logistic Regression's strength. It also has the advantage of being interpretable — the model coefficients indicate which features matter most for diagnosis, which is clinically useful. |

### Streamlit App Features

- **CSV Upload:** Upload test data in CSV format to evaluate models
- **Model Selection:** Dropdown to select any of the 5 classification models
- **Metrics Display:** Shows all 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion Matrix:** Visual heatmap of the confusion matrix for the selected model
- **Classification Report:** Detailed per-class precision, recall, and F1
- **Model Comparison:** Side-by-side table and bar chart comparing all models
