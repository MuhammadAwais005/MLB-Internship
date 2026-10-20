# Breast Cancer Prediction System
### Logistic Regression | Model Evaluation | Hyperparameter Tuning

## 📌 Project Overview

This project builds a Machine Learning classification system using the **Breast Cancer Wisconsin Diagnostic Dataset**, which is available directly through Scikit-learn.

The objective is to classify tumors as **Malignant (cancerous)** or **Benign (non-cancerous)** using features extracted from breast mass measurements.

The project demonstrates a complete Machine Learning workflow, including data preprocessing, model training, evaluation, hyperparameter tuning with `GridSearchCV`, and comparison of baseline and tuned models.

> **Disclaimer:** This project is for educational purposes only and is not intended for real-world medical diagnosis.

---

## 🎯 Project Objectives

- Load and explore the built-in Breast Cancer Wisconsin dataset.
- Prepare the features and target labels.
- Split the data into training and testing sets.
- Train a baseline Logistic Regression model.
- Evaluate model performance using multiple classification metrics.
- Optimize hyperparameters using `GridSearchCV`.
- Compare baseline and tuned model performance.
- Visualize confusion matrices and evaluation metrics.

---

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data manipulation and exploration
- **Scikit-learn** — Model training, preprocessing, evaluation, and hyperparameter tuning
- **Matplotlib** — Data visualization
- **Seaborn** — Confusion matrix heatmaps

---

## 📊 Dataset Information

The dataset is provided by Scikit-learn, so no separate download is required.

| Property | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin Diagnostic |
| Source | Scikit-learn |
| Total samples | 569 |
| Input features | 30 |
| Target classes | 2 |
| Classification type | Binary classification |
| Missing values | None in the original dataset features |

### Target Classes

The target variable represents the tumor diagnosis:

- **0 — Malignant:** Cancerous tumor
- **1 — Benign:** Non-cancerous tumor

The model uses 30 numerical features describing characteristics of cell nuclei, including radius, texture, perimeter, area, smoothness, and other measurements.

---

## 🔄 Project Workflow

1. **Data Loading:** Load the dataset using `load_breast_cancer()` from Scikit-learn.
2. **Data Exploration:** Convert the features into a Pandas DataFrame and inspect the data using `head()`, `info()`, `describe()`, and target class counts.
3. **Data Splitting:** Divide the dataset into 80% training data and 20% testing data.
4. **Preprocessing:** Apply `StandardScaler` to standardize numerical features.
5. **Baseline Model:** Train Logistic Regression using its default regularization settings, with `max_iter=5000`.
6. **Hyperparameter Tuning:** Use `GridSearchCV` with five-fold cross-validation to find a promising parameter combination.
7. **Model Evaluation:** Calculate accuracy, precision, recall, F1-score, and the confusion matrix.
8. **Performance Comparison:** Compare baseline and tuned model results using tables and visualizations.

A Scikit-learn `Pipeline` combines scaling and Logistic Regression so that preprocessing is fitted correctly within each cross-validation fold.

---

## 📈 Model Evaluation

Model evaluation helps determine how well a trained model performs on unseen data.

This project uses the following metrics:

### 1. Accuracy

Measures the proportion of all predictions that are correct.

### 2. Precision

Measures how many predicted positive cases are actually positive.

In this project, malignant tumors are treated as the positive class when reporting precision.

### 3. Recall

Measures how many actual malignant cases the model correctly identifies.

Recall is particularly important in this educational medical-classification example because a false negative means a malignant case was classified as benign.

### 4. F1-Score

Combines precision and recall into a single metric using their harmonic mean.

### 5. Confusion Matrix

Shows correct and incorrect predictions for each class.

It helps identify:
- True positives: malignant cases correctly identified.
- False positives: benign cases incorrectly classified as malignant.
- False negatives: malignant cases incorrectly classified as benign.
- True negatives: benign cases correctly identified.

**Important:** Precision, recall, and F1-score in this project use malignant (class `0`) as the positive class. Accuracy considers both classes.

---

## ⚙️ Hyperparameter Tuning with GridSearchCV

Hyperparameter tuning is the process of testing different model settings to find a configuration that performs well according to a selected evaluation metric.

This project uses `GridSearchCV` to test the following hyperparameters:

| Hyperparameter | Values tested | Purpose |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 | Controls the inverse strength of regularization |
| `solver` | `liblinear`, `lbfgs` | Specifies the optimization algorithm |

The search uses:

- **Cross-validation:** 5 folds
- **Scoring metric:** Recall
- **Search combinations:** 10
- **Fold-level model fits:** 50, plus a final refit of the best configuration

The search optimizes mean cross-validation recall for the estimator's default positive class, which is class `1` (benign). The separate evaluation function reports recall for malignant (class `0`). These are different objectives, so the distinction matters when interpreting the results.

### Why Hyperparameter Tuning Matters

- It helps identify settings that improve performance on validation data.
- It provides a systematic alternative to manually guessing settings.
- It can improve generalization when appropriate settings are found.
- It helps compare configurations using a consistent evaluation procedure.

Tuning does not guarantee that every metric will improve. The final test set is reserved for an independent evaluation after model selection.

---

##  Baseline vs. Tuned Model

The baseline model uses the default Logistic Regression regularization settings. The tuned model uses the hyperparameters selected by `GridSearchCV`.

### Best Parameters Found

After running the project, replace the placeholders below with the values printed by:

`grid_search.best_params_`

| Parameter | Best value |
|---|---|
| `model__C` | **[Insert actual value]** |
| `model__solver` | **[Insert actual value]** |
| Best mean CV score | **[Insert actual recall score]** |

### Performance Comparison

Replace the placeholders with the test-set results printed by your program.

| Evaluation Metric | Baseline Model | Tuned Model |
|---|---:|---:|
| Accuracy | [Insert result] | [Insert result] |
| Precision — Malignant | [Insert result] | [Insert result] |
| Recall — Malignant | [Insert result] | [Insert result] |
| F1-Score — Malignant | [Insert result] | [Insert result] |

These values should come from the actual project run rather than estimated or assumed scores.

### Confusion Matrix Visualization

The project produces two confusion matrix heatmaps:

- Baseline Logistic Regression
- Tuned Logistic Regression

These visualizations make it easier to compare correct classifications and the types of errors made by each model.

A second chart compares the baseline and tuned models across accuracy, precision, recall, and F1-score.

---

## 🔍 Key Observations

The following observations should be completed after examining the actual results.

1. **Baseline performance:** Record how well Logistic Regression performs before tuning.
2. **Best hyperparameters:** Identify the `C` and `solver` selected by GridSearchCV.
3. **Tuned performance:** Determine whether the tuned model improves the selected test metrics.
4. **Recall and false negatives:** Check whether the model misses fewer malignant cases.
5. **Precision and false positives:** Determine whether improved recall comes with more benign cases incorrectly classified as malignant.
6. **Overall comparison:** Consider all metrics and the confusion matrices rather than relying on accuracy alone.

### Final Observation

**[Write your conclusion after running the project.]**

For example, explain whether tuning improved malignant-class recall, whether precision changed, and whether the confusion matrix shows fewer false negatives.

---

## 📚 What I Learned

Through this project, I practiced and strengthened my understanding of the following Machine Learning concepts:

- Loading built-in datasets from Scikit-learn.
- Exploring datasets with Pandas.
- Understanding features and target variables.
- Splitting data into training and testing sets.
- Standardizing numerical features using `StandardScaler`.
- Building a Machine Learning pipeline.
- Training Logistic Regression for binary classification.
- Evaluating models using accuracy, precision, recall, F1-score, and confusion matrices.
- Understanding the importance of class-specific evaluation metrics.
- Using five-fold cross-validation.
- Performing hyperparameter tuning with `GridSearchCV`.
- Comparing baseline and tuned model performance.
- Visualizing model evaluation results.
- Avoiding data leakage during preprocessing and model selection.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project folder

```bash
cd <your-project-folder>
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### 4. Run the Python file

```bash
python breast_cancer_prediction.py
```

The program prints the baseline metrics, best hyperparameters, tuned model metrics, and comparison table. It also displays the confusion matrix heatmaps and metric comparison chart.

---

## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest and Support Vector Machine.
- Explore `RandomizedSearchCV` for larger hyperparameter search spaces.
- Evaluate ROC-AUC and precision-recall curves.
- Investigate classification thresholds and their effect on false negatives.
- Add a reproducible experiment report with actual results.
- Build a simple web interface for educational demonstrations.

---

## 📝 Conclusion

This project demonstrates how to build and evaluate a binary classification model using Scikit-learn. It combines data exploration, preprocessing, Logistic Regression, cross-validation, hyperparameter tuning, and model evaluation into a single workflow.

The key takeaway is that **training a model is only one part of Machine Learning. Evaluating it correctly, selecting suitable metrics, and testing its ability to generalize to unseen data are equally important.**

*Developed as a Machine Learning practice project focused on model evaluation and hyperparameter tuning.*
