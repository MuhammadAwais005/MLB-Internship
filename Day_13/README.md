# Day 13: Classification Models

This folder contains two classification exercises: breast-cancer diagnosis with logistic regression, and Iris flower classification with logistic regression and a decision tree. The Iris scripts save their test results for `Comparison.py`, which compares the models on the same held-out set and with 5-fold cross-validation.

## Project Files

- `data.csv`: Breast Cancer Wisconsin diagnostic dataset. It contains diagnosis labels and 30 numeric measurements, plus an ID column and an empty `Unnamed: 32` column.
- `BreastCancer_Logistic.py`: Cleans the breast-cancer data, trains logistic regression, reports test metrics, predicts a sample, and displays a confusion matrix.
- `iris_Logistic.py`: Trains and evaluates logistic regression on scikit-learn's built-in Iris dataset, predicts a sample flower, and saves results to `iris_logistic.pkl`.
- `iris_DT.py`: Trains and evaluates a decision tree on the Iris dataset, predicts a sample flower, and saves results to `iris_decision_tree.pkl`.
- `Comparison.py`: Loads both saved Iris result files, compares test metrics and confusion matrices, prints classification reports, and runs 5-fold cross-validation.

## Setup and Run

From this folder, install the required packages:

```powershell
py -m pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

Run the breast-cancer exercise:

```powershell
py BreastCancer_Logistic.py
```

Run the Iris exercises and comparison in this order, from the same folder. The first two scripts create the `.pkl` files required by the comparison script.

```powershell
py iris_Logistic.py
py iris_DT.py
py Comparison.py
```

The scripts display confusion-matrix plots. Close each plot window to let execution continue.

## What I Learned About Data Preprocessing

- Inspect the dataset and check for missing values before training. The breast-cancer script drops `id`, which is an identifier rather than a useful measurement, and `Unnamed: 32`, which is empty.
- Convert categorical target labels into numeric values before fitting scikit-learn estimators. In the cancer dataset, `LabelEncoder` maps `B` (benign) to `0` and `M` (malignant) to `1`. The Iris scripts encode species labels similarly.
- Scale features when their numeric ranges differ substantially. The Iris scripts fit `StandardScaler` on the training features, then use that fitted scaler to transform the test and sample data. Fitting preprocessing on training data only helps avoid data leakage.
- Keep the same feature order and preprocessing for new observations as for training data. The cancer script passes 30 measurements; each Iris script scales its four measurements before prediction.

## Why Train-Test Splitting Matters

The models are trained on one part of the data and evaluated on held-out observations. This gives a more useful estimate of how the models may perform on data they did not see during training than evaluating on the training set. The scripts use an 80/20 split and a fixed `random_state` so the result is repeatable. The Iris split is stratified, preserving the class proportions in each subset; the breast-cancer split is not stratified.

The comparison script also uses 5-fold cross-validation on Iris. It evaluates fresh model instances across several folds, giving a broader view than a single train-test split, though the result still depends on the dataset and split strategy.

## Evaluation Metrics Used

- **Accuracy**: fraction of test predictions that are correct.
- **Precision**: fraction of predictions for a class that are correct.
- **Recall**: fraction of actual examples of a class that are found.
- **F1-score**: harmonic mean of precision and recall.
- **Confusion matrix**: counts correct and incorrect predictions by actual and predicted class.
- **5-fold cross-validation accuracy**: accuracy on each fold, plus the mean and standard deviation across folds.

The breast-cancer script prints accuracy and a classification report, and plots a confusion matrix. The Iris comparison reports accuracy, weighted precision, weighted recall, weighted F1-score, confusion matrices, per-class reports, and cross-validation accuracy.

## Model Performance and Observations

Results below were measured using the split settings in the scripts (`random_state=42`; Iris also uses `stratify=y`). These are dataset-specific results, not a guarantee of future performance.

### Breast Cancer Logistic Regression

- Test accuracy: **95.61%** (109 of 114 test records).
- Benign: precision **94.59%**, recall **98.59%**, F1-score **96.55%**.
- Malignant: precision **97.50%**, recall **90.70%**, F1-score **93.98%**.
- The model performed well overall, but malignant recall is lower than benign recall. In this test split, it missed 4 of 43 malignant cases, so accuracy alone would hide an important error type.

### Iris Model Comparison

| Model               | Test accuracy | Weighted precision | Weighted recall | Weighted F1 | Mean 5-fold accuracy | CV standard deviation |
| ------------------- | ------------: | -----------------: | --------------: | ----------: | -------------------: | --------------------: |
| Logistic Regression |        93.33% |             93.33% |          93.33% |      93.33% |               96.00% |                 3.89% |
| Decision Tree       |        93.33% |             93.33% |          93.33% |      93.33% |               95.33% |                 3.40% |

Both models correctly classified 28 of 30 Iris test examples. The test split gives them the same scores, while logistic regression has a slightly higher mean cross-validation accuracy and the decision tree has a slightly lower fold-to-fold standard deviation. This small difference is not enough to establish that one model will always perform better.

## Notes and Limitations

- `BreastCancer_Logistic.py` currently reads the CSV using an absolute path for the original Windows machine. Update that path if the project is moved to another location or computer.
- The cancer model does not scale its features, even though the measurements have different ranges. Testing a training-fitted scaler would be a useful follow-up, with performance compared on the same held-out data.
- The Iris scripts save the fitted classifier and test predictions, but not the fitted scaler or label encoder. Their `.pkl` files support the current comparison workflow; the saved classifier alone is not ready to predict directly from raw flower measurements.
- The `.pkl` files are generated when the Iris scripts run and must be in the working directory when `Comparison.py` runs.
