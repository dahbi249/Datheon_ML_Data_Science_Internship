# Wine Classification — Machine Learning Internship

## 1. Project Overview

This project implements a multiclass classification system using the Wine dataset from Scikit-learn.

The objective is to predict the class of a wine sample based on 13 chemical features using two machine learning algorithms:

- Logistic Regression
- Decision Tree Classifier

This project was completed as part of the Datheon Machine Learning & Data Science Internship.

## 2. Dataset

The Wine dataset contains:

- 178 samples
- 13 numerical input features
- 3 target classes
- No missing values
- No duplicate rows

The dataset was loaded directly from `sklearn.datasets`.

### Class Distribution

| Class | Samples |
|---|---:|
| Class 0 | 59 |
| Class 1 | 71 |
| Class 2 | 48 |

## 3. Technologies Used

- Python 3.11
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 4. Methodology

### Data Exploration
- Inspected dataset dimensions, feature names, and data types.
- Analyzed class distribution.
- Checked missing values and duplicate rows.
- Examined descriptive statistics, feature distributions, and correlations.

### Data Preprocessing
- Separated features and target labels.
- Split the dataset into training (80%) and testing (20%) sets.
- Used stratified splitting to preserve class proportions.
- Applied StandardScaler for Logistic Regression.
- Used the original unscaled features for Decision Tree.

### Model Training
Two classification algorithms were trained:

1. Logistic Regression
2. Decision Tree Classifier

### Evaluation
Models were evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Confusion Matrix
- Five-fold Stratified Cross-Validation

Decision boundaries were also visualized using two selected features: `alcohol` and `flavanoids`.

## 5. Model Performance

### Test Set Results

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Accuracy | 97.22% | 94.44% |
| Weighted Precision | 97.41% | 95.14% |
| Weighted Recall | 97.22% | 94.44% |
| Weighted F1-score | 97.20% | 94.50% |

### Cross-Validation Results

Five-fold stratified cross-validation was performed on the training data.

| Metric | Logistic Regression | Decision Tree |
|---|---:|---:|
| Mean Accuracy | 97.91% | 90.22% |
| Standard Deviation | 0.0277 | 0.0586 |

## 6. Results and Observations

Logistic Regression achieved the best performance in this experiment.

It obtained 97.22% test accuracy and 97.91% mean cross-validation accuracy.

The Decision Tree achieved 94.44% test accuracy and 90.22% mean cross-validation accuracy.

The cross-validation results also showed that Logistic Regression had lower variation across folds.

The decision-boundary visualizations demonstrated the difference between the two algorithms:

- Logistic Regression produced relatively smooth, linear decision boundaries.
- Decision Tree produced more irregular, threshold-based decision regions.

The decision-boundary experiment used only two features for visualization, whereas the main classification models used all 13 features.

## 7. Project Structure

```text
Task_1_Classification/
├── notebooks/
│   └── 01_wine_classification.ipynb
├── src/
├── results/
│   ├── confusion_matrix_logistic.png
│   ├── confusion_matrix_tree.png
│   ├── decision_boundaries.png
│   ├── model_comparison.csv
│   └── cross_validation_results.csv
└── README.md
```

## 8. How to Run

1. Clone the repository.
2. Activate the project's Python virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open `01_wine_classification.ipynb` in VS Code or Jupyter.
5. Select the project's Python environment.
6. Run all notebook cells.

## 9. Conclusion

This project provided practical experience with the complete multiclass classification workflow, including exploratory data analysis, preprocessing, model training, evaluation, cross-validation, and visualization.

Logistic Regression performed better than the default Decision Tree configuration on this dataset and train/test split.

Further experiments could investigate hyperparameter tuning and additional classification algorithms.