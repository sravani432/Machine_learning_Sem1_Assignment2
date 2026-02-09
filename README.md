# Machine_learning_Sem1_Assignment2
Machine Learning assignment 2 for 1st semister

The data set is loaded from kaggle : https://www.kaggle.com/datasets/thedevastator/uncovering-breast-cancer-diagnosis-with-wisconsi
This dataset is having 32 features 


# Machine Learning Assignment 2

## Problem Statement
The goal of this assignment is to implement and compare multiple machine learning classification models on a chosen dataset. The workflow includes:
- Data preprocessing
- Training six classification models
- Evaluating models using multiple metrics
- Building and deploying an interactive Streamlit application
- Documenting results and observations

This exercise demonstrates an end-to-end ML pipeline, from modeling to deployment.

---

## Dataset Description
- **Dataset Source:** [[Add Kaggle/UCI link here](https://www.kaggle.com/datasets/thedevastator/uncovering-breast-cancer-diagnosis-with-wisconsi)]
- **Type:** [Binary  classification]
- **Number of Instances:** [570]
- **Number of Features:** [32]
- **Target Variable:** [diagnosis]
- **Preprocessing Steps:** Missing value handling, encoding categorical variables, feature scaling.

---

## Comparison of Models and Metrics

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression  | [val]    | [val] | [val]     | [val]  | [val] | [val] |
| Decision Tree        | [val]    | [val] | [val]     | [val]  | [val] | [val] |
| kNN                  | [val]    | [val] | [val]     | [val]  | [val] | [val] |
| Naive Bayes          | [val]    | [val] | [val]     | [val]  | [val] | [val] |
| Random Forest        | [val]    | [val] | [val]     | [val]  | [val] | [val] |
| XGBoost              | [val]    | [val] | [val]     | [val]  | [val] | [val] |

*(Replace `[val]` with actual computed metrics from your results.)*

---

## Observations on Model Performance

| ML Model Name        | Observation about model performance |
|----------------------|--------------------------------------|
| Logistic Regression  | Performs well on linearly separable data; moderate accuracy and balanced precision/recall. |
| Decision Tree        | Captures non-linear relationships; prone to overfitting if depth not controlled. |
| kNN                  | Sensitive to feature scaling; performance depends on choice of k. |
| Naive Bayes          | Fast and simple; works best when features are independent; may underperform with correlated features. |
| Random Forest        | Strong ensemble model; reduces overfitting; generally high accuracy and recall. |
| XGBoost              | Most powerful ensemble; often achieves best overall performance; requires careful tuning to avoid overfitting. |

---

## How to Run the Streamlit App
1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd project-folder

