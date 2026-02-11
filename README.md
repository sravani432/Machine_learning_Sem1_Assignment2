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
- **Dataset Source:** [(https://www.kaggle.com/datasets/thedevastator/uncovering-breast-cancer-diagnosis-with-wisconsi)]
- **Type:** [Binary  classification]
- **Number of Instances:** [570]
- **Number of Features:** [32]
- **Target Variable:** [diagnosis]
- **Preprocessing Steps:** Missing value handling, encoding categorical variables, feature scaling.

---
## Project Structure

project-folder/
│── app.py
│── models/
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│── saved_models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│── data/
│   └── breast-cancer-wisconsin-data.csv
│── requirements.txt
│── README.md

----

## Comparison of Models and Metrics

| Model | Accuracy | AUC | Precision | Recall | F1 |
|-------|-------|-------|-------|-------|-------|
| logistic_regression | 0.9561 | 0.9948 | 0.9750 | 0.9070 | 0.9398 |
| decision_tree | 0.9386 | 0.9369 | 0.9091 | 0.9302 | 0.9195 |
| knn | 0.7544 | 0.8102 | 0.7419 | 0.5349 | 0.6216 |
| naive_bayes | 0.6140 | 0.8919 | 0.0000 | 0.0000 | 0.0000 |
| random_forest | 0.9649 | 0.9972 | 0.9756 | 0.9302 | 0.9524 |
| xgboost | 0.9561 | 0.9951 | 0.9524 | 0.9302 | 0.9412 |


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
# clone repo
git clone https://github.com/sravani432/Machine_learning_Sem1_Assignment2
cd project-folder

# Install dependencies
pip install -r requirements.txt

# Run the streamlit app 
streamlit run app.py

After running, trained models are automatically saved in the saved_models/ folder:

saved_models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── xgboost.pkl

These .pkl files can be reloaded later for predictions without retraining. 