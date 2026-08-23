🚨 Risk Alert Classifier — Machine Learning Project



A machine learning classification project for identifying customer credit-risk status, with a strong focus on reducing False Negatives.







🎬 Project Workflow



The animation represents the same major stages implemented in the notebook: data preparation, class balancing, model comparison, hyperparameter tuning and final evaluation.

📌 About the Project

This project builds a Risk Alert Classifier using supervised machine learning.

The notebook uses the target column:

risk_status

The objective is to classify customers into risk categories and compare multiple classification approaches.

A key business objective is to minimize False Negatives, because a False Negative means a genuinely high-risk customer is classified as low-risk.

🧠 Project Pipeline



Main workflow

Load the Risk Alert Classifier dataset

Separate features and target

Convert categorical features using one-hot encoding

Split data into training and testing sets

Handle missing values using KNNImputer

Standardize features using StandardScaler

Train a Logistic Regression baseline

Test class-balancing techniques

Compare Decision Tree and Random Forest

Tune Random Forest using Grid Search and Randomized Search

Compare ROC/AUC and False Negatives

Select the final model based on the project objective

🗂️ Dataset & Target

The notebook loads:

df = pd.read_csv("Risk_Alert_Classifier_Dataset_4600.csv.csv")

Target:

y = df["risk_status"]

Features:

X = df.drop("risk_status", axis=1)

Categorical variables are converted with:

X = pd.get_dummies(X, drop_first=True)

🧹 Data Preprocessing



The notebook performs:

1. Train/Test Split

train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

2. Missing Value Imputation

KNNImputer(n_neighbors=5)

3. Feature Scaling

StandardScaler()

The preprocessing is fitted on the training data and then applied to the test data.

⚖️ Handling Class Imbalance



The notebook compares the baseline model with:

Random Under Sampling

Random Over Sampling

SMOTE

ADASYN

Example:

methods = {
    "Under": RandomUnderSampler(random_state=42),
    "Over": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42)
}

The comparison focuses on:

Recall

F1 Score

AUC-ROC

🤖 Machine Learning Models



Logistic Regression

Used as the baseline classification model.

LogisticRegression(max_iter=1000)

Decision Tree

DecisionTreeClassifier(random_state=42)

Random Forest

RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

Tuned Random Forest

Random Forest is further optimized using:

RandomizedSearchCV

GridSearchCV

5-fold cross-validation

F1 scoring

🔍 Hyperparameter Tuning

The notebook searches over:

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

Two tuning strategies are used:

Randomized Search

RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    n_iter=5,
    cv=5,
    scoring="f1",
    random_state=42
)

Grid Search

GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=5,
    scoring="f1"
)

📊 Evaluation Metrics



The project evaluates classification performance using:

Metric

Purpose

Accuracy

Overall correct predictions

Recall

Ability to identify positive/high-risk cases

F1 Score

Balance between precision and recall

AUC-ROC

Model discrimination ability

False Positive

Low-risk predicted as high-risk

False Negative

High-risk predicted as low-risk

Why False Negatives are important

In this project, the final model is selected using the minimum number of False Negatives.

A False Negative can represent a high-risk customer incorrectly classified as low-risk, which may create financial and credit-risk consequences.

🎯 Confusion Matrix



The notebook extracts:

TN, FP, FN, TP

and explicitly evaluates:

Type I Error → False Positive

Type II Error → False Negative

📈 ROC & AUC



The notebook generates ROC curves for:

Logistic Regression

Decision Tree

Random Forest

Tuned Random Forest

AUC-ROC is calculated using:

roc_auc_score(y_test, prob)

and the ROC curve is generated with:

roc_curve(y_test, prob)

🏆 Final Model Selection

The final comparison includes:

Model
Accuracy
Recall
F1 Score
AUC-ROC
False Positive
False Negative

The notebook selects the final model using:

best_row = final_df.loc[
    final_df["False Negative"].idxmin()
]

Business recommendation

The project prioritizes the model with the lowest False Negative count, because missing a genuinely high-risk customer is considered more costly than incorrectly flagging a low-risk customer.

Note: Exact accuracy, recall, F1, AUC and False Negative values should be taken from the actual notebook execution output rather than hard-coded into this README.

🛠️ Technologies Used

Python

Pandas

Scikit-learn

imbalanced-learn

NumPy

Matplotlib

Jupyter Notebook

📁 Project Structure

Risk-Alert-Classifier/
│
├── project3.ipynb
├── Risk_Alert_Classifier_Dataset_4600.csv.csv
├── README.md
│
└── assets/
    ├── 01_project_overview.png
    ├── 02_data_preprocessing.png
    ├── 03_sampling_methods.png
    ├── 04_models_used.png
    ├── 05_evaluation_metrics.png
    ├── 06_roc_auc.png
    ├── 07_confusion_matrix.png
    └── risk_alert_ml_workflow.gif

▶️ How to Run

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Install dependencies

pip install pandas scikit-learn imbalanced-learn matplotlib

3. Open the notebook

jupyter notebook project3.ipynb

4. Keep the dataset in the same project folder

Make sure the CSV filename matches the filename used in the notebook:

Risk_Alert_Classifier_Dataset_4600.csv.csv

5. Run all notebook cells

Execute the notebook from top to bottom to generate the actual model results.

📌 Key Learning Outcomes

Through this project, the following machine learning concepts are demonstrated:

Data preprocessing

Categorical encoding

KNN-based missing-value imputation

Feature scaling

Train/test splitting

Class imbalance handling

Random Under Sampling

Random Over Sampling

SMOTE

ADASYN

Logistic Regression

Decision Tree

Random Forest

Hyperparameter tuning

Grid Search

Randomized Search

ROC curve

AUC-ROC

Confusion matrix

Recall and F1 Score

False Positive / False Negative analysis

Business-oriented model selection

🚀 Future Improvements

Add feature importance visualization

Compare additional classifiers such as XGBoost

Add probability-based risk categories

Build a Streamlit web application

Add model explainability using SHAP

Save the trained model with Joblib

Create an API for real-time risk prediction

Add automated model monitoring

👨‍💻 Project Focus

Risk Alert Classification using Machine Learning

The main principle of this project is:

A good classification model is not selected only by accuracy; it should match the actual business cost of prediction errors.

For this project, minimizing False Negatives is the primary decision criterion.

⭐ If you find this project useful, consider giving the repository a star!
