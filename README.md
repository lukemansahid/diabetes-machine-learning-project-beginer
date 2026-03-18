# Diabetes Prediction Using Machine Learning

## 1. Project Title
**Diabetes Prediction Using Structured Medical Data and Machine Learning**

---

## 2. Project Overview
This project develops a machine learning system for predicting diabetes risk using a structured/tabular medical dataset. The dataset is based on the **BRFSS 2015 diabetes health indicators dataset**, which contains health, lifestyle, and demographic variables associated with diabetes status.

The goal of the project is to build a predictive model that can distinguish between individuals with **no diabetes** and individuals with **diabetes-related conditions** using routinely collected health indicators.

This project satisfies the requirements of a typical applied machine learning healthcare task, including:

- dataset selection
- medical problem definition
- data preprocessing
- model training
- performance evaluation
- interpretation of results

---

## 3. Medical / Clinical Problem
### Clinical Question
Can we predict whether a person is at risk of diabetes using structured health indicators such as BMI, blood pressure, cholesterol status, physical activity, age, and general health?

### Why This Problem Matters
Diabetes is one of the most important chronic diseases worldwide. It is associated with severe long-term complications, including:

- cardiovascular disease
- kidney failure
- nerve damage
- vision problems
- reduced quality of life

Early identification of individuals at risk can support:

- preventive care
- early screening
- timely intervention
- better public health planning

Because of this, diabetes prediction is a highly relevant application of machine learning in healthcare.

---

## 4. Dataset Description
This project uses a **structured medical dataset** derived from the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** diabetes health indicators data.

The dataset contains tabular features that represent demographic, behavioral, and health-related information.

### Dataset Type
- Structured / tabular data
- Supervised learning dataset
- Binary classification problem

### Target Variable
The final target variable used in this project is:

- `Diabetes_binary`
  - `0` = No diabetes
  - `1` = Diabetes / Prediabetes

### Original Target Structure
In the original BRFSS diabetes dataset, the target variable had three classes:

- `0` = No diabetes
- `1` = Prediabetes
- `2` = Diabetes

For this project, classes `1` and `2` were combined into a single positive class to create a binary classification problem.

### Why the Target Was Reframed
This transformation was useful because:

- it simplifies the prediction task
- it aligns with real screening use cases
- it helps identify people with any diabetes-related condition
- it creates a clearer clinical decision boundary

---

## 5. Input Features
The dataset includes several structured health indicators, such as:

- High blood pressure (`HighBP`)
- High cholesterol (`HighChol`)
- Cholesterol check (`CholCheck`)
- Body Mass Index (`BMI`)
- Smoking status (`Smoker`)
- History of stroke (`Stroke`)
- Heart disease or attack (`HeartDiseaseorAttack`)
- Physical activity (`PhysActivity`)
- Fruit consumption (`Fruits`)
- Vegetable consumption (`Veggies`)
- Heavy alcohol consumption (`HvyAlcoholConsump`)
- Health care access (`AnyHealthcare`, `NoDocbcCost`)
- General health (`GenHlth`)
- Mental health (`MentHlth`)
- Physical health (`PhysHlth`)
- Difficulty walking (`DiffWalk`)
- Sex (`Sex`)
- Age (`Age`)
- Education (`Education`)
- Income (`Income`)

These variables are clinically meaningful because diabetes risk is often associated with obesity, age, cardiovascular risk, inactivity, and overall health status.

---

## 6. Problem Formulation
This project is framed as a **binary classification problem**:

Given a set of health-related indicators, predict whether an individual belongs to:

- **Class 0:** No diabetes
- **Class 1:** Diabetes / Prediabetes

---

## 7. Data Preparation

### 7.1 Handling Missing Values
The preprocessing pipeline uses **median imputation** to handle missing values.

### Why Median Imputation?
Median imputation was chosen because:

- it is simple and robust
- it works well for numeric tabular data
- it is less sensitive to extreme values than mean imputation
- it ensures the model can train even if some observations are incomplete

Even if the dataset has few or no missing values, including this step makes the pipeline more reliable and reusable.

---

### 7.2 Feature Selection
All available structured input features were included in the model.

### Why Use All Features?
All variables in the dataset are relevant health indicators and may contribute useful information to diabetes prediction. Since the dataset is already well curated, using all features provides a strong baseline before applying more advanced feature selection methods.

---

### 7.3 Train/Test Split
The dataset was divided into:

- **80% training set**
- **20% test set**

A **stratified split** was used.

### Why Stratified Splitting?
Stratified splitting ensures that the class distribution remains consistent across training and test sets. This is important for fair evaluation, especially in medical prediction tasks.

---

### 7.4 Feature Scaling
Numeric features were standardized using **StandardScaler**.

### Why Scaling Was Applied
Scaling was especially important for **Logistic Regression** because:

- the model is sensitive to the scale of input variables
- standardization improves optimization
- it allows coefficients to be compared more meaningfully

For Random Forest, scaling is not strictly necessary, but it was included through the same preprocessing pipeline for consistency.

---

## 8. Model Development

Two machine learning models were used:

### 8.1 Logistic Regression
Logistic Regression is a strong baseline model for medical classification problems because it is:

- simple
- interpretable
- efficient
- widely accepted in healthcare research

It estimates the probability that a patient belongs to the positive diabetes class.

### Why Logistic Regression Was Chosen
This model is useful because:

- it provides interpretable coefficients
- it is easy to explain in a presentation or report
- it performs well on many tabular healthcare datasets
- it allows identification of important predictors

---

### 8.2 Random Forest
Random Forest is an ensemble learning method that builds many decision trees and combines their predictions.

### Why Random Forest Was Chosen
Random Forest was included because:

- it handles nonlinear patterns well
- it is strong for tabular data
- it is robust to noise
- it can capture more complex feature interactions than Logistic Regression

---

## 9. Evaluation Metrics
The following classification metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Why These Metrics Matter
In medical prediction, accuracy alone is not enough.

- **Accuracy** shows overall correctness
- **Precision** tells us how many predicted diabetes cases were truly positive
- **Recall** tells us how many actual diabetes cases were correctly detected
- **F1-score** balances precision and recall
- **ROC-AUC** measures how well the model separates the two classes across thresholds
- **Confusion Matrix** gives a detailed picture of prediction errors

This is especially important in healthcare, where missing a true positive case can have serious consequences.

---

## 10. Results

### Best Model
The best-performing model in this project was:

**Logistic Regression**

### Test Set Performance
The following results were obtained on the held-out test set:

- **Accuracy:** 0.7444
- **Precision:** 0.7370
- **Recall:** 0.7601
- **F1-score:** 0.7484
- **ROC-AUC:** 0.8195

---

## 11. Interpretation of the Results

### 11.1 Accuracy Interpretation
An accuracy of **74.44%** means that the model correctly classified about three out of every four individuals in the test set.

This is a solid result for a real-world medical dataset containing behavioral and health survey information, especially because the prediction is based on indirect health indicators rather than laboratory measurements alone.

However, accuracy should not be the only focus in a medical setting.

---

### 11.2 Precision Interpretation
A precision of **73.70%** means that when the model predicts that a person has diabetes-related risk, about 74 out of 100 of those predictions are correct.

This indicates that the model is reasonably reliable when flagging positive cases.

In practice, this means the model does not generate an excessive number of false alarms, although some false positives still occur.

---

### 11.3 Recall Interpretation
A recall of **76.01%** means that the model correctly identifies about 76 out of every 100 actual diabetes or prediabetes cases.

This is one of the most important metrics in this project.

Why? Because in a healthcare screening context, failing to identify at-risk individuals can be more harmful than producing some false positives.

A recall above 0.75 suggests the model is reasonably effective at detecting diabetes-related cases.

---

### 11.4 F1-Score Interpretation
The F1-score of **74.84%** shows a good balance between precision and recall.

This is important because a good medical model should not focus only on reducing false positives or only on catching positives. Instead, it should maintain a balanced trade-off.

The F1-score indicates that the model has a stable and balanced classification performance.

---

### 11.5 ROC-AUC Interpretation
The ROC-AUC of **0.8195** is one of the strongest indicators of model quality in this project.

An ROC-AUC above **0.80** generally suggests good class separation ability. This means the model can effectively distinguish between individuals with no diabetes and those with diabetes-related conditions.

In simple terms, if we randomly pick one positive case and one negative case, the model has about an **82% chance** of ranking the positive case higher than the negative one.

This shows that the model has strong discriminative ability.

---

## 12. Clinical Meaning of the Results
These results suggest that machine learning can be used to identify diabetes-related risk from non-invasive structured health indicators.

This is useful because the model relies on variables such as:

- BMI
- blood pressure
- cholesterol status
- physical activity
- age
- general health

These are variables that are often available through surveys, routine screening, or primary care records.

Therefore, the model could potentially support:

- diabetes risk screening
- early warning systems
- public health decision-making
- prevention-focused healthcare programs

---

## 13. Confusion Matrix Interpretation
The confusion matrix helps explain the model's mistakes in more detail.

It separates predictions into four groups:

- **True Positives:** correctly predicted diabetes cases
- **True Negatives:** correctly predicted non-diabetes cases
- **False Positives:** non-diabetes cases predicted as diabetes
- **False Negatives:** diabetes cases predicted as non-diabetes

### Why This Matters
In this project, **false negatives** are especially important because they represent individuals who actually have diabetes-related risk but were missed by the model.

A screening model should try to minimize false negatives as much as possible, even if that means accepting some false positives.

Because the model achieved slightly higher recall than precision, it leans a bit more toward detecting more positive cases, which is often preferable in medical screening.

---

## 14. Why Logistic Regression Performed Best
Although Random Forest is powerful, Logistic Regression performed best in this project.

Possible reasons include:

- the dataset may contain relationships that are largely linear or monotonic
- the balanced data supports stable linear decision boundaries
- the features are already informative and well structured
- Logistic Regression generalizes well and avoids unnecessary complexity

Another important advantage is interpretability. In medical research, interpretable models are often preferred because they allow researchers and clinicians to understand how features influence predictions.

---

## 15. Feature Importance / Coefficient Interpretation
When Logistic Regression is used, model coefficients can provide insight into which variables are more strongly associated with diabetes risk.

Features such as the following are often expected to contribute strongly:

- BMI
- High blood pressure
- High cholesterol
- Age
- Difficulty walking
- General health
- Physical health

Positive coefficients indicate that higher values increase the likelihood of the positive diabetes class, while negative coefficients indicate the opposite.

This makes the model not only predictive, but also informative.

---

## 16. Strengths of the Project
This project has several strengths:

- uses a real medical/public health dataset
- addresses a clinically relevant problem
- applies proper preprocessing steps
- compares more than one machine learning model
- uses multiple evaluation metrics
- provides interpretable results

---

## 17. Limitations
Despite the good performance, the project has some limitations:

### 1. Balanced Dataset
The dataset used here is already balanced. While this helps training and evaluation, real-world medical datasets are often imbalanced.

### 2. Survey-Based Features
The BRFSS dataset is based largely on self-reported health and lifestyle information, which may introduce reporting bias.

### 3. Limited Clinical Detail
The model does not include laboratory values such as fasting glucose or HbA1c, which are directly relevant to diabetes diagnosis.

### 4. No Hyperparameter Tuning
The models were trained using standard settings. More tuning might improve performance.

### 5. No External Validation
The model was not tested on an independent external dataset, so generalizability remains uncertain.

---

## 18. Possible Improvements
Future work could improve the project by:

- performing hyperparameter tuning
- using cross-validation
- testing XGBoost or LightGBM
- applying feature selection methods
- using SHAP values for better interpretability
- validating on an external medical dataset
- comparing performance with resampling methods such as SMOTE

---

## 19. Project Structure
```text
diabetes_ml_project/
│
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv
│   └── diabetes_012_health_indicators_BRFSS2015_balanced.csv
│
├── src/
│   └── diabetes.py
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── classification-report.json
│
├── requirements.txt
└── README.md