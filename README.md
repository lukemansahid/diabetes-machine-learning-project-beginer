## Diabetes Prediction Using Machine Learning

## 1. Project Title
**Diabetes Prediction Using Structured Medical Data and Machine Learning**

---

## 2. Project Overview
This project develops a machine learning system for predicting diabetes risk using a structured/tabular medical dataset. The dataset is based on the **BRFSS 2015 diabetes health indicators dataset**, which contains health, lifestyle, and demographic variables associated with diabetes status.

The goal of the project is to build a predictive model that can distinguish between individuals with **no diabetes** and individuals with diabetes-related conditions using routinely collected health indicators.

---

## 3. Medical / Clinical Problem
Can we predict whether a person is at risk of diabetes using structured health indicators such as BMI, blood pressure, cholesterol status, physical activity, age, and general health?

Diabetes is one of the most important chronic diseases worldwide. It is associated with severe long-term complications including: cardiovascular disease, kidney failure, nerve damage, vision problems, and reduced quality of life.

Early identification of individuals at risk can support preventive care, timely intervention, and better public health planning.
Because of this, diabetes prediction is a highly relevant application of machine learning in healthcare.

---

## 4. Dataset Description
This project uses a **structured medical dataset** derived from the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** diabetes health indicators data at kaggle.
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

The dataset contains tabular features that represent demographic, behavioral, and health-related information of 253,680 survey responses.
In the original BRFSS diabetes dataset, the target variable had three categorical classes where `0` representing No diabetes,`1` representing Prediabetes, and
`2` representing Diabetes.

For this project, classes `1` and `2` were combined into a single positive class to create a binary classification problem that helps identify people with any diabetes-related condition.

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

## 6. Data Preparation

---
### 6.1 Target Variable Transformation
Since the primary objective of the research is detecting the presence of diabetes, the problem was reformulated as a binary classification task. Therefore, the classes Prediabetes(1) and Diabetes(2) were merged into a single class representing individuals with diabetes-related conditions.
After the transformation, the class distribution became 39,977 diabetes and 213,703 no diabetes.
The transformated dataset exhibited a significant class imbalance. A random subset of 39,977 samples was selected from the majority class using random permutation sampling strategy.
The final dataset was then randomly shuffled to prevent accidental learning patterns during model training.

### 6.2 Feature Selection
All variables in the dataset are relevant health indicators and may contribute useful information to diabetes prediction. Since the dataset is already well curated, using all features provides a strong baseline before applying more advanced feature selection methods.

---

### 6.3 Train/Test Split
The dataset was divided into:

- **80% training set**
- **20% test set**

A **stratified split** was used to ensures class distribution remains consistent across training and test sets. This is important for fair evaluation, especially in medical prediction tasks.

---

### 6.4 Feature Scaling
Numeric features were standardized using **StandardScaler**.
 The is especially important for **Logistic Regression** due to the model sensitive to the scale of input variables.
---

## 7. Model Development

### 7.1 Logistic Regression
Logistic Regression is a strong baseline model for medical classification problems. It estimates the probability that a patient belongs to the positive diabetes class.
 and widely accepted in healthcare research.

This model is useful because it provides interpretable coefficients, performs well on many tabular healthcare datasets, and allows identification of important predictors

---

## 8. Evaluation Metrics
The following classification metrics were used:
- **Accuracy**: shows overall correctness
- **Precision**: tells us how many predicted diabetes cases were truly positive
- **Recall**: tells us how many actual diabetes cases were correctly detected
- **F1-score**: balances precision and recall
- **ROC-AUC**: measures how well the model separates the two classes across thresholds
- **Confusion Matrix**: gives a detailed picture of prediction errors

This is especially important in healthcare, where missing a true positive case can have serious consequences.

---

## 9. Results
The following results were obtained on the held-out test set:
- **Accuracy:** 0.7444
- **Precision:** 0.7370
- **Recall:** 0.7601
- **F1-score:** 0.7484
- **ROC-AUC:** 0.8195

An **accuracy** of **74.44%** means that the model correctly classified about three out of every four individuals in the test set.

This is a solid result for a real-world medical dataset containing behavioral and health survey information, especially because the prediction is based on indirect health indicators rather than laboratory measurements alone.

However, accuracy should not be the only focus in a medical setting.

A **precision** of **73.70%** means that when the model predicts that a person has diabetes-related risk, about 74 out of 100 of those predictions are correct.
This indicates that the model is reasonably reliable when flagging positive cases.

A **recall** of **76.01%** means that the model correctly identifies about 76 out of every 100 actual diabetes or prediabetes cases.
A recall above 0.75 suggests the model is reasonably effective at detecting diabetes-related cases.

The **F1-score** of **74.84%** shows a good balance between precision and recall.
This is important because a good medical model should not focus only on reducing false positives or only on catching positives. Instead, it should maintain a balanced trade-off.
The F1-score indicates that the model has a stable and balanced classification performance.


The **ROC-AUC** of **0.8195** is one of the strongest indicators of model quality in this project.
An ROC-AUC above **0.80** generally suggests good class separation ability. This means the model can effectively distinguish between individuals with no diabetes and those with diabetes-related conditions.
 
The **confusion matrix** helps explain the model's mistakes in more detail.
It separates predictions into four groups:

- **True Positives:** correctly predicted diabetes cases
- **True Negatives:** correctly predicted non-diabetes cases
- **False Positives:** non-diabetes cases predicted as diabetes
- **False Negatives:** diabetes cases predicted as non-diabetes

In this project, **false negatives** are especially important because they represent individuals who actually have diabetes-related risk but were missed by the model.

---
## 10. Limitations
Despite the good performance, the project has some limitations:

### 1. Balanced Dataset
The dataset used here is already balanced. While this helps training and evaluation, real-world medical datasets are often imbalanced.

### 2. Survey-Based Features
The BRFSS dataset is based largely on self-reported health and lifestyle information, which may introduce reporting bias.

### 3. Limited Clinical Detail
The model does not include laboratory values such as fasting glucose which are directly relevant to diabetes diagnosis.

### 4. No Hyperparameter Tuning
The models were trained using standard settings. More tuning might improve performance.

---
## 11. Possible Improvements
Future work could improve the project by:

- performing hyperparameter tuning
- using cross-validation
- testing XGBoost or LightGBM
- applying feature selection methods
- validating on an external medical dataset
---

## 12. Project Structure
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


```

## How to Run the Project

### Step 1: Installation

```bash
git clone https://github.com/your-username/project.git
cd project
pip install -r requirements.txt
```
### Step 2: Run the Training Code
```bash
python src/diabetes.py
```
### Step 3: Check the Results
After running the code, the outputs will be saved in the results/ folder.

Files generated include:

- classification-report.json
- metrics.json
- confusion_matrix.png