import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)



# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes_012_health_indicators_BRFSS2015.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
dataset = pd.read_csv(DATA_PATH)

print("Dataset shape:", dataset.shape)
print("Columns:", dataset.columns.tolist())

target_column = "Diabetes_012"

print("Target_Colum:", dataset.groupby([target_column]).size())
# ------------------------------------------------
# Merge the (1)-prediabetes and the (2)-diabetes into a single class
# -----------------------------------------------
dataset[target_column] = dataset[target_column].replace({0:0, 1:1, 2:1})
dataset.groupby([target_column]).size()

print("Target_merge:", dataset.groupby([target_column]).size())
# ---------------------------------------------------------------------
# Now the taget variable is highly imbalance, needs to be balanced to avoid model bias
# Separate the no diabetes (0) from the diabetes (1) and perform random selection in the majority class
# -----------------------------------------------------------------
df0 = dataset[dataset[target_column] == 0]
df1 = dataset[dataset[target_column] == 1]

df0_sample = df0.sample(n=len(df1), random_state=42)
balanced_dataset = pd.concat([df0_sample, df1])

# ---------------------------------------------------------------------
# Shuffle the Dataset using random permutation so the classes are mixed
# --------------------------------------------------------------------
balanced_dataset = balanced_dataset.sample(frac=1, random_state=42)

balanced_dataset.groupby([target_column]).size()

balanced_dataset.to_csv('../data/diabetes_012_health_indicators_BRFSS2015_balanced.csv', sep=',', index=False)

# ----------------------------
# Separate Features and target
# ----------------------------
X = balanced_dataset.drop(columns=[target_column])
y = balanced_dataset[target_column]

print("Feature(X): ", X.shape)
print("Target(y): ", y.shape)

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# -----------------------------
# Scale the features
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train the model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Make predictions
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# Evaluation
# -------------------------------
accuracy = float(accuracy_score(y_test, y_pred))
precision = float(precision_score(y_test, y_pred))
recall = float(recall_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred))
roc_auc = float(roc_auc_score(y_test, y_prob))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("ROC-AUC:", roc_auc)

# -------------------------------
# Save results
# -------------------------------
os.makedirs("../results", exist_ok=True)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc
}

with open("../results/metrics.json","w") as f:
    json.dump(metrics,f,indent=4)

with open("../results/classification-report.json","w") as f:
    json.dump(classification_report(y_test, y_pred),f,indent=4)

# -----------------
# Confusion matrix
# ----------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.show()

print("Project completed.")