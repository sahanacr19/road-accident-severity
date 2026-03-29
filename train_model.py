import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier


print("Loading datasets...")

accidents = pd.read_csv("dataset/Accidents.csv")
vehicles = pd.read_csv("dataset/Vehicles.csv")
casualties = pd.read_csv("dataset/Casualties.csv")

print("Datasets loaded")


# merge datasets
data = accidents.merge(vehicles, on="Accident_Index")
data = data.merge(casualties, on="Accident_Index")

print("Datasets merged")


# select useful columns
data = data[
[
"Accident_Severity",
"Weather_Conditions",
"Road_Surface_Conditions",
"Light_Conditions",
"Speed_limit",
"Number_of_Vehicles",
"Age_of_Driver",
"Sex_of_Driver"
]
]

data = data.dropna()

print("Data cleaned")


# encode categorical columns
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

data["Weather_Conditions"] = le1.fit_transform(data["Weather_Conditions"])
data["Road_Surface_Conditions"] = le2.fit_transform(data["Road_Surface_Conditions"])
data["Light_Conditions"] = le3.fit_transform(data["Light_Conditions"])
data["Sex_of_Driver"] = le4.fit_transform(data["Sex_of_Driver"])


X = data.drop("Accident_Severity", axis=1)

# convert labels for ML models
y = data["Accident_Severity"] - 2


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining multiple algorithms...\n")

results = {}


# ---------------- DECISION TREE ----------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

results["Decision Tree"] = accuracy_score(y_test, dt_pred)


# ---------------- LOGISTIC REGRESSION ----------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

results["Logistic Regression"] = accuracy_score(y_test, lr_pred)


# ---------------- SVM ----------------
svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

results["SVM"] = accuracy_score(y_test, svm_pred)


# ---------------- RANDOM FOREST ----------------
print("Training Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

results["Random Forest"] = rf_accuracy


# Save Random Forest for Django
joblib.dump(rf_model, "model/accident_model.pkl")

print("Random Forest model saved")


# ---------------- XGBOOST ----------------
print("Training XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

results["XGBoost"] = accuracy_score(y_test, xgb_pred)


# ---------------- RESULTS ----------------
print("\nModel Accuracy Comparison\n")

print("{:<25} {}".format("Algorithm", "Accuracy"))

for model, acc in results.items():
    print("{:<25} {:.2f}%".format(model, acc*100))


# confusion matrix for Random Forest
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, rf_pred))


print("\nTraining completed")