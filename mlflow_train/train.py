# Importing Necessary Libraries
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data from CSV
data = pd.read_csv('../data/breast_cancer_modified.csv')

# Assuming the last column is the target and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Train test split with 20% in test and 80% in train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate accuracy, precision, recall, f1
def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='weighted')
    recall = recall_score(actual, pred, average='weighted')
    f1 = f1_score(actual, pred, average='weighted')
    return accuracy, precision, recall, f1

# Local MLFLOW server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLOPS_ASSIGNMENT")

# Start MLflow
mlflow.start_run()

# Logistic Regression with GridSearchCV
log_reg = LogisticRegression(max_iter=10000)
param_grid_log_reg = {
    'C': [0.01, 0.1, 1, 10, 100]
}
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='f1_weighted')
grid_search_log_reg.fit(X_train, y_train)
best_log_reg = grid_search_log_reg.best_estimator_
y_pred_log_reg = best_log_reg.predict(X_test)

log_reg_accuracy, log_reg_precision, log_reg_recall, log_reg_f1 = eval_metrics(y_test, y_pred_log_reg)
mlflow.log_metric("LR_accuracy", log_reg_accuracy)
mlflow.log_metric("LR_precision", log_reg_precision)
mlflow.log_metric("LR_recall", log_reg_recall)
mlflow.log_metric("LR_f1", log_reg_f1)
mlflow.sklearn.log_model(best_log_reg, "logistic_regression_model", input_example=X_test[:1])

# Log best parameters for Logistic Regression
for param, value in grid_search_log_reg.best_params_.items():
    mlflow.log_param(f"LR_{param}", value)

print(f"Best parameters for Logistic Regression: {grid_search_log_reg.best_params_}")

# Random Forest with GridSearchCV
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

rf_accuracy, rf_precision, rf_recall, rf_f1 = eval_metrics(y_test, y_pred_rf)
mlflow.log_metric("RF_accuracy", rf_accuracy)
mlflow.log_metric("RF_precision", rf_precision)
mlflow.log_metric("RF_recall", rf_recall)
mlflow.log_metric("RF_f1", rf_f1)
mlflow.sklearn.log_model(best_rf, "random_forest_model", input_example=X_test[:1])

# Log best parameters for Random Forest
for param, value in grid_search_rf.best_params_.items():
    mlflow.log_param(f"RF_{param}", value)

print(f"Best parameters for Random Forest: {grid_search_rf.best_params_}")

# Support Vector Machine with GridSearchCV
svc = SVC()
param_grid_svc = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='f1_weighted')
grid_search_svc.fit(X_train, y_train)
best_svc = grid_search_svc.best_estimator_
y_pred_svc = best_svc.predict(X_test)

svc_accuracy, svc_precision, svc_recall, svc_f1 = eval_metrics(y_test, y_pred_svc)
mlflow.log_metric("SVM_accuracy", svc_accuracy)
mlflow.log_metric("SVM_precision", svc_precision)
mlflow.log_metric("SVM_recall", svc_recall)
mlflow.log_metric("SVM_f1", svc_f1)
mlflow.sklearn.log_model(best_svc, "svm_model", input_example=X_test[:1])

# Log best parameters for SVM
for param, value in grid_search_svc.best_params_.items():
    mlflow.log_param(f"SVM_{param}", value)

print(f"Best parameters for SVM: {grid_search_svc.best_params_}")

# Stop MLFLOW
mlflow.end_run()

# Saving models locally
joblib.dump(best_log_reg, '../models/LR_model.joblib')
joblib.dump(best_rf, '../models/RF_model.joblib')
joblib.dump(best_svc, '../models/SVM_model.joblib')

# Printing metrics for reference
print(f"Logistic Regression metrics:\n Accuracy: {log_reg_accuracy}\n Precision: {log_reg_precision}\n Recall: {log_reg_recall}\n F1 Score: {log_reg_f1}")
print(f"Random Forest metrics:\n Accuracy: {rf_accuracy}\n Precision: {rf_precision}\n Recall: {rf_recall}\n F1 Score: {rf_f1}")
print(f"SVM metrics:\n Accuracy: {svc_accuracy}\n Precision: {svc_precision}\n Recall: {svc_recall}\n F1 Score: {svc_f1}")
