# Importing Necessary Libraries
import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Breast Cancer dataset from scikit-learn
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

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
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOPS_ASSIGNMENT")

# Start MLflow
mlflow.start_run()



# Logistic Regression
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

log_reg_accuracy, log_reg_precision, log_reg_recall, log_reg_f1 = eval_metrics(y_test, y_pred_log_reg)
mlflow.log_metric("LR_accuracy", log_reg_accuracy)
mlflow.log_metric("LR_precision", log_reg_precision)
mlflow.log_metric("LR_recall", log_reg_recall)
mlflow.log_metric("LR_f1", log_reg_f1)
mlflow.sklearn.log_model(log_reg, "logistic_regression_model", input_example=X_test[:1])




# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_accuracy, rf_precision, rf_recall, rf_f1 = eval_metrics(y_test, y_pred_rf)
mlflow.log_metric("RF_accuracy", rf_accuracy)
mlflow.log_metric("RF_precision", rf_precision)
mlflow.log_metric("RF_recall", rf_recall)
mlflow.log_metric("RF_f1", rf_f1)
mlflow.sklearn.log_model(rf, "random_forest_model", input_example=X_test[:1])




# Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

svc_accuracy, svc_precision, svc_recall, svc_f1 = eval_metrics(y_test, y_pred_svc)
mlflow.log_metric("SVM_accuracy", svc_accuracy)
mlflow.log_metric("SVM_precision", svc_precision)
mlflow.log_metric("SVM_recall", svc_recall)
mlflow.log_metric("SVM_f1", svc_f1)
mlflow.sklearn.log_model(svc, "svm_model", input_example=X_test[:1])



# Stop MLFLOW
mlflow.end_run()

# Saving models locally
joblib.dump(log_reg, '../models/LR_model.joblib')
joblib.dump(rf, '../models/RF_model.joblib')
joblib.dump(svc, '../models/SVM_model.joblib')

# Printing metrics for reference
print(f"Logistic Regression metrics:\n Accuracy: {log_reg_accuracy}\n Precision: {log_reg_precision}\n Recall: {log_reg_recall}\n F1 Score: {log_reg_f1}")
print(f"Random Forest metrics:\n Accuracy: {rf_accuracy}\n Precision: {rf_precision}\n Recall: {rf_recall}\n F1 Score: {rf_f1}")
print(f"SVM metrics:\n Accuracy: {svc_accuracy}\n Precision: {svc_precision}\n Recall: {svc_recall}\n F1 Score: {svc_f1}")
