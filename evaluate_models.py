from sklearn.metrics import roc_auc_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, (y_pred > 0.5)))
    print("AUROC:", roc_auc_score(y_test, y_pred))
