import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def load_models():
    knn = joblib.load('/content/knn_model.pkl')
    rf = joblib.load('/content/rf_model.pkl')
    xgb_clf = joblib.load('/content/xgb_model.pkl')
    le = joblib.load('/content/label_encoder.pkl')
    return knn, rf, xgb_clf, le

def plot_confusion_matrix(y_true, y_pred, classes, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_test, y_score, classes, title):
    y_test_bin = label_binarize(y_test, classes=np.arange(len(classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'Class {classes[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def evaluate_model(model, X_test, y_test, le, model_name, shap_analysis=False):
    y_pred = model.predict(X_test)
    y_test_encoded = le.transform(y_test)
    y_score = model.predict_proba(X_test)

    print(f"{model_name} Classifier")
    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

    plot_confusion_matrix(y_test_encoded, y_pred, le.classes_, f'Confusion Matrix ({model_name})')
    plot_roc_curve(y_test_encoded, y_score, le.classes_, f'ROC Curve ({model_name})')

def main():
    # Load datasets
    X_test = np.load('/content/X_test.npy')
    y_test = np.load('/content/y_test.npy', allow_pickle=True)

    # Load models and label encoder
    knn, rf, xgb_clf, le = load_models()

    # Evaluate models
    evaluate_model(knn, X_test, y_test, le, "KNN")
    evaluate_model(rf, X_test, y_test, le, "Random Forest")
    evaluate_model(xgb_clf, X_test, y_test, le, "XGBoost")

if __name__ == "__main__":
    main()
