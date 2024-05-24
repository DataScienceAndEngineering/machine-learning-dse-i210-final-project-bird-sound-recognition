import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def load_models():
    knn = joblib.load('/content/knn_model.pkl')
    rf = joblib.load('/content/rf_model.pkl')
    xgb_clf = joblib.load('/content/xgb_model.pkl')
    le = joblib.load('/content/label_encoder.pkl')
    return knn, rf, xgb_clf, le

def evaluate_model(model, X_test, y_test, le, model_name):
    y_pred = model.predict(X_test)
    y_test_encoded = le.transform(y_test)
    print(f"{model_name} Classifier")
    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

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
