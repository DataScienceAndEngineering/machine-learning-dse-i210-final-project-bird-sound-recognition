import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from google.colab import drive

def load_models():
    knn = joblib.load('/content/drive/MyDrive/knn_model.pkl')
    rf = joblib.load('/content/drive/MyDrive/rf_model.pkl')
    xgb_clf = joblib.load('/content/drive/MyDrive/xgb_model.pkl')
    le = joblib.load('/content/drive/MyDrive/label_encoder.pkl')
    return knn, rf, xgb_clf, le

def evaluate_model(model, X_test, y_test, le, model_name):
    y_pred = model.predict(X_test)
    y_test_encoded = le.transform(y_test)
    print(f"{model_name} Classifier")
    print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

def main():
    drive.mount('/content/drive')
    X_test = np.load('/content/drive/MyDrive/X_test.npy')
    y_test = np.load('/content/drive/MyDrive/y_test.npy', allow_pickle=True)
    knn, rf, xgb_clf, le = load_models()
    evaluate_model(knn, X_test, y_test, le, "KNN")
    evaluate_model(rf, X_test, y_test, le, "Random Forest")
    evaluate_model(xgb_clf, X_test, y_test, le, "XGBoost")

if __name__ == "__main__":
    main()
