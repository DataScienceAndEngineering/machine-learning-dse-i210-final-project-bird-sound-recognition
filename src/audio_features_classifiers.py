# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Function to evaluate the model and generate plots
def evaluate_model(model, X_test, y_test, le, model_name, shap_analysis=False):
    y_pred = model.predict(X_test)
    y_test_encoded = le.transform(y_test)
    y_score = model.predict_proba(X_test)

    # Plot confusion matrix
    plot_confusion_matrix(y_test_encoded, y_pred, le.classes_, f'Confusion Matrix ({model_name})')



def main():
    # Load datasets
    X_train_combined = np.load('/content/X_train.npy')
    y_train_combined = np.load('/content/y_train.npy', allow_pickle=True)
    X_test_combined = np.load('/content/X_test.npy')
    y_test_combined = np.load('/content/y_test.npy', allow_pickle=True)

    # Load models and label encoder
    knn = joblib.load('/content/knn_model.pkl')
    rf = joblib.load('/content/rf_model.pkl')
    xgb_clf = joblib.load('/content/xgb_model.pkl')
    le = joblib.load('/content/label_encoder.pkl')

    # Evaluate models
    evaluate_model(knn, X_test_combined, y_test_combined, le, "KNN")
    evaluate_model(rf, X_test_combined, y_test_combined, le, "Random Forest")
    evaluate_model(xgb_clf, X_test_combined, y_test_combined, le, "XGBoost")

if __name__ == "__main__":
    main()
