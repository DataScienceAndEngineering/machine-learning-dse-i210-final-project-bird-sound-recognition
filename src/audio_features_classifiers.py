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



