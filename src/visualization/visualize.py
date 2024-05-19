from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import zipfile
import librosa
import librosa.display
from collections import defaultdict
import os

def train_rf_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def plot_confusion_matrix(y_true, y_pred, classes, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_feature_importance(importances, title):
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importances)), importances)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title(title)
    plt.show()

def cumulative_feature_importance(importances, title):
    sorted_indices = np.argsort(importances)[::-1]
    cumulative_importances = np.cumsum(importances[sorted_indices])
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(cumulative_importances)), cumulative_importances)
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_test, y_score, classes, title):
    y_test_bin = label_binarize(y_test, classes=classes)
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

def classification_report_rf(X_train, y_train, X_test, y_test):
    clf = train_rf_model(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return clf, y_pred

def plot_mel_spectrogram(file_path, zip_ref, title):
    with zip_ref.open(file_path) as file:
        y, sr = librosa.load(file, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()

#bar graph to check species
def bar_graph_species(metadata):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=metadata, x='species')
    plt.title('Distribution of Species')
    plt.xticks(rotation=45)
    plt.show()

#check information about the metadata
def metadata_info(metadata):
    return metadata.info()

#Mel spctrogram limited to first 5 only
def mel_spectrogram_all_species(metadata, zip_path):
    species_count = defaultdict(int)
    max_plots_per_species = 5
    with zipfile.ZipFile(zip_path, 'r') as z:
        for index, row in metadata.iterrows():
            species = row['species']
            if species_count[species] < max_plots_per_species:
                file_path = os.path.join('wavfiles', row['filename'])
                if file_path in z.namelist():
                    with z.open(file_path) as file:
                        y, sr = librosa.load(file, sr=None)
                        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                        S_DB = librosa.power_to_db(S, ref=np.max)
                        plt.figure(figsize=(10, 4))
                        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
                        plt.colorbar(format='%+2.0f dB')
                        plt.title(f'Spectrogram for {species} ({file_path})')
                        plt.show()
                        species_count[species] += 1

#Feature importance, Confusion Matrix, Cumulative feature importance, and ROC for Log Mel using Random Forest
def log_mel_rf_analysis(X_train, y_train, X_test, y_test, classes):
    clf, y_pred = classification_report_rf(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, classes, 'Confusion Matrix for Log Mel')
    plot_feature_importance(clf.feature_importances_, 'Feature Importances for Log Mel')
    cumulative_feature_importance(clf.feature_importances_, 'Cumulative Feature Importances for Log Mel')
    y_score = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_score, classes, 'ROC Curve for Log Mel (Random Forest)')


# Feature importance, Confusion Matrix, Cumulative feature importance, and ROC for MFCC using Random Forest
def mfcc_rf_analysis(X_train, y_train, X_test, y_test, classes):
    clf, y_pred = classification_report_rf(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, classes, 'Confusion Matrix for MFCC')
    plot_feature_importance(clf.feature_importances_, 'Feature Importances for MFCC')
    cumulative_feature_importance(clf.feature_importances_, 'Cumulative Feature Importances for MFCC')
    y_score = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_score, classes, 'ROC Curve for MFCC (Random Forest)')

#Confusion Matrix, ROC, and Feature importance for Combined Log Mel+MFCC using Random Forest
def combined_rf_analysis(X_train, y_train, X_test, y_test, classes):
    clf, y_pred = classification_report_rf(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, classes, 'Confusion Matrix for Combined Features (Log Mel + MFCC)')
    plot_feature_importance(clf.feature_importances_, 'Feature Importances for Combined Features (Log Mel + MFCC)')
    y_score = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_score, classes, 'ROC Curve for Combined Features (Log Mel + MFCC)')

#Random Forest (Log Mel+MFCC) classification report and cross-validation results
def classification_report_combined_rf(X_train, y_train, X_test, y_test):
    clf = train_rf_model(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cross_val_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Cross-validation Scores:", cross_val_scores)
    print("Mean Cross-validation Score:", np.mean(cross_val_scores))
    return clf

#Classification report of Mel using KNN
def classification_report_knn_mel(X_train, y_train, X_test, y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    y_pred_knn = knn_classifier.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred_knn))
    return knn_classifier

# Top 2 High confidence and low confidence Mel Spectrograms using KNN
def high_low_confidence_spectrograms_knn(zip_path, metadata, X_test, knn_classifier):
    y_prob_knn = knn_classifier.predict_proba(X_test)
    confidence_scores = np.max(y_prob_knn, axis=1)
    high_confidence_indices = np.argsort(confidence_scores)[-2:]
    low_confidence_indices = np.argsort(confidence_scores)[:2]
    with zipfile.ZipFile(zip_path, 'r') as z:
        for idx in high_confidence_indices:
            original_index = X_test_indices[idx]
            file_name = metadata.iloc[original_index]['filename']
            file_path = os.path.join('wavfiles', file_name)
            plot_mel_spectrogram(file_path, z, f"High Confidence: Mel Spectrogram of {file_name}")
        for idx in low_confidence_indices:
            original_index = X_test_indices[idx]
            file_name = metadata.iloc[original_index]['filename']
            file_path = os.path.join('wavfiles', file_name)
            plot_mel_spectrogram(file_path, z, f"Low Confidence: Mel Spectrogram of {file_name}")

#Confusion Matrix and Classification report using Support Vector Machine
def confusion_matrix_classification_report_svm(X_train, y_train, X_test, y_test):
    svm_classifier = SVC(probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_svm, classes, 'Confusion Matrix (SVM)')
    print("Classification Report:\n", classification_report(y_test, y_pred_svm))

#SHAP with SVM
def shap_svm(X_train, X_test, y_train):
    svm_classifier = SVC(probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    explainer = shap.KernelExplainer(svm_classifier.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

# Confusion matrices of Log Mel, MFCC, Log Mel + MFCC using XGBoost
def confusion_matrices_xgboost(X_train_log_mel, y_train_log_mel, X_test_log_mel, y_test_log_mel,
                               X_train_mfcc, y_train_mfcc, X_test_mfcc, y_test_mfcc,
                               X_train_combined, y_train_combined, X_test_combined, y_test_combined):
    # Log Mel
    xgb_log_mel = xgb.XGBClassifier(random_state=42)
    xgb_log_mel.fit(X_train_log_mel, y_train_log_mel)
    y_pred_log_mel = xgb_log_mel.predict(X_test_log_mel)
    plot_confusion_matrix(y_test_log_mel, y_pred_log_mel, classes, 'Confusion Matrix - XGBoost (Log Mel)')
    
    # MFCC
    xgb_mfcc = xgb.XGBClassifier(random_state=42)
    xgb_mfcc.fit(X_train_mfcc, y_train_mfcc)
    y_pred_mfcc = xgb_mfcc.predict(X_test_mfcc)
    plot_confusion_matrix(y_test_mfcc, y_pred_mfcc, classes, 'Confusion Matrix - XGBoost (MFCC)')

    # Combined
    xgb_combined = xgb.XGBClassifier(random_state=42)
    xgb_combined.fit(X_train_combined, y_train_combined)
    y_pred_combined = xgb_combined.predict(X_test_combined)
    plot_confusion_matrix(y_test_combined, y_pred_combined, classes, 'Confusion Matrix - XGBoost (Combined)')

#Classification report, feature importance, Hyperparameters pair plot using XGBoost
def classification_report_feature_importance_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0]
    }
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    best_xgb.fit(X_train, y_train)
    y_pred_xgb = best_xgb.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_xgb, classes, 'Confusion Matrix - XGBoost')
    print(f"Classification Report:\n{classification_report(y_test, y_pred_xgb)}")
    plot_importance(best_xgb, importance_type='weight', max_num_features=10)
    plt.title('Feature Importance (weight plot)')
    plt.show()
    results = pd.DataFrame(random_search.cv_results_)
    params = results['params']
    params_df = pd.DataFrame(params.tolist())
    for column in params_df.columns:
        params_df[column] = params_df[column].astype(float)
    plt.figure(figsize=(15, 10))
    sns.pairplot(params_df)
    plt.suptitle('Pair Plot of Hyperparameters', y=1.02)
    plt.show()

#SHAP with XGBoost
def shap_xgboost(X_train, y_train, X_test):
    xg_model = XGBClassifier(random_state=42)
    xg_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xg_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

