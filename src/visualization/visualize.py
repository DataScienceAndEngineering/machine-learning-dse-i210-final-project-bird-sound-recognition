# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary packages
!pip install shap

# Import necessary libraries
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from collections import defaultdict
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shap
import librosa.display
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import export_text
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# Load metadata
metadata_path = '/content/drive/MyDrive/Bird sound classification data/bird_songs_metadata.csv'
zip_path = '/content/drive/MyDrive/Bird sound classification data/archive (1).zip'
metadata = pd.read_csv(metadata_path)
metadata.head()

# Extract features from audio files
def extract_features(file_path, zip_ref):
    with zip_ref.open(file_path) as file:
        y, sr = librosa.load(file, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        return np.mean(S_DB, axis=1)  # Using mean as a simple feature

# Prepare dataset
features = []
labels = []

with zipfile.ZipFile(zip_path, 'r') as z:
    for index, row in metadata.iterrows():
        file_path = os.path.join('wavfiles', row['filename'])
        if file_path in z.namelist():
            features.append(extract_features(file_path, z))
            labels.append(row['species'])

features = np.array(features)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
# Get the original indices of the test set
_, X_test_indices, _, y_test_indices = train_test_split(np.arange(len(labels_encoded)), labels_encoded, test_size=0.2, random_state=42)


# Log Mel and MFCC combined
def extract_mfcc_features(file_path, zip_ref):
    with zip_ref.open(file_path) as file:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)

# Prepare MFCC features
mfcc_features = []
labels_mfcc = []

with zipfile.ZipFile(zip_path, 'r') as z:
    for index, row in metadata.iterrows():
        file_path = os.path.join('wavfiles', row['filename'])
        if file_path in z.namelist():
            mfcc_features.append(extract_mfcc_features(file_path, z))
            labels_mfcc.append(row['species'])

mfcc_features = np.array(mfcc_features)
label_encoder_mfcc = LabelEncoder()
labels_encoded_mfcc = label_encoder_mfcc.fit_transform(labels_mfcc)

# Split the dataset for MFCC features
X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(mfcc_features, labels_encoded_mfcc, test_size=0.2, random_state=42)

# Combined features
combined_features = np.hstack((features, mfcc_features))
labels_combined = labels_encoded  # Assuming the same labels

# Split the dataset for combined features
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(combined_features, labels_combined, test_size=0.2, random_state=42)

# Classes
classes = label_encoder.classes_


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
def bar_graph_species(metadata):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=metadata, x='species')
    plt.title('Distribution of Species')
    plt.xticks(rotation=45)
    plt.show()
def metadata_info(metadata):
    return metadata.info()

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

def log_mel_rf_analysis(X_train, y_train, X_test, y_test, classes):
    clf, y_pred = classification_report_rf(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, classes, 'Confusion Matrix for Log Mel')
    plot_feature_importance(clf.feature_importances_, 'Feature Importances for Log Mel')
    cumulative_feature_importance(clf.feature_importances_, 'Cumulative Feature Importances for Log Mel')
    y_score = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_score, classes, 'ROC Curve for Log Mel (Random Forest)')

def mfcc_rf_analysis(X_train, y_train, X_test, y_test, classes):
    clf, y_pred = classification_report_rf(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, classes, 'Confusion Matrix for MFCC')
    plot_feature_importance(clf.feature_importances_, 'Feature Importances for MFCC')
    cumulative_feature_importance(clf.feature_importances_, 'Cumulative Feature Importances for MFCC')
    y_score = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_score, classes, 'ROC Curve for MFCC (Random Forest)')

def combined_rf_analysis(X_train, y_train, X_test, y_test, classes):
    clf, y_pred = classification_report_rf(X_train, y_train, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, classes, 'Confusion Matrix for Combined Features (Log Mel + MFCC)')
    plot_feature_importance(clf.feature_importances_, 'Feature Importances for Combined Features (Log Mel + MFCC)')
    y_score = clf.predict_proba(X_test)
    plot_roc_curve(y_test, y_score, classes, 'ROC Curve for Combined Features (Log Mel + MFCC)')

def classification_report_combined_rf(X_train, y_train, X_test, y_test):
    clf = train_rf_model(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    cross_val_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("Cross-validation Scores:", cross_val_scores)
    print("Mean Cross-validation Score:", np.mean(cross_val_scores))
    return clf

def classification_report_knn_mel(X_train, y_train, X_test, y_test):
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    y_pred_knn = knn_classifier.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred_knn))
    return knn_classifier

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

def confusion_matrix_classification_report_svm(X_train, y_train, X_test, y_test):
    svm_classifier = SVC(probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred_svm = svm_classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_svm, classes, 'Confusion Matrix (SVM)')
    print("Classification Report:\n", classification_report(y_test, y_pred_svm))

def shap_svm(X_train, X_test, y_train):
    svm_classifier = SVC(probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    explainer = shap.KernelExplainer(svm_classifier.predict, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

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

def shap_xgboost(X_train, y_train, X_test):
    xg_model = XGBClassifier(random_state=42)
    xg_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xg_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)

# Example function calls after setup
bar_graph_species(metadata)
mel_spectrogram_all_species(metadata, zip_path)
log_mel_rf_analysis(X_train, y_train, X_test, y_test, classes)
mfcc_rf_analysis(X_train_mfcc, y_train_mfcc, X_test_mfcc, y_test_mfcc, classes)
combined_rf_analysis(X_train_combined, y_train_combined, X_test_combined, y_test_combined, classes)
classification_report_combined_rf(X_train_combined, y_train_combined, X_test_combined, y_test_combined)
knn_classifier = classification_report_knn_mel(X_train, y_train, X_test, y_test)
high_low_confidence_spectrograms_knn(zip_path, metadata, X_test, knn_classifier)
confusion_matrix_classification_report_svm(X_train_combined, y_train_combined, X_test_combined, y_test_combined)
'''
shap_svm(X_train_combined, X_test_combined, y_train_combined)
confusion_matrices_xgboost(X_train, y_train, X_test, y_test, X_train_mfcc, y_train_mfcc, X_test_mfcc, y_test_mfcc, X_train_combined, y_train_combined, X_test_combined, y_test_combined)
classification_report_feature_importance_xgboost(X_train_combined, y_train_combined, X_test_combined, y_test_combined)
shap_xgboost(X_train_combined, y_train_combined, X_test_combined)
'''
shap_svm(X_train_combined, X_test_combined, y_train_combined)

confusion_matrices_xgboost(X_train, y_train, X_test, y_test, X_train_mfcc, y_train_mfcc, X_test_mfcc, y_test_mfcc, X_train_combined, y_train_combined, X_test_combined, y_test_combined)
classification_report_feature_importance_xgboost(X_train_combined, y_train_combined, X_test_combined, y_test_combined)

shap_xgboost(X_train_combined, y_train_combined, X_test_combined)


