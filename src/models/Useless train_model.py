import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

def train_models(X_train, y_train):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train_encoded)
    joblib.dump(knn, '/content/drive/MyDrive/knn_model.pkl')

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_encoded)
    joblib.dump(rf, '/content/drive/MyDrive/rf_model.pkl')

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_clf.fit(X_train, y_train_encoded)
    joblib.dump(xgb_clf, '/content/drive/MyDrive/xgb_model.pkl')

    joblib.dump(le, '/content/drive/MyDrive/label_encoder.pkl')

def main():
    X_train = np.load('/content/drive/MyDrive/X_train.npy')
    y_train = np.load('/content/drive/MyDrive/y_train.npy', allow_pickle=True)
    train_models(X_train, y_train)

if __name__ == "__main__":
    main()
