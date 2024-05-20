# -*- coding: utf-8 -*-
"""predict_model_deep.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wlJTzQ7yq02Buqms4_JYfvthN5R03t4y
"""

import zipfile
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from make_dataset_deep import make_filepaths_labels, train_test_split_deep


def unzip_model_file(zip_file_path, extract_to):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def model_load_predict(model_path, dataset_test):
    model_eval = tf.keras.models.load_model(model_path)
    print(model_eval.evaluate(dataset_test, return_dict=True))
    y_logits = model_eval.predict(dataset_test)
    y_prob = tf.nn.softmax(y_logits, axis=1)
    y_pred = tf.argmax(y_logits, axis=1)
    y_true = tf.concat(list(dataset_test.map(lambda s,lab: lab)), axis=0)
    return y_prob, y_pred, y_true

def main():
    features = ['LinearSpectrogram', 'MelSpectrogram', 'MFCC']
    metadata_path = '/content/drive/MyDrive/archive (17)/bird_songs_metadata.csv'
    audio_path = '/content/drive/MyDrive/archive (17)/wavfiles/'
    file_paths, labels = make_filepaths_labels(metadata_path, audio_path)
    dataset_train, dataset_test = train_test_split_deep(file_paths, labels, features[1])
    zip_file_path = '/model_CNN_mel.zip'
    extract_to_path = '/content/model'
    model_path = '/content/model'
    unzip_model_file(zip_file_path, extract_to_path)
    y_prob, y_pred, y_true = model_load_predict(model_path, dataset_test)
    print(roc_auc_score(y_true, y_prob, average='macro', multi_class='ovo'))

if __name__ == "__main__":
    main()