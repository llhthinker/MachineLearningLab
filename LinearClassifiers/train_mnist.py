import csv
import numpy as np
import json
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, accuracy_score, f1_score
from LMSE import LMSE
from perceptron import Perceptron
import os

def scaling(X):
    mean = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, ddof=1, keepdims=True)
    X = (X - mean) / sigma
    return X

def load_data(train_sample_path, train_label_path):
    train_X = np.loadtxt(train_sample_path, dtype=np.float32, delimiter=",")
    train_y = np.loadtxt(train_label_path, dtype=np.int64, delimiter=",")
    train_X = scaling(train_X)
    print(train_X.shape)
    print(train_y.shape)
    return train_X, train_y

def train(train_X, train_y, saved_model_path):
    train_size = train_X.shape[0]
    feat_dim = train_X.shape[1]
    epoch_num = 10
    lr = 0.0002
    l2_rate = 1e-6
    clf = LMSE(input_dim=feat_dim, lr=lr, l2_rate=l2_rate, epoch_num=epoch_num)
    clf.fit(train_X, train_y, batch_size=16)
    clf.save_model(saved_model_path)

def predict(test_X, saved_model_path):
    feat_dim = test_X.shape[1]
    clf = LMSE(input_dim=feat_dim)
    clf.load_model(saved_model_path)
    predict_y = clf.predict(test_X)
    return predict_y

def train_perceptron(train_X, train_y, saved_model_path):
    train_size = train_X.shape[0]
    feat_dim = train_X.shape[1]
    epoch_num = 15
    lr = 0.0001
    clf = Perceptron(input_dim=feat_dim, lr=lr, epoch_num=epoch_num)
    clf.fit(train_X, train_y)
    clf.save_model(saved_model_path)

def predict_perceptron(test_X, saved_model_path):
    test_size = test_X.shape[0]
    feat_dim = test_X.shape[1]
    clf = Perceptron(input_dim=feat_dim)
    clf.load_model(saved_model_path)
    predict_y = clf.predict(test_X)
    predict_y = np.array(predict_y).reshape((test_size, 1))
    return predict_y

def one_vs_all_train(train_X, train_y):
    lmse_flag = False
    for true_label in range(10):
        print("true label:", true_label)
        cur_saved_model_path = saved_model_path + "_" + str(true_label)
        cur_train_y = (train_y==true_label).astype(np.int64)
        if lmse_flag:
            train(train_X, cur_train_y, cur_saved_model_path)
        else:
            zero_index = np.where(cur_train_y==0)
            cur_train_y[zero_index] = -1
            train_perceptron(train_X, cur_train_y, cur_saved_model_path)

def one_vs_all_predict(test_X, test_y):
    predict_y = []
    for true_label in range(10):
        cur_saved_model_path = saved_model_path + "_" + str(true_label)
        # one_predict_y = predict(test_X, cur_saved_model_path)
        one_predict_y = predict_perceptron(test_X, cur_saved_model_path)
        predict_y.append(one_predict_y)
        print("Prediced ", true_label)
    predict_y = np.hstack(predict_y)
    predict_y = np.argmax(predict_y, axis=1)
    print(classification_report(predict_y, test_y, digits=3))
    print("Test acc:", accuracy_score(predict_y, test_y))


if __name__ == "__main__":
    train_sample_path = "./TrainSamples.csv"
    train_label_path = "./TrainLabels.csv"
	if not os.path.exists("./checkpoints/"):
		os.makedirs("./checkpoints/")
    # saved_model_path = "./checkpoints/LMSE.model"
    saved_model_path = "./checkpoints/Perceptron.model"
    re_training = True
    if re_training:
        train_X, train_y = load_data(train_sample_path, train_label_path)
        one_vs_all_train(train_X, train_y)

    test_sample_path = "./TestSamples.csv"
    test_label_path = "./TestLabels.csv"
    test_X, test_y = load_data(test_sample_path, test_label_path)
    one_vs_all_predict(test_X, test_y)

