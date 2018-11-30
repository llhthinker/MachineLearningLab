import csv
import numpy as np
import json
from collections import Counter

from kMeans import KMeans

def scaling(X):
    max_gray = 255
    X = 2* (X - max_gray/2) / max_gray
    # mean = np.mean(X, axis=1, keepdims=True)
    # sigma = np.std(X, axis=1, ddof=1, keepdims=True)
    # X = (X - mean) / sigma
    return X

def load_data(data_path):
    mnist_file = open(data_path, 'r', encoding='utf-8')
    lines = csv.reader(mnist_file)
    data = []
    for line in lines:
        sample = [int(i) for i in line]
        data.append(sample)
    data = np.array(data, dtype=np.float)
    data = scaling(data)
    return data

def mnist_train(data, result_path):
    k = 10
    k_means = KMeans(data, k)
    k_means.fit(random_num=10)
    print(k_means.loss)
    json.dump(k_means.clusters, open(result_path, 'w', encoding='utf-8'))

def analysis(result_path, true_label_path):
    result = json.load(open(result_path, 'r', encoding='utf-8'))
    true_label = open(true_label_path, "r", encoding='utf-8').read().strip().split("\n")
    true_label = np.array([int(i) for i in true_label])
    for i in range(10):
        i = str(i)
        cur_res = true_label[result[i]]
        print("Cluster: ", i, Counter(cur_res))
        
if __name__ == "__main__":
    data_path = "./ClusterSamples.csv"
    result_path = "./cluster_result.json"
    data = load_data(data_path)
    # mnist_train(data, result_path)
    true_label_path = "./SampleLabels.csv"
    analysis(result_path, true_label_path)