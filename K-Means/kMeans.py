import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
import datetime

def distance(x1, x2):
    dis = np.sum(np.square(x1 - x2))
    return dis

class KMeans:
    def __init__(self, X, k):
        self.X = X
        self.k = k
        self.train_size = X.shape[0]
        self.dim = self.X.shape[1]
        self.centers = dict()
        self.clusters = defaultdict(list)
        self.loss = float('inf')

    def init_center(self):
        center_idx = random.sample(range(self.train_size), self.k)
        return self.X[center_idx].astype(np.float)

    def build_cluster(self):
        self.clusters = defaultdict(list)
        loss = 0
        for i in range(self.train_size):
            min_dis = float('inf')
            min_c_idx = -1
            for c_idx in range(self.k):
                c = self.centers[c_idx]
                cur_dis = distance(self.X[i], c)
                
                if cur_dis < min_dis:
                    min_dis = cur_dis
                    min_c_idx = c_idx
            loss += min_dis
            self.clusters[min_c_idx].append(i)
        self.loss = loss / self.train_size

    def update_center(self):
        flag = False
        for i in range(self.k):
            cur_cluster = self.clusters[i]
            updated_c = np.mean(self.X[cur_cluster], axis=0)
            dis = distance(self.centers[i], updated_c)
            if dis > 1e-7:
                self.centers[i] = updated_c
                flag = True
        return flag

    def fit(self, random_num=1):
        min_loss = float('inf')
        final_center = None
        final_clusters = None
        for i in range(random_num):
            self.centers = self.init_center()
            self.build_cluster()
            iter_num = 0
            while self.update_center():
                self.build_cluster()
                iter_num += 1
                if iter_num % 1 == 0:
                    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print("%s, Random num: %d, Iter num: %d, Loss: %.3f" % (nowTime, i+1, iter_num, self.loss))

            if self.loss < min_loss:
                min_loss = self.loss
                final_center = copy.deepcopy(self.centers)
                final_clusters = copy.deepcopy(self.clusters)
            

        self.centers = final_center
        self.clusters = final_clusters
        self.loss = min_loss
        

    def print_clusters(self):
        print("train size: ", self.train_size)
        print("Loss:", self.loss)
        for i in range(self.k):
            print("Center: ", self.centers[i])
            print("Items: ", self.X[self.clusters[i]].tolist())
            

    def show_clusters(self):
        assert self.dim == 2
        for i in range(self.k):
            plt.scatter(x=self.centers[i][0], y=self.centers[i][1], s=80, marker="v")
            x_values = self.X[self.clusters[i]][:,0]
            y_values = self.X[self.clusters[i]][:,1]
            plt.scatter(x_values, y_values, s=50)
        plt.grid(ls="-.")
        plt.show()
        
        
def test():
    k = 2
    X = [(0, 0), (1, 0), (0, 1), (1, 1),
    (2, 1), (1, 2), (2, 2), (3, 2),
    (6, 6), (7, 6), (8, 6), (7, 7),
    (8, 7), (9, 7), (7, 8), (8, 8),
    (9, 8), (8, 9), (9, 9)]
    X = np.array(X)
    k_means = KMeans(X, k)
    k_means.fit(random_num=10)
    k_means.print_clusters()
    k_means.show_clusters()

if __name__ == "__main__":
    test()