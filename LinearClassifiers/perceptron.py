import numpy as np
import matplotlib.pyplot as plt
import pickle

class Perceptron():
    def __init__(self, input_dim, lr=0.001, epoch_num=1):
        self.lr = lr
        self.epoch_num = epoch_num
        self.weight = np.random.randn(input_dim)
        self.bias = 0

    def predict_one(self, x):
        score = np.sum(self.weight * x) + self.bias
        return score
    
    def predict(self, X):
        test_size = X.shape[0]
        all_score = []
        for i in range(test_size):
            x = X[i]
            score = self.predict_one(x)
            all_score.append(score)
        return all_score


    def _update_param(self, x, y):
        self.weight = self.weight + self.lr * x * y
        self.bias  = self.bias + self.lr * y
    
    def fit(self, train_X, train_y):
        train_size = train_X.shape[0]
        for epoch in range(self.epoch_num):
            total_count, right_count = 0, 0
            for i in range(train_size):
                x = train_X[i]
                y = train_y[i]
                score = self.predict_one(x)
                if score >= 0:
                    predicted_y = 1    # positive
                else:
                    predicted_y = -1   # negative
                if predicted_y != y:
                    self._update_param(x, y)
                else:
                    right_count += 1
                total_count += 1
            print("Epoch: %d, Acc: %.3f" % (epoch, right_count / total_count))

    def save_model(self, file_path):
        params = {
            'weight': self.weight,
            'bias': self.bias
        }
        pickle.dump(params, open(file_path, "wb"))
        
    def load_model(self, file_path):
        params = pickle.load(open(file_path, "rb"))
        self.weight = params['weight']
        self.bias = params['bias']

def load_data():
    train_X = np.array([[1, 1], [2, 2], [2, 0], 
                [0, 0], [1, 0], [0, 1]], dtype=np.float)
    train_y = np.array([1, 1, 1, -1, -1, -1], dtype=np.int)
    return train_X, train_y


if __name__ == "__main__":
    epoch_num = 50
    train_X, train_y = load_data()
    train_size = train_X.shape[0]
    feat_dim = train_X.shape[1]
    clf = Perceptron(input_dim=feat_dim, lr=0.1, epoch_num=20)
    clf.fit(train_X, train_y)
    clf.weight = [0.26784988, 0.18176198]
    clf.bias = -0.4
    print(clf.weight)
    print(clf.bias)
    for idx, cl in enumerate(np.unique(train_y)):
        plt.scatter(x=train_X[train_y == cl, 0], y=train_X[train_y == cl, 1],
        alpha=0.8, label=cl)
    x = np.linspace(0, 2, 10)  
    y = (clf.weight[0]*x + clf.bias) / (-clf.weight[1])
    plt.plot(x, y, color="red")  
    plt.show()