import numpy as np
import matplotlib.pyplot as plt
import pickle

def accuary(predict_y, y):
    predicted_y = np.array(predict_y > 0.5, dtype=int).reshape(-1)
    y = y.reshape(-1)
    return np.sum(predicted_y == y) / y.shape[0]

def mse(predict_y,  y):
    sub = predict_y - y
    return np.multiply(sub, sub).sum() / y.shape[0]

class LMSE():
    def __init__(self, input_dim, lr=0.001, l2_rate=0.0, epoch_num=1):
        self.lr = lr
        self.l2_rate = l2_rate  # L2 Regularization rate
        self.epoch_num = epoch_num
        self.input_dim = input_dim
        self.weight = np.mat(np.random.randn(input_dim, 1))
        self.bias = 0

    def predict(self, X):
        return X * self.weight + self.bias

    def _update_param(self, X, predict_y, y):
        m = y.shape[0]
        sub =  predict_y - y
        self.bias = self.bias - self.lr * (sub.sum() / m)
        self.weight = self.weight - self.lr * X.T * sub - (self.l2_rate / m) * self.weight

    def fit(self, train_X, train_y, batch_size=None):
        train_size = train_X.shape[0]
        if batch_size is None:
            batch_size = train_size
        for epoch in range(self.epoch_num):
            b = 0
            epoch_loss = 0.0
            epoch_acc = 0.0
            count = 0
            for e in range(batch_size, train_size+batch_size, batch_size):
                x = np.mat(train_X[b:e].reshape(-1, self.input_dim))
                y = train_y[b:e].reshape(-1, 1)
                predicted_y = self.predict(x)
                self._update_param(x, predicted_y, y)
                loss = mse(predicted_y, y)
                epoch_loss += loss
                acc = accuary(predicted_y, y)
                epoch_acc += acc
                count += 1            
                b = e
                if b >= train_size:
                    break
            print("Epoch: %d, Loss: %.3f, Acc: %.3f" % (epoch, epoch_loss/count, epoch_acc/count))
    
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
    train_y = np.array([1, 1, 1, 0, 0, 0], dtype=np.int)
    return train_X, train_y


if __name__ == "__main__":
    epoch_num = 50
    train_X, train_y = load_data()
    train_size = train_X.shape[0]
    feat_dim = train_X.shape[1]
    clf = LMSE(input_dim=feat_dim, lr=0.1, l2_rate=0.0, epoch_num=10)
    clf.fit(train_X, train_y, batch_size=2)
    weight = clf.weight.reshape(-1)
    weight = weight.tolist()[0]
    bias = clf.bias-0.5
    print(weight)
    print(bias)
    weight = [0.4469281, 0.185913]
    bias = -0.6
    for idx, cl in enumerate(np.unique(train_y)):
        plt.scatter(x=train_X[train_y == cl, 0], y=train_X[train_y == cl, 1],
        alpha=0.8, label=cl)
    x = np.linspace(0, 2, 10)
    y = (weight[0]*x + bias) / (-weight[1])
    plt.plot(x, y, color="red")  
    plt.show()