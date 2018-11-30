import numpy as np
import pickle
import os
import math

from sklearn.metrics import classification_report, accuracy_score

class GMM:
    def __init__(self, K, input_dim):
        self.K = K
        self.input_dim = input_dim
        self._init_params()
    
    def print_params(self):
        for i in range(self.K):
            print("alpha"+str(i+1)+":", self.alpha[i])
            print("mu"+str(i+1)+":", self.mu[i])
            print("Sigma"+str(i+1)+":", self.Sigma[i])

    def get_gaussian(self, x, mu, Sigma):
        dim = Sigma.shape[0]   # 计算维度
        Sigma_det = np.linalg.det(Sigma+np.eye(dim)*0.001)
        Sigma_inv = np.linalg.inv(Sigma+np.eye(dim)*0.001)
        m = x - mu
        z = -0.5 * np.dot(np.dot(m, Sigma_inv),m)    # 计算exp()里的值
        return 1.0 / (np.power(np.power(2*np.pi, dim)*abs(Sigma_det), 0.5)) * np.exp(z)

    def _init_params(self):
        self.alpha = [1.0 / self.K] * self.K
        self.mu = [np.random.randn(self.input_dim) for i in range(self.K)]
        self.Sigma = [np.mat(np.random.rand(self.input_dim, self.input_dim)) for i in range(self.K)]

    def fit(self, X, max_iter_num=100):
        N = X.shape[0]  # training data size
        # prob_mat表示第i个样本属于第j个混合高斯的概率
        prob_mat = [np.zeros(self.K) for i in range(N)]
        loglikelyhood = 0
        old_loglikelyhood = 1
        eps = 1e-4
        iter_num = 0 
        
        while np.abs(loglikelyhood - old_loglikelyhood) > eps:
            old_loglikelyhood = loglikelyhood
            # E step
            for i in range(N):
                res = [self.alpha[k] * self.get_gaussian(X[i], self.mu[k], self.Sigma[k]) for k in range(self.K)]
                sum_res = np.sum(res)
                for k in range(self.K):           
                    prob_mat[i][k] = res[k] / sum_res
            
            # M step
            for k in range(self.K):
                Nk = np.sum([prob_mat[i][k] for i in range(N)])
                self.alpha[k] = Nk / N
                self.mu[k] = np.sum([prob_mat[i][k] * X[i] for i in range(N)], axis=0) / Nk
                diff = X - self.mu[k]
                self.Sigma[k] = np.sum([prob_mat[i][k] * diff[i].reshape(self.input_dim, 1) * diff[i] for i in range(N)], axis=0) / Nk

            loglikelyhood = np.sum(
                        [np.log(np.sum([self.alpha[k] * self.get_gaussian(X[i], self.mu[k], self.Sigma[k]) 
                            for k in range(self.K)])) for i in range(N)]) / N
            iter_num += 1
            if math.isnan(loglikelyhood):
                self._init_params()
                loglikelyhood = 0
                old_loglikelyhood = 1
            print("Iter num: %d, loglikelyhood: %.6f" % (iter_num, loglikelyhood))
            if iter_num > max_iter_num:
                break

    def predict_prob(self, x):
        prob = np.sum([self.alpha[k] * self.get_gaussian(x, self.mu[k], self.Sigma[k]) for k in range(self.K)])
        return prob

    def save_model(self, file_path):
        params = {
            'alpha': self.alpha,
            'mu': self.mu,
            'Sigma': self.Sigma
        }
        pickle.dump(params, open(file_path, "wb"))
    
    def load_model(self, file_path):
        params = pickle.load(open(file_path, "rb"))
        self.alpha = params['alpha']
        self.mu = params['mu']
        self.Sigma = params['Sigma']


def load_data(data_path):
    X = np.loadtxt(data_path, delimiter=',')
    return X

def train(X, saved_model_path, k=2, force=False):
    input_dim = X.shape[1]
    gmm = GMM(k, input_dim)
    if os.path.exists(saved_model_path) and not force:
        gmm.load_model(saved_model_path)
    else:
        gmm.fit(X)
        gmm.save_model(saved_model_path)
    # gmm.print_params()
    return gmm

def test(X, gmm_models, true_label=None):
    N = X.shape[0]
    kind = len(gmm_models)
    true_count = 0
    predicted_labels = []
    for i in range(N):
        prob = [gmm.predict_prob(X[i]) for gmm in gmm_models]
        idx = np.argmax(prob)
        if true_label is not None and idx == true_label:
            true_count += 1
        predicted_labels.append(idx)
    if true_label is not None:
        print("True count: %d, Acc: %.3f" % (true_count, true_count / N))
    return predicted_labels

def simulation():
    gmm_models = []
    data_path_list = ["./Train1.csv", "./Train2.csv"]
    for i, data_path in enumerate(data_path_list):
        X = load_data(data_path)
        saved_model_path = "./checkpoints/gmm"+str(i+1)+".model"
        gmm = train(X, saved_model_path, force=False)
        gmm_models.append(gmm)
    data_path_list = ["./Test1.csv", "./Test2.csv"]
    for i, data_path in enumerate(data_path_list):
        X = load_data(data_path)
        true_label = i
        test(X, gmm_models, true_label)

def scaling(X):
    mean = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, ddof=1, keepdims=True)
    X = (X - mean) / sigma
    return X

def train_mnist():
    gmm_models = []
    train_sample_path = "./TrainSamples.csv"
    train_label_path = "./TrainLabels.csv"
    train_X = load_data(train_sample_path)
    train_X = scaling(train_X)
    train_y = load_data(train_label_path).astype(np.int64)
    k = 5
    for true_label in range(10):
        print("Training model:", true_label)
        indices = np.where(train_y==true_label)
        cur_train_X = train_X[indices]
        saved_model_path = "./checkpoints/gmm_label"+str(true_label) + "_k" + str(k) + ".model"
        gmm = train(cur_train_X, saved_model_path, k=k, force=False)
        gmm_models.append(gmm)

    test_sample_path = "./TestSamples.csv"
    test_label_path = "./TestLabels.csv"
    test_X = load_data(test_sample_path)
    test_X = scaling(test_X)
    test_y = load_data(test_label_path).astype(np.int64)
    predicted_y = test(test_X, gmm_models)
    print(classification_report(predicted_y, test_y))
    print(accuracy_score(test_y, predicted_y))

if __name__ == "__main__":
    # simulation()
    train_mnist()
