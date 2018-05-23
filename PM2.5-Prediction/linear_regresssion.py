#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:08:33 2017

@author: llh
"""
import numpy as np
from sklearn import datasets

def train(X, y):
    w = np.mat(np.zeros((X.shape[1],1)))
#    w = stochastic_grad_desc(X, w, y, iter_num=10, alpha=0.001, lamb=0.08)
#    w = grad_desc(X, w, y, iter_num=5000, alpha=0.001, lamb=0.08)
#    w = stochastic_adagrad_desc(X, w, y, iter_num=30, alpha=0.3, lamb=0.08)
    w = batch_adagrad_desc(X, w, y, batch_size=20, epoch=150, alpha=0.3, lamb=0.08)

    return w


def grad_desc(X, w, y, iter_num=1000, alpha=0.01, lamb=0.01):
    m = y.shape[0]
    for i in range(iter_num):
        tmp_sub = (X * w) - y
#        print(w[0,0])
        w[0,0] = w[0,0] - alpha * (tmp_sub.sum() / m)
        w[1:] = w[1:] - alpha * ((X[:,1:].T * tmp_sub) / m) - (lamb / m) * w[1:]
        if i % 1000 == 0:
            print(cost(predict(X,w), y))
        
    return w


def adagrad_desc(X, w, y, iter_num=500, alpha=0.01, lamb=0.01):
    m = y.shape[0]
    grad_bias_sum = 0
    grad_sum = np.mat(np.zeros((w.shape[0]-1, 1)))
    for i in range(iter_num):
        tmp_sub = (X * w) - y
#        print(w[0,0])
        grad_bias = tmp_sub.sum() / m
        grad_bias_sum += grad_bias ** 2 
        w[0,0] = w[0,0] - alpha / (np.sqrt(grad_bias_sum)) * grad_bias
        
        grad = (X[:,1:].T * tmp_sub) / m
        grad_sum += np.multiply(grad, grad)
        w[1:] = w[1:] - np.multiply(np.divide(alpha, np.sqrt(grad_sum)), grad) \
        - (lamb / m) * w[1:]
        if i % 100 == 0:
            print(cost(predict(X,w), y))
        
    return w


def stochastic_grad_desc(X, w, y, iter_num=100, alpha=0.01, lamb=0.1):
    m = y.shape[0]
    for i in range(iter_num):
        tmp_sub = (X * w) - y
        for j in range(m): 
    #        print(w[0,0])
            w[0,0] = w[0,0] - alpha * (tmp_sub.sum() / m)
            w[1:] = w[1:] - alpha * ((X[j,1:].T * tmp_sub[j])) - (lamb) * w[1:]
        if i % 2 == 0:
            print(cost(predict(X,w), y))
        
    return w


def stochastic_adagrad_desc(X, w, y, iter_num=1000, alpha=0.3, lamb=0.01):
    m = y.shape[0]
    grad_bias_sum = 0
    grad_sum = np.mat(np.zeros((w.shape[0]-1, 1)))
    for i in range(iter_num):
        tmp_sub = (X * w) - y
#        print(w[0,0])
        for j in range(m):
            grad_bias = tmp_sub.sum() / m
            grad_bias_sum += grad_bias ** 2 
            w[0,0] = w[0,0] - alpha / (np.sqrt(grad_bias_sum)) * grad_bias
            
            grad = (X[j,1:].T * tmp_sub[j])
            grad_sum += np.multiply(grad, grad)
            w[1:] = w[1:] - np.multiply(np.divide(alpha, np.sqrt(grad_sum)), grad) \
            - (lamb) * w[1:]
            
        if i % 10 == 0:
            print(cost(predict(X,w), y))
        
    return w


def batch_adagrad_desc(X, w, y, batch_size=10, epoch=1000, alpha=0.3, lamb=0.01):
    m = y.shape[0]
    grad_bias_sum = 0
    grad_sum = np.mat(np.zeros((w.shape[0]-1, 1)))
    for i in range(epoch):
        tmp_sub = (X * w) - y
#        print(w[0,0])
        for j in range(0, m, batch_size):
            grad_bias = tmp_sub.sum() / m
            grad_bias_sum += grad_bias ** 2 
            w[0,0] = w[0,0] - alpha / (np.sqrt(grad_bias_sum)) * grad_bias
            if j+batch_size < m:
                grad = (X[j:j+batch_size,1:].T * tmp_sub[j:j+batch_size]) / batch_size
                grad_sum += np.multiply(grad, grad)
                w[1:] = w[1:] - np.multiply(np.divide(alpha, np.sqrt(grad_sum)), grad) \
                - (lamb / batch_size) * w[1:]
            else:
                grad = (X[j:,1:].T * tmp_sub[j:]) / (m-j)
                grad_sum += np.multiply(grad, grad)
                w[1:] = w[1:] - np.multiply(np.divide(alpha, np.sqrt(grad_sum)), grad) \
                - (lamb / (m-j)) * w[1:]

            
        if i % 10 == 0:
            print(cost(predict(X,w), y))
        
    return w


def normlize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def predict(X, w):
    return X * w


def cost(predict_y,  y):
    sub = predict_y - y
    return np.multiply(sub, sub).sum() / y.shape[0]

def test():
    boston_data = datasets.load_boston()
    X = boston_data.data
    y = boston_data.target
    w = train(X, y)
    print(cost(predict(X,w), y))


def cv():
    pass

#test()
boston_data = datasets.load_boston()
y = np.mat(boston_data.target).T

#X = np.mat([[1],[2]])
#y = np.mat([[3],[4]])
X = np.mat(boston_data.data)
X_power2 = np.multiply(X,X)
X_power3 = np.multiply(X_power2,X)
X_power4 = np.multiply(X_power3,X)
X0_X = np.multiply(X[:,0], X[:,1:])
X1_X = np.multiply(X[:,1], X[:,2:])
#X = np.hstack((X,X_power2,X_power3,X_power4,X0_X,X1_X))
#X = np.hstack((X,X_power2,X_power3))

X = normlize(X)
X = np.insert(X, 0, 1, axis=1)
np.random.shuffle(X)
m = X.shape[0]

split1 = int(m*0.6)
split2 = int(m*0.8)

train_X = X[:split1]
train_y = y[:split1]

cv_X = X[split1:split2]
cv_y = y[split1:split2]

test_X = X[split2:]
test_y = y[split2:]


w = train(train_X, train_y)
predict_y = predict(cv_X, w)
cost = cost(predict_y, cv_y)

#==============================================================================
# from sklearn import linear_model as lm
# lr = lm.LinearRegression()
# lr.fit(train_X,train_y)
# predict_y_2 = np.mat(lr.predict(cv_X))
# cost2 = cost(predict_y_2, cv_y)
# 
#==============================================================================
