import numpy as np
import pandas as pd
import sys

hidden_virginica_weights = np.random.rand(5)
print(hidden_virginica_weights)

def update_weights(training_samples, weights, classes, learning_rate):
    """

    :param training_samples:
    :param weights: weight[:4], four features represent weights
    :param classes: e.g. setosa or non-setosa, 0 means setosa, 1 means non-setosa
    :param learning_rate: use 1 represent 'without learning rate'
    :return:
    """
    # weights[4] means bias
    dot_product = np.dot(training_samples, weights[:4]) + weights[4]

    # setosa
    if classes[i] == 0:
        # classify incorrectly
        if dot_product >= 0:
            # update weights
            weights[:4] -= learning_rate * training_samples
        else:
            # classify correctly
            pass
    # non-setosa
    else:
        # classify correctly
        if dot_product >= 0:
            pass
        else:
            # update weights
            weights[:4] += learning_rate * training_samples

    return weights

















#
#
# class Perceptron(object):
#     """
#     n_iter: iteration steps
#     """
#     def __init__(self, eta, n_iter, bool_lr):
#         self.eta = eta
#         self.n_iter = n_iter
#         self.bool_lr = bool_lr
#         self.w = None
#         self.b = 0
#         self.error_count_history = []
#
#     def fit(self, x, y):
#         """
#         fit training data
#         :param x:
#         :param y: sample classification
#         :return:
#         """
#         # self.w_ = np.zeros(1 + x.shape[1])
#         # self.errors_ = []
#         #
#         # for _ in range(self.n_iter):
#         #     errors = 0
#         #     for x_col, y_col in zip(x, y):
#         #         if self.bool_lr == 1:
#         #             update = self.eta * (y_col - self.predict(x_col))
#         #         else:
#         #             update = y_col - self.predict(x_col)
#         #         self.w_[1:] += update * x_col
#         #         self.w_[0] += update
#         #         print(self.w_[1:])
#         #         errors += int(update != 0.0)
#         #     self.errors_.append(errors)
#         # return self
#
#         self.w, self.b = np.zeros(x.shape[1]), 0
#         for _ in range(self.n_iter):
#             error_count = 0
#             # 随机梯度下降法
#             for xi, yi in zip(x, y):
#                 if yi * self.predict(xi) <= 0:
#                     if self.bool_lr == 1:
#                         self.w += self.eta * yi * xi
#                         self.b += self.eta * yi
#                     else:
#                         self.w += yi * xi
#                         self.b += yi
#                     error_count += 1
#             self.error_count_history.append(error_count)
#             print(self.w, self.b)
#             # if error_count == 0:
#             #     break
#
#     def net_input(self, x):
#         """
#         z = w0 * 1 + w1 * x1 + ... + wn * xn
#         :param x:
#         :return:
#         """
#         # return np.dot(x, self.w_[1:]) + self.w_[0]
#         return np.dot(x, self.w) + self.b
#
#     def predict(self, x):
#         """
#         if z >= 0, output = 1; if z < 0, output = -1
#         :param x:
#         :return:
#         """
#         # return np.where(self.net_input(x) >= 0.0, 1, -1)
#         return np.sign(self.net_input(x))
#
#     def evaluate(self, x_test, y_test):
#         TP, FN, FP, TN = 0, 0, 0, 0
#         for xi, yi in zip(x_test, y_test):
#             y_ = self.predict(xi)
#             if y_ == 1 and yi == 1:
#                 TP += 1
#             if y_ == 1 and yi == -1:
#                 FP += 1
#             if y_ == -1 and yi == 1:
#                 FN += 1
#             if y_ == -1 and yi == -1:
#                 TN += 1
#
#         accuracy = (TP + TN) / (TP + FN + FP + TN)
#         precision = TP / (TP + FP)
#         recall = TP / (TP + FN)
#         f1 = 2 * TP / (2 * TP + FP + FN)
#         return accuracy, precision, recall, f1
#
#

# class Perceptron(object):
#     def __init__(self, eta=0.01, iteration_count=10):
#         self.eta = eta
#         self.iteration_count = iteration_count
#
#     def fit(self, X, y):
#         self.w_ = np.zeros(1 + X.shape[1])
#         self.errors_ = []
#
#         for _ in range(self.iteration_count):
#             errors = 0
#             for xi, target in zip(X, y):
#                 update = self.eta * (target - self.predict(xi))
#                 self.w_[1:] += update * xi
#                 self.w_[0] += update
#                 errors += int(update != 0.0)
#             self.errors_.append(errors)
#         return self
#
#     def net_input(self, X):
#         return np.dot(X, self.w_[1:]) + self.w_[0]
#
#     def predict(self, X):
#         return np.where(self.net_input(X) >= 0.0, 1, -1)
