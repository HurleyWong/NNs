from perceptron_test import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.axes._axes import _log as matplotlib_axes_logger

df = pd.read_csv("iris.csv")
df = df.reindex(np.random.permutation(df.index))

df = np.array(df)
df = df.tolist()
print(df)
for i in range(len(df)):
    df[i].pop()
    df[i].append(1)
print(df)

X = df.iloc[0:150, [0, 1, 2, 3]].values
y = df.iloc[0:150, 4].values


training_X = df.iloc[0:100, [0, 1, 2, 3]].values
testing_X = df.iloc[101:150, [0, 1, 2, 3]].values

training_y = df.iloc[0:100, 4].values
training_y = np.where(training_y == 'Iris-setosa', -1, 1)

testing_y = df.iloc[101:150, 4].values
testing_y = np.where(testing_y == 'Iris-setosa', -1, 1)

perceptron = Perceptron(iteration_count=100)
perceptron.fit(training_X, training_y)

print(perceptron.w_ ) # final parameters

print("Predicted")
print(perceptron.predict(testing_X))

print("Actual")
print(testing_y)

# def test(X, y, classifier, resolution=0.02):
#     # setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     # plot the decision surface
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 0].max() + 1
#
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#
#     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     # plot class samples
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#                     alpha=0.8, c=cmap(idx), marker=markers[idx],
#                     label=cl)
#
#
# def plot_decision_regions(x, y, classifier, resolution=0.02):
#     marker = ('s', 'x', 'o', 'v')
#     colors = ('blue', 'red', 'green', 'gray', 'cyan')
#     # assign different colors according to the types of different elements in vector y
#     # len(np.unique(y)) is 2 because y = 1 or y = -1
#     print(len(np.unique(y)))
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     # get the min and max values of these two columns
#     x1_min, x1_max = x[:, 1].min() - 1, x[:, 1].max() + 1
#     x2_min, x2_max = x[:, 0].min() - 1, x[:, 0].max() + 1
#
#     print(x1_min)
#     print(x1_max)
#     print(x2_min)
#     print(x2_max)
#
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
#
#     z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     z = z.reshape(xx1.shape)
#     plt.contour(xx1, xx2, z, alpha=0.4, cmap=cmap)
#
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=x[y == cl, 1], y=x[y == cl, 0], alpha=0.8, c=cmap(idx), marker=marker[idx], label=cl)
#
#
# df = pd.read_csv("iris.csv")
#
# # separate the fourth column and assign it to y
# y = df.iloc[0:150, 4].values
#
# # if y == 'Iris-setosa', y = 1; otherwise, y = -1
# y = np.where(y == 'Iris-setosa', -1, 1)
#
# # extract columns 2 and 3 to assign data to x
# x = df.iloc[0:150, [2, 3]].values
#
# plt.scatter(x[:50, 1], x[:50, 0], color='blue', label='setosa')
# plt.scatter(x[50:100, 1], x[50:100, 0], color='green', label='versicolor')
# plt.scatter(x[100:150, 1], x[100:150, 0], color='red', label='virginica')
# plt.xlabel("Petal Width")
# plt.ylabel("Petal Length")
# plt.legend(loc='upper left')
# plt.show()
#
# print(df.head())
#
# b = df.iloc[0:100, 4].values
# b = np.where(b == 'Iris-setosa', -1, 1)
# a = df.iloc[0:100, [0, 2]].values
#
# ppn = Perceptron(eta=0.0000001, n_iter=5, bool_lr=1)
# # ppn.fit(x, y)
# # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# # plt.xlabel('Epochs')
# # plt.ylabel("MisClassification")
# # plt.show()
#
# matplotlib_axes_logger.setLevel('ERROR')
# # plot_decision_regions(x, y, classifier=ppn)
# # plt.xlabel('Petal Width')
# # plt.ylabel('Petal Length')
# # plt.legend(loc='upper left')
# # plt.show()
#
#
# ppn.fit(a, b)
# # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.plot(range(1, len(ppn.error_count_history) + 1), ppn.error_count_history, marker='o')
# plt.xlabel('Epoches')
# plt.ylabel('Number of misclassifications')
# plt.show()
#
#
# accuracy, precision, recall, f1 = ppn.evaluate(a, b)
# print("accuracy: %.2f" % accuracy)
