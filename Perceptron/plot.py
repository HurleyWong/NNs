import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv('iris.csv')

# first flower
iris_setosa = iris.iloc[:50]
# second flower
iris_versicolor = iris.iloc[50:100]
# third flower
iris_virginica = iris.iloc[100:150]


size = 5
setosa_color = 'b'
versicolor_color = 'g'
virginica_color = 'r'

sepal_width_ticks = np.arange(2, 5, step=0.5)
sepal_length_ticks = np.arange(4, 8, step=0.5)
petal_width_ticks = np.arange(0, 2.5, step=0.5)
petal_length_ticks = np.arange(1, 7, step=1)

ticks = [sepal_length_ticks, sepal_width_ticks, petal_length_ticks, petal_width_ticks]
label_text = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

plt.figure(figsize=(12, 12))

# plt.scatter(x=iris_setosa.iloc[:, 3], y=iris_setosa.iloc[:, 2], color=setosa_color, s=size)
# plt.scatter(x=iris_versicolor.iloc[:, 3], y=iris_versicolor.iloc[:, 2], color=versicolor_color, s=size)
# plt.scatter(x=iris_virginica.iloc[:, 3], y=iris_virginica.iloc[:, 2], color=virginica_color, s=size)
# plt.title('{} vs {}'.format(label_text[2], label_text[3]))
# plt.xlabel(label_text[3])
# plt.ylabel(label_text[2])
# plt.xticks(ticks[3])
# plt.yticks(ticks[2])

for i in range(0, 4):
    for j in range(0, 4):
        plt.tight_layout()
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        if i == j:
            plt.xticks([])
            plt.yticks([])
            plt.text(0.1, 0.4, label_text[i], size=18)
        else:
            plt.scatter(x=iris_setosa.iloc[:, j], y=iris_setosa.iloc[:, i], color=setosa_color, s=size, label='setosa')
            plt.scatter(x=iris_versicolor.iloc[:, j], y=iris_versicolor.iloc[:, i], color=versicolor_color, s=size, label='versicolor')
            plt.scatter(x=iris_virginica.iloc[:, j], y=iris_virginica.iloc[:, i], color=virginica_color, s=size, label='virginica')
            plt.title('{} vs {}'.format(label_text[i], label_text[j]))
            plt.xlabel(label_text[j])
            plt.ylabel(label_text[i])
            plt.xticks(ticks[j])
            plt.yticks(ticks[i])

plt.savefig('iris.png', format='png')
