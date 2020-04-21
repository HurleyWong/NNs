import pandas as pd

data = pd.read_csv('data/iris.data', header=None,
                   names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'variety'])

data.to_csv('./data/iris.csv', index=False, sep=',')

print(data.shape)

print(data.head(10))