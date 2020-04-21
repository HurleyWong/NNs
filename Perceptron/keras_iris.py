# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/iris.csv")

df.head()

inputs_x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
inputs_y = df['variety']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

inputs_x_scaler = scaler.fit_transform(inputs_x.values)
df_scaler = pd.DataFrame(inputs_x_scaler, index=inputs_x.index, columns=inputs_x.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_scaler.values, inputs_y, test_size=0.2, random_state=42)

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=4))
model.add(Dense(3, activation='softmax'))
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

trained_model = model.fit(X_train, y_train, epochs=300, batch_size=32)

test_result = model.evaluate(X_test, y_test, verbose=0)

print("Test accuracy: {}".format(test_result[1]))

y_pred = model.predict(X_test)

df_result = pd.DataFrame.from_dict(trained_model.history)

df_result['accuracy'].plot()

df_result['loss'].plot()