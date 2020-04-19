import numpy as np
import pandas as pd
import random


def update_weights(inputs, weights, classes, learning_rate):
    dot_product = np.dot(inputs, weights[:4]) + weights[4]

    if classes[i] == 0:
        if dot_product >= 0:
            weights[:4] = np.subtract(weights[:4], learning_rate * inputs)
            # bias weight
            weights[4] = np.subtract(weights[4], learning_rate)

    else:
        if dot_product < 0:
            weights[:4] = np.add(weights[:4], learning_rate * inputs)
            # bias weight
            weights[:4] = np.add(weights[:4], learning_rate)

    return weights


def predict(inputs, weights):
    if weights.size == 5:  # hidden layer
        dot_product = np.dot(inputs, weights[:4]) + weights[4]
    else:  # output layer
        dot_product = np.dot(inputs, weights[:2]) + weights[2]
    if dot_product >= 0:
        return 1
    else:
        return 0


def classify(classes_name, setosa_outputs, virginica_outputs, versicolor_outputs):
    misclassified = 0
    for i in range(0, 150):
        if versicolor_outputs[i] == 1 and setosa_outputs[i] == 0 and \
                virginica_outputs[
                    i] == 0 and classes_name[i] != "Iris-versicolor":
            error_list.append(i)
            misclassified += 1
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 1 and \
                virginica_outputs[i] == 0 and classes_name[i] != "Iris-setosa":
            error_list.append(i)
            misclassified += 1
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 0 and \
                virginica_outputs[i] == 1 and classes_name[i] != "Iris-virginica":
            error_list.append(i)
            misclassified += 1

    if misclassified == 0:
        print("All points classified correctly!")
    else:
        print("\nTotal misclassified points: ", misclassified)

    accuracy = (150 - misclassified) / 150 * 100
    print("Accuracy: ", accuracy, "%\n")


def error_prediction(classes_name, setosa_outputs, virginica_outputs, versicolor_outputs):
    print("Error Prediction of the Iris dataset:\n")
    print("Data Point\tActual Class\t\tPredicted Class")
    # for i in range(0, 150):
    #
    #     print(i, classes_name[i], sep="\t\t\t", end="\t\t\t")
    #     if versicolor_outputs[i] == 1 and setosa_outputs[i] == 0 and virginica_outputs[i] == 0:
    #         print("Iris-versicolor")
    #     elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 1 and virginica_outputs[i] == 0:
    #         print("Iris-setosa")
    #     elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 0 and virginica_outputs[i] == 1:
    #         print("Iris-virginica")

    for i in error_list:
        print(i, classes_name[i], sep="\t\t\t", end="\t\t\t")
        if versicolor_outputs[i] == 1 and setosa_outputs[i] == 0 and virginica_outputs[i] == 0:
            print("Iris-versicolor")
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 1 and virginica_outputs[i] == 0:
            print("Iris-setosa")
        elif versicolor_outputs[i] == 0 and setosa_outputs[i] == 0 and virginica_outputs[i] == 1:
            print("Iris-virginica")


df = pd.read_csv("iris.data", header=None)
inputs = df.iloc[0:150, [0, 1, 2, 3]].values
classes_name = df.iloc[0:150, 4].values
# error classification list
error_list = []

"""
Hidden Layer of setosa
"""
# setosa vs. non-setosa
classes_setosa = np.where(classes_name == "Iris-setosa", 1, 0)
# the output of setasa in hidden layer
hidden_setosa_outputs = np.zeros(150)

# random weights
hidden_setosa_weights = np.random.rand(5)

epochs = 0
max_epochs = 1500
num_converge = 0
max_converge = 300

# training
while num_converge < max_converge and epochs < max_epochs:

    i = random.randint(0, 149)
    epochs += 1
    prev_hidden_setosa_weights = np.copy(hidden_setosa_weights)

    hidden_setosa_weights = update_weights(inputs=inputs[i], weights=hidden_setosa_weights, classes=classes_setosa,
                                           learning_rate=1)

    if np.array_equal(hidden_setosa_weights, prev_hidden_setosa_weights):
        num_converge += 1
    else:
        num_converge = 0

# predict
for i in range(0, 150):
    hidden_setosa_outputs[i] = predict(inputs[i], hidden_setosa_weights)

"""
Hidden Layer of virginica
"""
# virginica vs. non-virginica
classes_virginica = np.where(classes_name == "Iris-virginica", 1, 0)
# the output of setasa in hidden layer
hidden_virginica_outputs = np.zeros(150)
# random weights
hidden_virginica_weights = np.random.rand(5)

epochs = 0
max_epochs = 15000
num_converge = 0
max_converge = 1500

# training
while num_converge < max_converge and epochs < max_epochs:

    i = random.randint(0, 149)
    epochs += 1
    prev_hidden_virginica_weights = np.copy(hidden_virginica_weights)

    hidden_virginica_weights = update_weights(inputs=inputs[i], weights=hidden_virginica_weights,
                                              classes=classes_virginica,
                                              learning_rate=0.01)

    if np.array_equal(hidden_virginica_weights, prev_hidden_virginica_weights):
        num_converge += 1
    else:
        num_converge = 0

# predict
for i in range(0, 150):
    hidden_virginica_outputs[i] = predict(
        inputs[i], hidden_virginica_weights)

"""
Output Layer: setosa
"""
# assign weights to output layer of setosa
output_setosa_weights = np.array([1, 0, -0.5])
output_setosa_outputs = np.zeros(150)

# predict
for i in range(0, 150):
    x = np.array([hidden_setosa_outputs[i], hidden_virginica_outputs[i]])
    output_setosa_outputs[i] = predict(x, output_setosa_weights)

"""
Output Layer: virginica
"""
# assign weights to output layer of virginica
output_virginica_weights = np.array([0, 1, -0.5])
output_virginica_outputs = np.zeros(150)

# predict
for i in range(0, 150):
    x = np.array([hidden_setosa_outputs[i], hidden_virginica_outputs[i]])
    output_virginica_outputs[i] = predict(x, output_virginica_weights)

"""
Output Layer: versicolor
"""
# assign weights to output layer of versicolor
output_versicolor_weights = np.array([-1, -1, 0.5])
output_versicolor_outputs = np.zeros(150)

# predict
for i in range(0, 150):
    x = np.array([hidden_setosa_outputs[i], hidden_virginica_outputs[i]])
    output_versicolor_outputs[i] = predict(x, output_versicolor_weights)

"""
Accuracy
"""
classify(classes_name, output_setosa_outputs,
         output_virginica_outputs, output_versicolor_outputs)

print("Error Prediction List: {}\n".format(error_list))

error_prediction(classes_name, output_setosa_outputs,
                 output_virginica_outputs, output_versicolor_outputs)
