import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

if len(sys.argv) > 2:
    classification = sys.argv[1]
    bool_lr = sys.argv[2]

if len(sys.argv) != 3 or (
        classification != "setosa" and classification != "versicolor" and classification != "virginica"):
    print("\npy perceptron.py <class>\nWhere class can be:\n- setosa (for Setosa Vs non-Setosa classification)\n- "
          "versicolor (for Versicolor Vs non-Versicolor classification)\n- virginica (for Virginica Vs non-Virginica "
          "classification)")
    print("\nWhether use learning rate can be:\n- with (use learning rate, default = 0.01)\n- without (do not use "
          "learning rate)")
    sys.exit(0)
print("\nClassifying ", classification, " Vs non-", classification, sep="")

# set default learning rate or without learning rate
learning_rate = 1
if bool_lr == 'with':
    learning_rate = 0.01
elif bool_lr == 'without':
    learning_rate = 1
else:
    pass

# read data
df = pd.read_csv("iris.csv")
# random sort
df = df.reindex(np.random.permutation(df.index))
# ndarray to store weights
inputs = df.iloc[0:150, [0, 1, 2, 3]].values
# ndarray to store classes name
original_classes = df.iloc[0:150, 4].values
# transfer dataframe to list
df = np.array(df).tolist()
# list to store classes name
classes_name = []
for i in range(len(df)):
    # add class name to list
    classes_name.append(df[i][-1])
    # delete tha last element of weights
    df[i].pop()
    # add weight 1, make weight size to 5
    df[i].append(1)

classes_string = np.where(original_classes == "Iris-" +
                          classification, classification, "non-" + classification)
array_training_samples = np.asarray(df)

training_weights = np.random.normal(size=5, scale=0.5)
num_epochs = 0
num_misclassification = 0
num_converge = 0
accuracy_list = []
index_value = 0
unique_random_list = []
# if the misclassification more than num_stop, stop the loop
num_stop = 50

print(classification)

while num_epochs < 150 and num_misclassification < num_stop:

    is_unique_index = False
    while not is_unique_index:
        index_value = np.random.randint(0, 150)
        if index_value in unique_random_list:
            pass
        else:
            is_unique_index = True
            unique_random_list.append(index_value)

    pre_weights = np.copy(training_weights)

    # calculate the dot product
    dot_product = np.dot(training_weights, array_training_samples[num_epochs])

    print(array_training_samples[num_epochs])

    if classes_name[num_epochs] == 'Iris-' + classification:
        # classify correctly
        if dot_product < 0:
            pass
        else:
            # change the weights without learning rate and increase the misclassification
            training_weights = np.subtract(training_weights, learning_rate * array_training_samples[num_epochs])
            num_misclassification += 1
    elif classes_name[num_epochs] != 'Iris-' + classification:
        # classify correctly
        if dot_product >= 0:
            pass
        else:
            training_weights = np.add(training_weights, learning_rate * array_training_samples[num_epochs])
            num_misclassification += 1
    num_epochs += 1

    if np.array_equal(pre_weights, training_weights):
        num_converge += 1
    elif num_converge < 50:
        num_converge = 0

    accuracy = ((num_epochs - num_misclassification) / float(num_epochs))
    accuracy_list.append(accuracy)

    print("The converge count is {}".format(num_converge))
    print("The weights is {}".format(training_weights))
    print("The number of misclassification is {}".format(num_misclassification))
    print("The accuracy after {} epochs is {:.4%}".format(num_epochs, accuracy))

print("The learning rate is {}".format(learning_rate))
print("The perceptron was stopped after {} epochs".format(num_epochs))

# x = np.arange(0, 150)
# plt.figure()
# plt.plot(x, accuracy_list, "r", linewidth=1)
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# plt.title("Accuracy of {} by erceptron without learning rate".format(classification))
# plt.show()

def predict_class(x, weights):
    if weights.size == 5:
        dot_product = np.dot(x, weights[:4]) + weights[4]
    else:
        dot_product = np.dot(x, weights[:2]) + weights[2]
    if dot_product >= 0:
        return 0
    else:
        return 1

# outputs = np.zeros(150)
# for i in range(150):
#     outputs[i] = predict_class(inputs[i], training_weights)

# misclassified = 0
# for i in range(150):
#     if outputs[i] == 1:
#         myClass = classification
#     else:
#         myClass = "non-" + classification
#
#     if myClass != classes_string[i]:
#         print("Misclassified Point: ", inputs[i], ". Actual class is ", original_classes[i],
#               ", but classified as", myClass)
#         misclassified += 1
#
# if misclassified == 0:
#     print("All points classified correctly!\n")
# else:
#     print("\nTotal misclassified points: ", misclassified, "\n")
