import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_squared_error(x):
    return (1 / 2) * (np.power(x, 2))


input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target_output = np.array([[0, 1, 1, 0]]).reshape(4, 1)
weight_hidden = np.random.rand(2, 4)
weight_output = np.random.rand(4, 1)

lr = 0.05

for epoch in range(200000):
    # Start Hidden layer
    input_hidden = np.dot(input_features, weight_hidden)

    output_hidden = sigmoid(input_hidden)
    # End Hidden layer

    # Start output layer
    input_op = np.dot(output_hidden, weight_output)

    output_op = sigmoid(input_op)
    # End output layer

    error_out = mean_squared_error(output_op - target_output)

    # Derivatives output layer
    derror_douto = output_op - target_output
    douto_dino = sigmoid_der(input_op)
    dino_dwo = output_hidden

    derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)

    # Derivatives hidden layer
    derror_dino = derror_douto * douto_dino
    dino_douth = weight_output
    derror_douth = np.dot(derror_dino, dino_douth.T)
    douth_dinh = sigmoid_der(input_hidden)
    dinh_dwh = input_features
    derror_wh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)

    weight_hidden -= lr * derror_wh
    weight_output -= lr * derror_dwo

print(weight_hidden, weight_output)

predict = sigmoid(np.dot(sigmoid(np.dot([0, 0], weight_hidden)), weight_output))
print(predict)
predict = sigmoid(np.dot(sigmoid(np.dot([1, 0], weight_hidden)), weight_output))
print(predict)
predict = sigmoid(np.dot(sigmoid(np.dot([0, 1], weight_hidden)), weight_output))
print(predict)
predict = sigmoid(np.dot(sigmoid(np.dot([1, 1], weight_hidden)), weight_output))
print(predict)
