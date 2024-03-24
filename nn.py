import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


class NeuralNetwork():
    """
    Simple NN with two hidden layers for classification.
    """
    def __init__(self, layer_neurons, input_size, class_count):
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = self.init_params(layer_neurons, input_size, class_count)
        self.input_size = input_size
        self.class_count = class_count
        self.layer_neurons = layer_neurons

    def init_params(self, layer_neurons, input_size, class_count):
        weights1 = np.random.randn(layer_neurons, input_size) # layer 1
        biases1 = np.random.randn(layer_neurons, 1)
        weights2 = np.random.randn(layer_neurons, layer_neurons) # layer 2
        biases2 = np.random.randn(layer_neurons, 1)
        weights3 = np.random.randn(class_count, layer_neurons) # output layer
        biases3 = np.random.randn(class_count, 1)
        return weights1, biases1, weights2, biases2, weights3, biases3
    
    def feed_forward(self, input):
        input1 = self.w1.dot(input) + self.b1
        activation1 = self.relu(input1)
        input2 = self.w2.dot(activation1) + self.b2
        activation2 = self.relu(input2)
        input3 = self.w3.dot(activation2) + self.b3
        activation3 = self.softmax(input3)
        return input1, activation1, input2, activation2, input3, activation3
    
    def feed_backward(self, input1, activation1, input2, activation2, input3, activation3, output, input_data):
        input_len = len(input_data)
        y = self.one_hot_encode(output)

        d_y = activation3 - y
        d_weights3 = 1 / input_len * d_y.dot(activation2.T)
        d_biases3 = 1 / input_len * np.sum(d_y, axis=1, keepdims=True)

        dz2 = self.w3.T.dot(d_y) * self.drelu(input2)
        d_weights2 = 1 / input_len * dz2.dot(activation1.T)
        d_biases2 = 1 / input_len * np.sum(dz2, axis=1, keepdims=True)

        dz1 = self.w2.T.dot(dz2) * self.drelu(input1)
        d_weights1 = 1 / input_len * dz1.dot(input_data.T)
        d_biases1 = 1 / input_len * np.sum(dz1, axis=1, keepdims=True)

        return d_weights1, d_biases1, d_weights2, d_biases2, d_weights3, d_biases3
    
    def update_weights(self, d_weights1, d_biases1, d_weights2, d_biases2, d_weights3, d_biases3, a):
        self.w1 -= a * d_weights1
        self.b1 -= a * d_biases1
        self.w2 -= a * d_weights2
        self.b2 -= a * d_biases2
        self.w3 -= a * d_weights3
        self.b3 -= a * d_biases3

    def train(self, X, Y, iters, a):

        for i in range(iters):
            input1, activation1, input2, activation2, input3, activation3 = self.feed_forward(X)

            d_weights1, d_biases1, d_weights2, d_biases2, d_weights3, d_biases3 = \
                self.feed_backward(input1, activation1, input2, activation2, input3, activation3, Y, X)
            
            self.update_weights(d_weights1, d_biases1, d_weights2, d_biases2, d_weights3, d_biases3, a)

            # accuracy
            predictions = self.predict(X)
            accuracy = np.mean(predictions == Y)
            print(f'Iter: {i}, Accuracy: {accuracy}')


    def predict(self, X):
        _, _, _, _, _, activation3 = self.feed_forward(X)
        return np.argmax(activation3, axis=0)

    def one_hot_encode(self, y):
        return np.eye(10, dtype=int)[y].T

    def relu(self, input):
        return np.maximum(0, input)
    
    def drelu(self, input):
        return np.where(input > 0, 1, 0)
    
    def softmax(self, input):
        exp_values = np.exp(input - np.max(input, axis=0, keepdims=True))  # Prevent overflow
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # normalize
    X_train = X_train / 255
    X_test = X_test / 255

    # take only 1000 samples
    X_train = X_train[:50000].T
    Y_train = Y_train[:50000]
    X_test = X_test[:10000].T
    Y_test = Y_test[:10000]

    nn = NeuralNetwork(128, X_train.shape[0], 10)

    nn.train(X_train, Y_train, 100, 0.0005)

    predictions = nn.predict(X_test)
    print(predictions.shape)
    accuracy = np.mean(predictions == Y_test)
    print(f'Accuracy: {accuracy}')

    fig, ax = plt.subplots(1, 10)
    for i in range(10):
        ax[i].imshow(X_test[:, i].reshape(28, 28))
        ax[i].set_title(f'{predictions[i]}')
        ax[i].axis('off')
    plt.show()
