from numpy import *


class NeuralNet(object):
    def __init__(self):
        # Generate random numbers
        random.seed(1)

        # Assign random weights to a 3 x 1 matrix,
        self.synaptic_weights = 2 * random.random((4, 1)) - 1

    # The Sigmoid function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Train the neural network and adjust the weights each time.
    def train(self, inputs, outputs, training_iterations):
        for iteration in range(training_iterations):
            # Pass the training set through the network.
            output = self.learn(inputs)

            # Calculate the error
            error = outputs - output

            # Adjust the weights
            adjustment = dot(inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))





if __name__ == "__main__":
    # Initialize the network
    neural_network = NeuralNet()

    # The training set.
    inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    outputs = array([[1, 0, 1]]).T

    inputs = []
    outputs = []
    for r in range(10):
        import operator
        import functools

        line = [int(floor(random.random() * 1.5)) for _ in range(4)]
        c = functools.reduce(operator.ior, line)

        inputs.append(line)
        outputs.append((c,))

    inputs = array(inputs)
    outputs = array(outputs)

    print(inputs)
    print(outputs)


    # Train the network
    neural_network.train(inputs, outputs, 1000000)

    # Test the neural network with a test example.
    print(neural_network.learn(array([0, 0, 0, 0])))