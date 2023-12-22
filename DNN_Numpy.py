import numpy as np


class NeuralNetwork:
    def __init__(self, layers: tuple) -> None:
        self.weights = [np.ones((layers[i-1], layers[i])) for i in range(1, len(layers))]
        self.b = [np.ones(x) for x in layers[1:]]
        self.layers = layers

    def prepare(self, x):
        if type(x) == list or type(x) == tuple:
            x = np.array(x)
        elif type(x) == np.ndarray:
            pass
        else:
            x = np.array([x])
        return x


######## Overridable functions ##########
    def activation(self, x):
        if x<=0:
            return 0
        else:
            return 1

    def gradient(self, inputs, outputs, alpha, target=None):
        for i in range(len(outputs)-1, -1, -1):
            a = inputs[i]
            while len(a)<len(self.weights[i].flatten()):
                a = np.concatenate((a, inputs[i]))
            _1, _2 = self.weights[i].shape
            a = a.reshape(_2, _1).transpose()
            d_w = np.divide(np.multiply(target-outputs[i], a), self.weights[i])
            #w = self.weights[i] + d_w
            self.weights[i] += alpha*d_w
            self.b[i] += target-outputs[i]
            target = np.array(list(map(self.activation, np.dot(self.weights[i], target))))
        return self.weights

#########################################

    def feed_forward(self, input):
        input = self.prepare(input)
        for i in range(len(self.layers)-1):
            input = np.dot(input, self.weights[i]) + self.b[i]
            input = np.array(list(map(self.activation, input)))
        return input
    
    def back_propagation(self, input, output, alpha):
        outputs = []
        inputs = [input]
        for i in range(len(self.layers)-1):
            input = np.dot(input, self.weights[i]) + self.b[i]
            input = np.array(list(map(self.activation, input)))
            outputs.append(input)
        inputs += outputs[:-1]
        self.gradient(inputs, outputs, alpha, output)


    def train(self, inputs, outputs, alpha, epochs, verbose=True):
        for j in range(epochs):
            if verbose:
                print(f"epoch ====== {j}")
            for i in range(len(outputs)):
                x = inputs[i]
                res = outputs[i]
                self.back_propagation(x, res, alpha)

    def predict(self, x):
        y = []
        for i in x:
            y.append(self.feed_forward(i))
            #print(f"{i}:\n\t {y[-1]}")
        return np.array(y)






if __name__ == "__main__":
    #AND Gate implementation

    from sklearn.metrics import accuracy_score
    
    nn = NeuralNetwork((3, 10, 4, 1))
    input = np.array([[0, 1, 0], [0 ,0, 1], [1,1, 1], [1, 0, 1]])
    output = np.array([[0], [0], [1], [0]])
    
    test_x = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 0]])
    test_y = np.array([[0], [0], [1], [0], [0], [0], [0], [0]])

    nn.train(input, output, 0.01, 2000)

    pred_y = nn.predict(test_x)

    print("Accuracy:", accuracy_score(test_y, pred_y))