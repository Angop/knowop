# Name:         Angela Kerlin
# Name:         Marcelo Jimenez
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Know Op
# Term:         Summer 2021

import math
import itertools
import random
from typing import Callable, Dict, List, Tuple


class Math:
    """A collection of static methods for mathematical operations."""

    @staticmethod
    def dot(xs: List[float], ys: List[float]) -> float:
        """Return the dot product of the given vectors."""
        return sum(x * y for x, y in zip(xs, ys))

    @staticmethod
    def matmul(xs: List[List[float]],
               ys: List[List[float]]) -> List[List[float]]:
        """Multiply the given matrices and return the resulting matrix."""
        product = []
        for x_row in range(len(xs)):
            row = []
            for y_col in range(len(ys[0])):
                col = [ys[y_row][y_col] for y_row in range(len(ys))]
                row.append(Math.dot(xs[x_row], col))
            product.append(row)
        return product

    @staticmethod
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Return the transposition of the given matrix."""
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    @staticmethod
    def relu(z: float) -> float:
        """
        The activation function for hidden layers.
        """
        return z if z > 0 else 0.01 * z

    @staticmethod
    def relu_prime(z: float) -> float:
        """
        Return the derivative of the ReLU function.
        """
        return 1.0 if z > 0 else 0.0

    @staticmethod
    def sigmoid(z: float) -> float:
        """
        The activation function for the output layer.
        """
        epsilon = 1e-5
        return min(max(1 / (1 + math.e ** -z), epsilon), 1 - epsilon)

    @staticmethod
    def sigmoid_prime(z: float) -> float:
        """
        The activation function for the output layer.
        """
        return Math.sigmoid(z) * (1 - Math.sigmoid(z))

    @staticmethod
    def loss(actual: float, expect: float) -> float:
        """
        Return the loss between the actual and expected values.
        """
        return -(expect * math.log10(actual)
                 + (1 - expect) * math.log10(1 - actual))

    @staticmethod
    def loss_prime(actual: float, expect: float) -> float:
        """
        Return the derivative of the loss.
        """
        return -expect / actual + (1 - expect) / (1 - actual)




class Layer:  # do not modify class

    def __init__(self, size: Tuple[int, int], is_output: bool) -> None:
        """
        Create a network layer with size[0] levels and size[1] inputs at each
        level. If is_output is True, use the sigmoid activation function;
        otherwise, use the ReLU activation function.

        An instance of Layer has the following attributes.

            g: The activation function - sigmoid for the output layer and ReLU
               for the hidden layer(s).
            w: The weight matrix (randomly-initialized), where each inner list
               represents the incoming weights for one neuron in the layer.
            b: The bias vector (zero-initialized), where each value represents
               the bias for one neuron in the layer.
            z: The result of (wx + b) for each neuron in the layer.
            a: The activation g(z) for each neuron in the layer.
           dw: The derivative of the weights with respect to the loss.
           db: The derivative of the bias with respect to the loss.
        """
        self.g = Math.sigmoid if is_output else Math.relu
        self.w: List[List[float]] = \
            [[random.random() * 0.1 for _ in range(size[1])]
             for _ in range(size[0])]
        self.b: List[float] = [0.0] * size[0]

        # use of below attributes is optional but recommended
        self.z: List[float] = [0.0] * size[0]
        self.a: List[float] = [0.0] * size[0]
        self.dw: List[List[float]] = \
            [[0.0 for _ in range(size[1])] for _ in range(size[0])]
        self.db: List[float] = [0.0] * size[0]

    def __repr__(self) -> str:
        """
        Return a string representation of a network layer, with each level of
        the layer on a separate line, formatted as "W | B".
        """
        s = "\n"
        fmt = "{:7.3f}"
        for i in range(len(self.w)):
            s += " ".join(fmt.format(w) for w in self.w[i])
            s += " | " + fmt.format(self.b[i]) + "\n"
        return s

    def activate(self, inputs: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Given an input (x) of the same length as the number of columns in this
        layer's weight matrix, return g(wx + b).
        """
        self.z = [Math.dot(self.w[i], inputs) + self.b[i]
                   for i in range(len(self.w))]
        self.a = [self.g(real) for real in self.z]
        return tuple(self.a)




def create_samples(f: Callable[..., int], n_args: int, n_bits: int,
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Return a dictionary that maps inputs to expected outputs.
    """
    samples = {}
    max_arg = 2 ** n_bits
    for inputs in itertools.product((0, 1), repeat=n_args * n_bits):
        ints = [int("".join(str(bit) for bit in inputs[i:i + n_bits]), 2)
                for i in range(0, len(inputs), n_bits)]
        try:
            output = f(*ints)
            if 0 <= output < max_arg:
                bit_string = ("{:0" + str(n_bits) + "b}").format(output)
                samples[inputs] = tuple(int(bit) for bit in bit_string)
        except ZeroDivisionError:
            pass
    return samples




class Network:

    def __init__(self, i_size: int, o_size: int):
        #Hyperparameters
        self.batchSize = 100
        self.learningRate = 0.9

        self.i_size = i_size
        self.o_size = o_size
        self.layers = self.initLayers()

    def initLayers(self):
        """
        Initializes the layers
        """
        layers = []
        # input layer
        layers.append(Layer((self.i_size, 0), False))

        # one hidden layer
        avgSize = (self.i_size + self.o_size) // 2
        layers.append(Layer((avgSize, self.i_size), False))

        # output layer
        layers.append(Layer((self.o_size, 0), True))

        return layers

    def train(self, trainSet):
        """
        Trains the model given a training set
        """
        for count in range(10000):
            # Get a random batch of inputs from training set
            batch = [(x, trainSet[x]) for x in random.sample(list(trainSet),
                self.batchSize)]
            # Run each input through the network
            res = []
            for input, expected in batch:
                res.append((self.forwardProp(input), expected))
                # print cost of iteration
            # Backpropagate the error
            self.backPropBatch(res)
            self.updateLRate(count)

    def forwardProp(self, input):
        """
        Propogates the input through the input and returns the output
        """
        i = 0
        for layer in self.layers:
            # print(f"Layer {i}")
            # print(f"initial: {layer.a}")
            input = layer.activate(input)
            # print(f"after: {layer.a}")
            i += 1
        return input

    def backPropBatch(self, results):
        cost = self.getCost(results)
        # print(f"COST: {cost} LEARNING RATE: {self.learningRate}")

        weightGrads = []
        biasGrads = []
        for output, expected in results:
            dwns, dbns = self.backProp(output, expected)
            weightGrads.append(dwns)
            biasGrads.append(dbns)
            # print(self.layers[0].a)
        
        avgWeights = avgWeightArrs(weightGrads)
        avgBiases = avgBiasArrs(biasGrads)
        print(f"AVG WEIGHT GRADS: {avgWeights}")
        print(f"AVG BIAS GRADS: {avgBiases}")
# 
        self.updateWeights(avgWeights)
        self.updateBiases(avgBiases)
        
    def backProp(self, output, expected):
        dwns = []
        dbns = []
        # Initialize da to the derivative of the loss function
        dan = [Math.loss_prime(output[j], expected[j])
                for j in range(len(output))]
        for i in range(len(self.layers) - 1, 0, -1):
            # print("\n\nLAYER ", i)
            # Get Layer
            l = self.layers[i]
            
            # dzn = dan ⊙ gn'(zn)
            gnzn = []
        
            for n in range(len(l.z)):
                # for each neuron
                gnzn.append(Math.sigmoid_prime(l.z[n]) if l.g == Math.sigmoid
                    else Math.relu_prime(l.z[n]))
            
            # print(f"dan: {dan}")
            # print(f"gnzn: {gnzn}")
            dzn = Math.dot(gnzn, dan)
            # print(f"dzn: {dzn}")

            # dWn = dzn * aTn-1
            # print(f"Layer {i-1}: {self.layers[i - 1].a}")
            atn1 = Math.transpose([self.layers[i - 1].a])
            # print(f"atn1: {atn1}")
            dwn = Math.matmul([[dzn]], atn1)
            # print(f"dwn: {dwn}")
            # Add to weight matrices list
            dwns.append(dwn)

            # dbn = dzn
            dbn = dzn
            dbns.append(dbn)
             
            # dan-1 = WTn * dzn
            wtn = Math.transpose(l.w)
            dan = Math.matmul(wtn, [[dzn]])
            # Turn dan to a 1d list
            dan = [x[0] for x in dan]

            # Save the gradient
        return dwns, dbns

    def updateWeights(self, gradients):
        """
        """
        # print("updating")
        # print(f"dwn: {gradients}")
        # reverse gradients list
        gradients.reverse()
        # print(f"\nUpdating parameters: {gradients}")
        for i in range(len(self.layers) - 1, 0, -1):
            l = self.layers[i]
            # Calulate lr * dan1
            lr = self.learningRate
            print("GRADIENTS: ", gradients)
            subs = lr * gradients[i - 1]
            print(f"Subs: {subs}")
            for j in range(len(l.w)):
                for k in range(len(l.w[j])):
                    l.w[j][k] -= subs

    def updateBiases(self, gradients):
        """
        """
        # print("BIASGRADIENTS: ", gradients)
        # print("BIASES: ", self.layers[0].b)
        # reverse gradients list
        gradients.reverse()
        # print(f"\nUpdating parameters: {gradients}")
        for i in range(len(self.layers) - 1, 0, -1):
            l = self.layers[i]
            # Calulate lr * dan1
            dbn = gradients[i - 1]
            lr = self.learningRate
            sub = lr * dbn
            # print(f"Subs: {sub}")
            for j in range(len(l.b)):
                # print(f"Weight of neuron: {l.b[j]}")
                # Substract subs from weight of neuron
                l.b[j] -= sub
                # print(f"Updated weight of neuron: {l.b[j]}")

    def updateLRate(self, count):
        """
        """
        # TODO: START LOWER. FROM DANIEL:
        """You probably won't want the learning rate to start above 0.1 - maybe
        even less than that. With a good learning rate decay on each iteration,
        you can have a very low learning rate (e.g. 1e-5) to be the terminating
        condition for training."""
        baseRate = 0.1
        mult = 0.00001
        mini = 1e-5
        lRate =  -mult * count + baseRate
        if lRate < mini:
            self.learningRate = mini
        self.learningRate = lRate

    def getCost(self, results):
        """
        Reurns the cost, which is the average of each loss
        """
        loss = []
        for i in range(len(results)):
            # for one output/expected pair
            # print("OUTPUT: ", results[i][0], "EXPECTED: ", results[i][0])
            for j in range(len(results[i][0])):
                # for each bit in the ouput/expected pair
                loss.append(Math.loss(results[i][0][j], results[i][1][j]))

        return sum(loss) / len(loss)




def train_network(samples: Dict[Tuple[int, ...], Tuple[int, ...]],
                  i_size: int, o_size: int) -> List[Layer]:
    """
    Given a training set (with labels) and the sizes of the input and output
    layers, create and train a network by iteratively propagating inputs
    (forward) and their losses (backward) to update its weights and biases.
    Return the resulting trained network.
    """
    # Create network
    network = Network(i_size, o_size)
    # Train network
    network.train(samples)
    # Return trained network
    return network

def main() -> None:
    random.seed(0)
    f = lambda x, y: x + y  # operation to learn
    n_args = 2              # arity of operation
    n_bits = 8              # size of each operand

    samples = create_samples(f, n_args, n_bits)
    train_pct = 0.95
    train_set = {inputs: samples[inputs]
               for inputs in random.sample(list(samples),
                                           k=int(len(samples) * train_pct))}
    test_set = {inputs: samples[inputs]
               for inputs in samples if inputs not in train_set}
    print("Train Size:", len(train_set), "Test Size:", len(test_set))

    network = train_network(train_set, n_args * n_bits, n_bits)
    # for inputs in test_set:
    #     output = tuple(round(n, 2) for n in propagate_forward(network, inputs))
    #     bits = tuple(round(n) for n in output)
    #     print("OUTPUT:", output)
    #     print("BITACT:", bits)
    #     print("BITEXP:", samples[inputs], end="\n\n")

def avgWeightArrs(arrs):
    """
    """
    avged = []
    for j in range(len(arrs[0])):
        # for each element in the list
        temp = []
        for i in range(len(arrs)):
            # for each list
            temp.append(arrs[i][j][0][0])
        avged.append(sum(temp) / len(temp))
    return avged

def avgBiasArrs(arrs):
    """
    """
    avged = []
    for j in range(len(arrs[0])):
        # for each element in the list
        temp = []
        for i in range(len(arrs)):
            # for each list
            temp.append(arrs[i][j])
        avged.append(sum(temp) / len(temp))
    return avged

if __name__ == "__main__":
    main()
