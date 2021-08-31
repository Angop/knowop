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


def propagate_forward(layers: List[Layer],
 inpt: Tuple[float, ...]) -> Tuple[float, ...]:
    '''
    Given trained layers, propagate an input forward
    '''
    i = 0
    for layer in layers:
        # print(f"Layer {i}")
        # print(f"initial: {layer.a}")
        inpt = layer.activate(inpt)
        # print(f"layer a: {layer.a}")
        i += 1
    return inpt


def hadamard(arr1: List[float], arr2: List[float]) -> List[float]:
    """
    Performs hadamard multiplication
    """
    res = [0.0] * len(arr1)

    for i in range(len(arr1)):
        res[i] = arr1[i] * arr2[i]
    return res

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
        self.learningRate = 0.1

        self.i_size = i_size
        self.o_size = o_size
        self.layers = self.initLayers()

    def initLayers(self):
        """
        Initializes the layers
        """
        layers = []
        # input layer LITERALLY DONT LITERALLY DEFINE THE INPUT LAYER!!!!!!!!!!
        # layers.append(Layer((self.i_size, 0), False))

        # # one hidden layer
        avgSize = (self.i_size + self.o_size) // 2
        layers.append(Layer((avgSize, self.i_size), False))
        layers.append(Layer((self.o_size, avgSize), True))

        # ONLY ONE LAYER
        # layers.append(Layer((self.o_size, self.i_size), True))

        return layers

    def forwardProp(self, inpt: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Propogates the input through the network and returns the output
        """
        i = 0
        for layer in self.layers:
            # print(f"Layer {i}")
            # print(f"initial: {layer.a}")
            inpt = layer.activate(inpt)
            # print(f"layer a: {layer.a}")
            i += 1
        return inpt
    
    def updateDwns(self, dwn: List[List[float]], x: int):
        '''
        for each list in self.dw add dwn to each coloum in self.dw
        '''
        # For each neuron in layer
        for i in range(len(self.layers[x].dw)):
            # For each weight in neuron
            for j in range(len(self.layers[x].dw[0])):
                self.layers[x].dw[i][j] += dwn[i][0]

    def updateBns(self, dbn: List[float], x: int):
        '''
        for each neuron in self.db add dbn to each coloum in self.db
        '''
        # For each neuron in layer
        for i in range(len(self.layers[x].db)):
            # For each weight in neuron
            self.layers[x].db[i] += dbn[i]

    def updateParams(self):
        '''
        divide each db and dw for each layer by the batchsize
        '''
        # for each layer
        for i in range(len(self.layers)):
            # print(f"total dw: {self.layers[i].dw}")
            # print(f"total db: {self.layers[i].db}")
            # divide each dbn by batch size
            self.layers[i].db =\
                 [db / self.batchSize for db in self.layers[i].db]
            # print(f"final db: {self.layers[i].db}")
            # for each neuron in weight matrix
            for n in range(len(self.layers[i].w)):
                # for each w in neuron
                for w in range(len(self.layers[i].w[0])):
                    # divide dw by batch size
                    self.layers[i].dw[n][w] =\
                         self.layers[i].dw[n][w] / self.batchSize
                    # substract learning rate - dw from w
                    self.layers[i].w[n][w] -=\
                         self.learningRate * self.layers[i].dw[n][w]
            # print(f"final dw: {self.layers[i].dw}")



    def backPropBatch(self, results: List[List[float]]):
        cost = self.getCost(results)
        print(f"COST: {cost} LEARNING RATE: {self.learningRate}")
        weightGrads = []
        biasGrads = []
        for output, expected, inpt in results:
            dwns, dbns = self.backProp(output, expected, inpt)
            weightGrads.append(dwns)
            biasGrads.append(dbns)
            
        # print(f"WEIGHT GRADS: {weightGrads}")
        # print(f"BIAS GRADS: {biasGrads}")
        avgWeights = self.avgWeightArrs(weightGrads)
        avgBiases = self.avgWeightArrs(biasGrads)
        # print(f"AVG WEIGHT GRADS: {avgWeights}")
        # print(f"AVG BIAS GRADS: {avgBiases}")
        
        self.updateParams()
        # self.updateWeights(avgWeights)
        # self.updateBiases(avgBiases)
        
    def backProp(self, output: List[float], expected: List[int],
        inpt: List[int]):
        dwns = []
        dbns = []
        # Initialize da to the derivative of the loss function
        dan = [Math.loss_prime(output[j], expected[j])
                for j in range(len(output))]
        for i in range(len(self.layers) - 1, -1, -1):
            # Get Layer
            l = self.layers[i]            
            # dzn = dan ⊙ gn'(zn)
            gnzn = []
            for n in range(len(l.z)):
                # for each neuron
                gnzn.append(Math.sigmoid_prime(l.z[n]) if l.g is Math.sigmoid
                    else Math.relu_prime(l.z[n])) 
            # print(f"dan: {dan}\n")
            # print(f"gnzn: {gnzn}\n")
            dzn = hadamard(gnzn, dan)
            dzn = [[x] for x in dzn]
            # print(f"dzn: {dzn}\n")
            # dWn = dzn * aTn-1
            if i == 0:
                # print(f"INPUT: {inpt}")
                atn1 = Math.transpose([list(inpt)])
            else:
                atn1 = Math.transpose([self.layers[i - 1].a])
            # print(f"atn1: {atn1}")
            dwn = Math.matmul(dzn, atn1)
            # print(f"dwn: {dwn}")
            # Add to weight matrices list
            dwns.append(dwn)
            dbn = dzn
            # print(f"dbn: {dbn}")
            dbns.append(dbn)
            dbn = [x[0] for x in dzn]
            # dan-1 = WTn * dzn
            wtn = Math.transpose(l.w)
            dan = Math.matmul(wtn, dzn)
            # Turn dan to a 1d list
            dan = [x[0] for x in dan]
            # Store in layer object
            self.updateDwns(dwn, i)
            self.updateBns(dbn, i)
        return dwns, dbns

    def updateWeights(self, gradients: List[float]):
        """
        # TODO works only for 1 layer
        """
        # print(f"dwn: {gradients}")
        for i in range(len(self.layers) - 1, -1, -1):
            # for each layer
            l = self.layers[i]
            # print(f"Before Average: {l.w}")
            # Calulate lr * dan1
            grad = gradients[i]
            lr = self.learningRate
            # print(f"Subs: {subs}")
            for j in range(len(l.w)):
                # for each neuron
                subs = lr * grad[j]
                # print("GRADIENTS: ", grad)
                for k in range(len(l.w[j])):
                    # for each weight in the neuron
                    l.w[j][k] -= subs
            # print(f"After Average: {l.w}")
        

    def updateBiases(self, gradients: List[float]):
        """
        Using the gradient, update the biases
        """
        # print("BIASGRADIENTS: ", gradients)
        # print("BIASES: ", self.layers[0].b)
        # reverse gradients list
        # gradients.reverse()
        # print(f"\nUpdating parameters: {gradients}")
        for i in range(len(self.layers) - 1, -1, -1):
            # for each layer
            l = self.layers[i]
            # Calulate lr * dan1
            grad = gradients[i]
            lr = self.learningRate
            
            # print(f"Subs: {sub}")
            for j in range(len(l.b)):
                dbn = grad[j]
                sub = lr * dbn
                # for each neuron
                # print(f"Weight of neuron: {l.b[j]}")
                # Substract subs from weight of neuron
                l.b[j] -= sub
                # print(f"Updated weight of neuron: {l.b[j]}")

    def updateLRate(self, count: int):
        """
        Update the learning rate given the number of iterations "count"
        """
        baseRate = 0.1
        mult = 0.001
        mini = 1e-5
        lRate = - mult * count + baseRate
        if lRate < mini:
            self.learningRate = mini
        self.learningRate = lRate

    def getCost(self, results: List[List[float]]):
        """
        Returns the cost, which is the average of each loss
        """
        loss = []
        for i in range(len(results)):
            # for one output/expected pair
            # print("OUTPUT: ", results[i][0], "EXPECTED: ", results[i][0])
            for j in range(len(results[i][0])):
                # for each bit in the ouput/expected pair
                loss.append(Math.loss(results[i][0][j], results[i][1][j]))

        return sum(loss) / len(loss)
    
    def print_weights(self):
        print()
        for i in range(len(self.layers)):
            print(f"Layer {i}: ")
            for j in range(len(self.layers[i].w)):
                print(f" Neuron {j + 1} weight:\
 {[round(x, 2) for x in self.layers[i].w[j]]}")
        print()

    def avgWeightArrs(self, arrs):
        """
        Get average of a list of weight arrays
        """
        # print("ARRS: ", arrs)
        avged = []
        for k in range(len(self.layers)):
            # for each layer
            ltemp = []

            for j in range(len(arrs[0][k])):
                # for each element in the list
                ntemp = []

                for i in range(len(arrs)):
                    # for each list
                    ntemp.append(arrs[i][k][j][0])
                    # print(f"AVG: {avged} i:{i} k:{k} j:{j} LTEMP: {ltemp}")
                ltemp.append(sum(ntemp) / len(ntemp))
                # print(f"\nNTEMP: {ntemp} LTEMP: {ltemp}\n")
            avged.append(ltemp)
            # print(f"AVG: {avged} k:{k} j:{j} i:{i}")
        return avged
    
    def train(self, trainSet):
        """
        Trains the model given a training set
        """
        # print("Trainset:", trainSet)
        for count in range(100):
            # Get a random batch of inputs from training set
            batch = [(x, trainSet[x]) for x in random.sample(list(trainSet),
                self.batchSize)]
            # print(f"Batch: {batch}")
            # Run each input through the network
            res = []
            for inpt, expected in batch:
                res.append((self.forwardProp(inpt), expected, inpt))
                # print cost of iteration
            # Backpropagate the error
            self.backPropBatch(res)
            self.updateLRate(count)
        self.print_weights()

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
    return network.layers

def main() -> None:
    random.seed(0)
    # f = lambda x, y: x + y  # operation to learn
    # f = lambda x: ~x
    f = lambda x: x + 100
    # n_args = 2              # arity of operation
    n_args = 1 
    n_bits = 8              # size of each operand

    samples = create_samples(f, n_args, n_bits)
    train_pct = 0.95
    train_set = {inputs: samples[inputs]
               for inputs in random.sample(list(samples),
                                           k=int(len(samples) * train_pct))}
    test_set = {inputs: samples[inputs]
               for inputs in samples if inputs not in train_set}
    print("Train Size:", len(train_set), "Test Size:", len(test_set))
    # print(train_set)

    network = train_network(train_set, n_args * n_bits, n_bits)
    for inputs in test_set:
        output = tuple(round(n, 2) for n in propagate_forward(network, inputs))
        bits = tuple(round(n) for n in output)
        print("OUTPUT:", output)
        print("BITACT:", bits)
        print("BITEXP:", samples[inputs], end="\n\n")

# def avgBiasArrs(arrs: List[List[List[float]]]):
#     """
#     """
#     print("BARRS: ", arrs)
#     avged = []
#     for j in range(len(arrs[0])):
#         # for each element in the list
#         temp = []
#         for i in range(len(arrs)):
#             # for each list
#             temp.append(arrs[i][j][0][0])
#         avged.append(sum(temp) / len(temp))
#     return avged

if __name__ == "__main__":
    main()
