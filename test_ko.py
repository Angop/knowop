# Test cases for knowop assignment
# Angela Kerlin
# Marcelo Jimenez

import unittest
from knowop import *


class TestKnowOp(unittest.TestCase):
    
    def test_init(self):
        net = Network(16, 8)

        # self.assertEqual(len(net.layers), 3)

    def test_understanding(self):
        random.seed(0)
        f = lambda x, y: x + y  # operation to learn
        n_args = 2              # arity of operation
        n_bits = 8              # size of each operand
        net = Network(16, 8)
        samples = create_samples(f, n_args, n_bits)
        train_pct = 0.95
        train_set = {inputs: samples[inputs]
               for inputs in random.sample(list(samples),
                                           k=int(len(samples) * train_pct))}

        # print(train_set)
        batch = [(x, train_set[x]) for x in random.sample(list(train_set), net.batchSize)]
        # print(batch)

        # i = 0
        # for layer in net.layers:
        #     print(f"WEIGHTS of layer {i}: ", layer.w)
        #     print("\n")
        #     i += 1

    
    # def test_forwardProp(self):
    #     net = Network(16, 8)
    #     self.assertEqual(len(net.layers), 3)

    #     input = (0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0)
    #     res = net.forwardProp(input)
    #     # print(res)

    #     self.assertEqual(len(res), 8)
    
    def test_layer_shapes(self):
        net = Network(16, 8)

        # print(net.layers[0])

    # def test_backProp(self):
    #     net = Network(16, 8)

    #     f = lambda x, y: x + y  # operation to learn
    #     n_args = 2              # arity of operation
    #     n_bits = 8              # size of each operand
    #     samples = create_samples(f, n_args, n_bits)
    #     train_pct = 0.95
    #     train_set = {inputs: samples[inputs]
    #                for inputs in random.sample(list(samples),
    #                                            k=int(len(samples) * train_pct))}
    #     net.train(train_set)
    
    # def test_getCost(self):
    #     # TODO
    #     pass



    # def test_full_addition(self):
    #     random.seed(0)
    #     f = lambda x, y: x + y  # operation to learn
    #     n_args = 2              # arity of operation
    #     n_bits = 8              # size of each operand

    #     samples = create_samples(f, n_args, n_bits)
    #     train_pct = 0.95
    #     train_set = {inputs: samples[inputs]
    #                for inputs in random.sample(list(samples),
    #                                            k=int(len(samples) * train_pct))}
    #     test_set = {inputs: samples[inputs]
    #                for inputs in samples if inputs not in train_set}
    #     # print("Train Size:", len(train_set), "Test Size:", len(test_set))

    #     network = train_network(train_set, n_args * n_bits, n_bits)

    # def test_full_identity_two_input(self):
    #     # given x, y just returns x
    #     random.seed(0)
    #     f = lambda x, y: x      # operation to learn
    #     n_args = 2              # arity of operation
    #     n_bits = 8              # size of each operand

    #     samples = create_samples(f, n_args, n_bits)
    #     train_pct = 0.95
    #     train_set = {inputs: samples[inputs]
    #                for inputs in random.sample(list(samples),
    #                                            k=int(len(samples) * train_pct))}
    #     test_set = {inputs: samples[inputs]
    #                for inputs in samples if inputs not in train_set}
    #     # print("Train Size:", len(train_set), "Test Size:", len(test_set))

    #     network = train_network(train_set, n_args * n_bits, n_bits)

    def test_full_identity(self):
        # given x just returns x
        random.seed(0)
        f = lambda x: x  # operation to learn
        n_args = 1              # arity of operation
        n_bits = 8              # size of each operand

        samples = create_samples(f, n_args, n_bits)
        train_pct = 0.95
        train_set = {inputs: samples[inputs]
                   for inputs in random.sample(list(samples),
                                               k=int(len(samples) * train_pct))}
        test_set = {inputs: samples[inputs]
                   for inputs in samples if inputs not in train_set}
        # print("Train Size:", len(train_set), "Test Size:", len(test_set))

        network = train_network(train_set, n_args * n_bits, n_bits)

        # test the training
        input, expected = (0, 0, 0, 1, 1, 0, 1, 1), (0, 0, 0, 1, 1, 0, 1, 1)
        output = network.forwardProp(input)

        print(f"ACTUAL: {output}")
        print(f"INPUT: {input}")
        print(f"EXPECTED: {expected}")
        self.assertEqual(tuple([round(x) for x in output]), expected)

    def test_avgarrs(self):
        test = [[[[0]], [[3]]], [[[1]], [[6]]], [[[2]], [[9]]]]

        self.assertEqual(avgWeightArrs(test), [1, 6])