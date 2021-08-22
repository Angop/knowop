# Test cases for knowop assignment
# Angela Kerlin
# Marcelo Jimenez

import unittest
from knowop import *


class TestKnowOp(unittest.TestCase):
    
    def test_init(self):
        net = Network(16, 8)

        self.assertEqual(len(net.layers), 3)

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
    
    def test_forwardProp(self):
        net = Network(16, 8)
        self.assertEqual(len(net.layers), 3)

        input = (0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0)
        res = net.forwardProp(input)
        print(res)

        self.assertEqual(len(res), 8)
    
    def test_getCost(self):
        # TODO
        pass