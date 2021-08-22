# Test cases for knowop assignment
# Angela Kerlin
# Marcelo Jimenez

import unittest
from knowop import *


class TestKnowOp(unittest.TestCase):
    
    def test_init(self):
        net = Network(16, 8)

        self.assertEqual(len(net.layers), 3)

    def test_init_bad(self):
        with self.assertRaises(ValueError):
            net = Network(16, 8, 1)
    
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

        batch = random.sample(train_set, self.batchSize)
        print(batch)