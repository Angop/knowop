# Test cases for knowop assignment
# Angela Kerlin
# Marcelo 

import unittest
from knowop import *


class TestKnowOp(unittest.TestCase):
    
    def test_init(self):
        net = Network(16, 8)

        self.assertEqual(len(net.layers), 3)