import os
import logging
import unittest

import random
import numpy as np
from dotenv import load_dotenv

from utils.args import get_args
from utils.utils import seed_anything, initLogging

class UtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        initLogging('test.log')

    def test_logging(self):
        logging.info("Test logging")
        self.assertTrue(os.path.exists('test.log'))

    def test_args(self):
        args, _ = get_args()
        self.assertEqual(args['dataset'], 'MNIST')

    def test_load_pyenv(self):
        load_dotenv()
        self.assertEqual(os.environ['RAY_COLOR_PREFIX'], '1')
        logging.info(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

    def test_seed(self):
        seed_anything(seed=42)
        self.assertEqual(random.randint(0, 100), 81)
        self.assertEqual(np.random.randint(0, 100), 51)

if __name__ == "__main__":
    unittest.main()
