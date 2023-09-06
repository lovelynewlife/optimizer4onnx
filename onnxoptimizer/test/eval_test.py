import unittest

import onnxoptimizer.query.pandas
import pandas as pd


class TestEval(unittest.TestCase):
    def test_my_eval(self):
        a = 1
        b = 2

        res1 = pd.eval("a+b+1")
        res2 = pd.my_eval("a+b+1")

        assert res1 == res2 == 4

    def test4debug(self):
        def say_hello():
            print("Hello world")

        pd.my_eval("say_hello()")


if __name__ == "__main__":
    unittest.main()
