import unittest

import numpy as np

import onnxoptimizer.query.pandas
import pandas as pd


class TestEval(unittest.TestCase):
    def test_if_right_env_scope(self):

        a = 1
        b = 2
        res1 = pd.eval("a+b+1")
        res2 = pd.predict_eval("a+b+1")

        assert res1 == res2 == 4

        df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
        res3 = df.predict_eval("abs(age)")
        assert np.all(res3 == np.abs(np.array([10, 20])))

        res4 = df.predict_filter("abs(age) > 10 and abs(age) < 21")
        assert len(res4) == 1 and res4['animal'][1] == 'pig'

    def test4debug(self):
        def say_hello():
            print("Hello world")

        pd.my_eval("say_hello()")


if __name__ == "__main__":
    unittest.main()
