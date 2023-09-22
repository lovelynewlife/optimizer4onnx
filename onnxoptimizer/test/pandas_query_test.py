import unittest
from typing import Callable

import numpy as np
import pandas as pd

from onnxoptimizer.query.pandas.api import model_udf

DATA_DIR = "/home/uw1/MLquery/reference/snippets/py_onnx/expedia"


class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path1 = f"{DATA_DIR}/data/S_listings.csv"
        path2 = f"{DATA_DIR}/data/R1_hotels.csv"
        path3 = f"{DATA_DIR}/data/R2_searches.csv"

        S_listings = pd.read_csv(path1)
        R1_hotels = pd.read_csv(path2)
        R2_searches = pd.read_csv(path3)
        # join three tables
        data = pd.merge(pd.merge(S_listings, R1_hotels, how='inner'), R2_searches, how='inner')

        data.dropna(inplace=True)

        # 8 numerical, 20 categorical
        numerical_columns = ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                             'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']
        categorical_columns = ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                               'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id',
                               'visitor_location_country_id',
                               'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
                               'srch_adults_count',
                               'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                               'random_bool']

        cls.input_columns = numerical_columns + categorical_columns

        cls.df = data.loc[:, numerical_columns + categorical_columns]

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

    def test_model_udf_wrapper(self):
        def model_func_test(path: str):
            def model_eval(func: Callable[[...], dict]):
                def mc(**kwargs):
                    print(path)
                    ret_str = ""
                    for k, v in kwargs.items():
                        ret_str += f"{k}:{v}"

                    return ret_str

                def wrapper(*args, **kwargs):
                    input_map = func(*args, **kwargs)
                    return mc(**input_map)

                return wrapper

            return model_eval

        @model_func_test("/here/for/test")
        def a_model_function(a, b, c):
            return {
                "a": a,
                "b": b,
                "c": c
            }

        res = a_model_function(1, 2, 3)

        assert res == f"a:1b:2c:3"

    def test4debug(self):
        def say_hello(a):
            print("Hello world")
            return a

        df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
        res = df.predict_eval('''new=@say_hello(a=age)
        new2=@say_hello(a=age)''', engine='python')
        b = 3
        # pd.predict_eval("b + 1 > 1")
        df.predict_filter("sin(age) > 15")
        print(res)

    def test_predict_end2end(self):
        batch = self.df.iloc[: 4096, :]

        @model_udf(f"{DATA_DIR}/expedia_lr.onnx")
        def expedia_infer(infer_df):
            return infer_df.to_dict(orient="series")

        new_df = batch.predict_eval('''result=@expedia_infer(@batch)
                                    result2=@expedia_infer(@batch)''')

        print(new_df)

        assert np.all(new_df["result"] == new_df["result2"])

    def test_predict_filter_eval(self):
        batch = self.df.iloc[: 120000, :]

        @model_udf(f"{DATA_DIR}/expedia_lr.onnx")
        def expedia_infer(infer_df):
            return infer_df.to_dict(orient="series")

        new_df = batch.predict_filter("@expedia_infer(@batch)==0 or @expedia_infer(@batch)==1")
        assert np.all(new_df["prop_location_score1"] == batch["prop_location_score1"])
        new_df = batch.predict_filter("not @expedia_infer(@batch)==3")
        assert np.all(new_df["prop_location_score1"] == batch["prop_location_score1"])
        print(new_df)


if __name__ == "__main__":
    unittest.main()
