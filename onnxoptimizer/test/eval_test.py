import unittest

import numpy as np

import onnxoptimizer.query.pandas
import pandas as pd

from onnxoptimizer.query.onnx_eval.model_context import ModelContext


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
        def say_hello(a):
            print("Hello world")
            return a

        df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
        res = df.predict_eval('''new=@say_hello(a=age)
        new2=@say_hello(a=age)''', engine='python')
        print(res)

    def test_predict_end2end(self):
        path1 = "/home/uw1/snippets/py_onnx/expedia/data/S_listings.csv"
        path2 = "/home/uw1/snippets/py_onnx/expedia/data/R1_hotels.csv"
        path3 = "/home/uw1/snippets/py_onnx/expedia/data/R2_searches.csv"
        # 读取csv表
        S_listings = pd.read_csv(path1)
        R1_hotels = pd.read_csv(path2)
        R2_searches = pd.read_csv(path3)
        # 连接3张表
        data = pd.merge(pd.merge(S_listings, R1_hotels, how='inner'), R2_searches, how='inner')
        # print(data.isnull().any())    #检测缺失值
        data.dropna(inplace=True)  # 删除NaNret

        # 获取分类label
        y = np.array(data.loc[:, 'promotion_flag'])
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

        df = data.loc[:, numerical_columns + categorical_columns]

        input_columns = numerical_columns + categorical_columns

        batch = df.iloc[: 4096, :]

        mc = ModelContext("/home/uw1/snippets/py_onnx/expedia/expedia.onnx")

        args_string = ','.join([
            f'{elem}={elem}'
            for elem in input_columns
        ])

        eval_str = f'''result=@mc({args_string})
        result2=@mc({args_string})
        '''
        print(eval_str)

        new_df = batch.predict_eval(eval_str, engine='python')

        print(new_df)


if __name__ == "__main__":
    unittest.main()
