import unittest
import pandas as pd
from src.transformers import CountryTransformer


class TestCountryTransformer(unittest.TestCase):

    def test_correct_country_returned_with_simple_df(self):
        df = pd.DataFrame({'country':["CA", "GB"]})
        country_transformer = CountryTransformer()

        result_df = country_transformer.transform(df)

        self.assertEqual(len(result_df.index), 2)
        self.assertEqual(result_df["country"][0], "Canada")
        self.assertEqual(result_df["country"][1], "UK & Ireland")


    # def test_unkown_country_returns_default(self):
    #     df = pd.DataFrame({'country':["BE"]})
    #     country_transformer = CountryTransformer()
    #
    #     result_df = country_transformer.transform(df)
    #
    #     # TODO: fix transformer to handle NaN / default
    #     self.assertEqual(len(result_df.index), 1)
    #     self.assertEqual(result_df["country"][0], "SA")

