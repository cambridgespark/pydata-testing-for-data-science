import pandas as pd
from src.transformers import CountryTransformer

def test_correct_country_returned_with_simple_df():
    df = pd.DataFrame({'country': ["CA", "GB"]})
    country_transformer = CountryTransformer()

    result_df = country_transformer.transform(df)

    assert len(result_df.index) == 2
    assert result_df["country"][0] == "Canada"
    assert result_df["country"][1] == "UK & Ireland"


# def test_unkown_country_returns_default():
#     df = pd.DataFrame({'country': ["SA"]})
#     country_transformer = CountryTransformer()
#
#     result_df = country_transformer.transform(df)
#
#     # TODO: fix transformer to handle NaN / default
#     assert len(result_df.index) == 1
#     assert result_df["country"][0] == "Other"
