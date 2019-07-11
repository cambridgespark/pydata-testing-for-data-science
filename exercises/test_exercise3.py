"""
This exercise is about refactoring a unit test to improve it's readability and maintenance
"""
import pandas as pd
from src.transformers import CountryTransformer

import pytest
def test_correct_country_returned_with_simple_df():
    """
    Refactor this unit test to apply the Given/When/Then pattern
    :return:
    """
    df = pd.DataFrame({'country': ["CA", "GB"]})
    country_transformer = CountryTransformer()
    assert len(country_transformer.transform(df).index) ==  2
    assert country_transformer.transform(df)["country"][0] == "Canada"
    assert country_transformer.transform(df)["country"][1] == "UK & Ireland"