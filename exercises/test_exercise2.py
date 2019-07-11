"""
This exercise is about writing a parameterised unit test using pytest
"""

import pytest
from src.transformers import TimeTransformer

def test_time_transformer(sample_df, expected_df):
    """
    Write a parameterised unit test for TimeTransformer
    :param sample_df: sample df to test with three columns: deadline, created_at, launched_at
    :param expected_df: result with two columns: launched_to_deadline, created_to_launched
    :return:
    """
    pass