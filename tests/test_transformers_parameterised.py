import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.transformers import GoalAdjustor


test_goal_transformer_testdata = [
    (pd.DataFrame({'goal': [5], 'static_usd_rate': [2]}), pd.DataFrame({'adjusted_goal': [10]})),
    (pd.DataFrame({'goal': [0], 'static_usd_rate': [1]}), pd.DataFrame({'adjusted_goal': [0]})),
    (pd.DataFrame({'goal': [0], 'static_usd_rate': [1]}), pd.DataFrame({'adjusted_goal': [0]})),
]

@pytest.mark.parametrize("sample_df, expected_df", test_goal_transformer_testdata)
def test_goal_adjustor(sample_df, expected_df):
    adjustor = GoalAdjustor()

    result_df = adjustor.transform(sample_df)
    assert_frame_equal(result_df, expected_df)

