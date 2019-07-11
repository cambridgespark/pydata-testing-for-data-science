import json

from hypothesis import given
from hypothesis.extra.pandas import column, data_frames
import pandas as pd
from pandas.util.testing import assert_frame_equal

from src.transformers import GoalAdjustor


@given(data_frames([column('goal', dtype=float), column('static_usd_rate', dtype=float)]))
def test_goal_adjustor(sample_df):
    adjustor = GoalAdjustor()

    result_df = adjustor.transform(sample_df)

    assert len(sample_df.index) == len(result_df.index)


    # example of invariant: raises the question of where should validation be
    #expected_df = pd.DataFrame({'adjusted_goal': sample_df["goal"] * sample_df["static_usd_rate"]})
    #assert (expected_df["adjusted_goal"] >= 0).all()