import pandas as pd
from pandas.testing import assert_frame_equal
from src.transformers import GoalAdjustor, TimeTransformer


def test_time_transformer():
    time_transformer = TimeTransformer()
    deadline_timestamp = 1459283229
    created_at_timestamp = 1455845363
    launched_at_timestamp = 1456694829
    sample_df = pd.DataFrame({'deadline': [deadline_timestamp], 'created_at': [created_at_timestamp], 'launched_at': [
        launched_at_timestamp]})

    expected_df = pd.DataFrame({'launched_to_deadline': [29], 'created_to_launched': [9]})

    result_df = time_transformer.transform(sample_df)

    assert_frame_equal(result_df, expected_df)

def test_goal_adjustor_with_value():
    adjustor = GoalAdjustor()
    goal_value = 10
    usd_rate_value = 2
    sample_df = pd.DataFrame({'goal': [goal_value], 'static_usd_rate': [usd_rate_value]})

    result_df = adjustor.transform(sample_df)

    expected_adjusted_goal_value = 20
    expected_df = pd.DataFrame({'adjusted_goal': [expected_adjusted_goal_value]})
    assert_frame_equal(result_df, expected_df)

    #TODO: show problem if we just use assert from py test. the diagnostics makes no sense
    # have to use assert_frame_equal


