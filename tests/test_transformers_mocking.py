from unittest.mock import MagicMock, call

import pandas as pd
from pandas.util.testing import assert_frame_equal

from src.transformers import CountryFullTransformer


def test_correct_country_returned_with_simple_df():
    df = pd.DataFrame({'country': ["CA", "GB"]})

    country_transformer = CountryFullTransformer()

    country_transformer.getRegionFromCode = MagicMock()
    country_transformer.getRegionFromCode.side_effect = ["Canada", "UK & Ireland"]

    expected_df = pd.DataFrame({'country': ["Canada", "UK & Ireland"]})
    result_df = country_transformer.transform(df)

    country_transformer.getRegionFromCode.assert_has_calls([call("CA"), call("GB")])
    assert_frame_equal(result_df, expected_df)

