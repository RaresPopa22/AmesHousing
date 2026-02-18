import pandas as pd
import pytest

from src.pipeline.feature_selection import FeatureSelection


class TestFeatureSelection:

    def test_validate_columns(self):
        df = pd.DataFrame({})
        config = {'apply': True, 'values': ['Overall Qual']}
        selector = FeatureSelection(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*FeatureSelection)(?=.*Overall Qual)"):
            res = selector.transform(df)

    def test_transform_happy_path(self, sample_feature_selection):
        data, config = sample_feature_selection
        selector = FeatureSelection(config)

        res = selector.transform(data)

        assert not {'Year Built', 'Full Bath'}.issubset(res)

    def test_pass_through(self, sample_feature_selection):
        data, _ = sample_feature_selection
        config = {'apply': False}
        selector = FeatureSelection(config)
        res = selector.transform(data)

        pd.testing.assert_frame_equal(data, res, check_names=False)