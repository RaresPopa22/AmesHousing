import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.pipeline.feature_engineer import FeatureEngineer


class TestFeatureEngineer:

    def test_validate_columns(self):
        data = pd.DataFrame({})
        config = {
            'surface': {
                'op': 'sum',
                'new_name': 'Total SF',
                'old': ['Outlier']
            }
        }
        handler = FeatureEngineer(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*FeatureEngineer)(?=.*Outlier)"):
            res = handler.transform(data)

    def test_transform_happy_path(self, sample_feature_engineer):
        data, config, cols_to_drop, new_names = sample_feature_engineer
        handler = FeatureEngineer(config)
        res = handler.transform(data)

        assert not set(cols_to_drop).issubset(res.columns)
        assert set(new_names).issubset(res.columns)

    def test_enhance_features_happy_path(self, sample_feature_engineer):
        data, config, _, _ = sample_feature_engineer
        handler = FeatureEngineer(config)
        handler._sum_features = MagicMock()
        handler._weighted_sum_features = MagicMock()
        handler._difference_features = MagicMock()
        handler._unequal_features = MagicMock()
        handler._greater_than_zero_features = MagicMock()

        for _, feature_config in config.items():
            handler.enhance_features(data, feature_config)

        handler._sum_features.assert_called_once()
        handler._weighted_sum_features.assert_called_once()
        handler._difference_features.assert_called_once()
        handler._unequal_features.assert_called_once()
        handler._greater_than_zero_features.assert_called_once()

    def test_enhance_features_not_supported_op(self, sample_feature_engineer):
        data, _, _, _ = sample_feature_engineer
        config = {
            'surface': {
                'op': 'power',
                'new_name': 'Total Bsmt SF Polynomial',
                'old': ['Total Bsmt SF']
            }
        }

        handler = FeatureEngineer(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*operation)(?=.*not)(?=.*power)"):
            for _, feature_config in config.items():
                handler.enhance_features(data, feature_config)

    def test_sum_happy_path(self, sample_sum_data):
        data, config, old_columns = sample_sum_data
        handler = FeatureEngineer({})
        res, actual_old = handler._sum_features(data, config)

        assert res['Total SF'].tolist() == [8, 8, 9]
        assert actual_old == old_columns

    def test_sum_with_nan(self, sample_sum_with_nan_data):
        # panda sum(axis=1) skips nan by default
        data, config = sample_sum_with_nan_data
        handler = FeatureEngineer({})
        res, _ = handler._sum_features(data, config)

        assert not res['Total SF'].isna().any()
        assert res['Total SF'].tolist() == [8, 8, 5]

    def test_sum_one_col(self, sample_sum_with_one_col):
        data, config, old_columns = sample_sum_with_one_col
        handler = FeatureEngineer({})
        res, actual_old = handler._sum_features(data, config)

        assert res['Total SF'].tolist() == [5, 5, 5]
        assert actual_old == old_columns

    def test_sum_weighted_sum_happy_path(self, sample_weighted_sum_data):
        data, config, old_columns = sample_weighted_sum_data
        handler = FeatureEngineer({})
        res, actual_old = handler._weighted_sum_features(data, config)

        assert res['Total Bath'].tolist() == [12, 6, 1]
        assert actual_old == old_columns

    def test_difference_features_happy_path(self, sample_difference_data):
        data, config, old_columns = sample_difference_data
        handler = FeatureEngineer({})
        res, actual_old = handler._difference_features(data, config)

        assert res['House Age'].tolist() == [5, 10, 30]
        assert actual_old == old_columns

    def test_unequal_happy_path(self, sample_unequal_data):
        data, config, old_cols = sample_unequal_data
        handler = FeatureEngineer({})
        res, actual_old = handler._unequal_features(data, config)
        
        assert res['Was Remodeled'].tolist() == [0, 1, 1]
        assert actual_old == old_cols

    def test_greater_than_zero_happy_path(self, sample_greater_than_zero_data):
        data, config, old_cols = sample_greater_than_zero_data
        handler = FeatureEngineer({})
        res, actual_old = handler._greater_than_zero_features(data, config)

        assert res['Has Pool'].tolist() == [1, 0, 1]
        assert actual_old == old_cols



