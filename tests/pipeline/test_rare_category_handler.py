import pandas as pd
import pytest
from src.pipeline.rare_category_handler import RareCategoryHandler


class TestRareCategoryHandler:

    def test_validate_columns(self):
        df = pd.DataFrame({})
        config = {'values': ['not_in_X']}
        handler = RareCategoryHandler(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*RareCategoryHandler)(?=.*not_in_X)"):
            res = handler.fit(df)

    def test_missing_configuration(self):
        with pytest.raises(ValueError, match=r"(?i)(?=.*values)"):
            handler = RareCategoryHandler({})

    def test_fit_happy_path(self, sample_rare_category_data):
        data, config = sample_rare_category_data
        handler = RareCategoryHandler(config)
        handler.fit(data)

        low_freq_dict = dict(handler.low_freq_)
        rare_streets = low_freq_dict['Street'].to_list()

        assert 'Tiny little street' in rare_streets
        assert 'Unknown street' in rare_streets

    def test_fit_no_value_over_threshold(self, sample_rare_category_data_no_outlier):
        data, config = sample_rare_category_data_no_outlier
        handler = RareCategoryHandler(config)
        handler.fit(data)

        assert handler.low_freq_ == []

    def test_fit_does_not_mutate_input(self, sample_rare_category_data):
        data, config = sample_rare_category_data
        original = data.copy()
        handler = RareCategoryHandler(config)
        handler.fit(data)

        pd.testing.assert_frame_equal(data, original)

    def test_transformer_happy_path(self, sample_rare_category_data):
        data, config = sample_rare_category_data
        handler = RareCategoryHandler(config)
        handler.fit(data)
        res = handler.transform(data)

        assert 'Tiny little street' not in res['Street'].values
        assert 'Unknown street' not in res['Street'].values
        assert 'Main Street' in res['Street'].values
        assert 'Boulevard' in res['Street'].values