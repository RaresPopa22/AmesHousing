import numpy as np
import pandas as pd
import pytest
from src.pipeline.missing_value_handler import MissingValueHandler


class TestMissingValueHandler:

    def test_happy_path(self, sample_cat_and_num_data):
        df, config = sample_cat_and_num_data
        handler = MissingValueHandler(config)
        res = handler.transform(df)

        assert not res.isna().any().any()
        assert (res['Misc Feature'] == 'None').all()
        assert (res['Bsmt Half Bath'] == 0).all()

    def test_unsupported_feature_type(self, sample_cat_and_num_data):
        df, config = sample_cat_and_num_data
        config = config.copy()
        config['imaginary'] = ['cat']

        with pytest.raises(ValueError, match=r"(?i)(?=.*unsupported)(?=.*imaginary)"):
            handler = MissingValueHandler(config)
            res = handler.transform(df)

    def test_missing_mandatory_type(self, sample_cat_and_num_data):
        df, config = sample_cat_and_num_data
        config = config.copy()
        config.pop('categorical')
        with pytest.raises(TypeError):
            handler = MissingValueHandler(config)
            res = handler.transform(df)

    def test_passthrough(self, sample_clean_cat_and_num_data):
        df, config = sample_clean_cat_and_num_data

        handler = MissingValueHandler(config)
        res = handler.transform(df)
        pd.testing.assert_frame_equal(df, res)

    def test_validate_columns(self, sample_clean_cat_and_num_data):
        df, config = sample_clean_cat_and_num_data
        config = config.copy()
        cat = config.get('categorical')
        cat.append('not_in_X')

        handler = MissingValueHandler(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*MissingValueHandler)(?=.*not_in_X)"):
            res = handler.transform(df)
