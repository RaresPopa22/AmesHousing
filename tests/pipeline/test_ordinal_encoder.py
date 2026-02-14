import numpy as np
import pandas as pd
import pytest
import pandas.api.types as ptypes
from src.pipeline.ordinal_encoder import OrdinalEncoder


class TestOrdinalEncoder:

    def test_happy_path(self, sample_ordinal_data):
        df, config = sample_ordinal_data
        encoder = OrdinalEncoder(config)

        res = encoder.transform(df)
        for feature, ordering in config.items():
            expected_map = {cat: i for i, cat in enumerate(ordering)}
            expected = df[feature].map(expected_map)
            pd.testing.assert_series_equal(res[feature], expected)

    def test_pre_existing_nulls(self, sample_ordinal_data):
        df, config = sample_ordinal_data
        df = df.copy()
        df.loc[len(df), 'Lot Shape'] = np.nan

        encoder = OrdinalEncoder(config)
        
        with pytest.raises(ValueError, match=r"(?i)(?=.*Pre-existing)(?=.*Lot Shape)"):
            encoder.transform(df)

    def test_unseen_category(self, sample_ordinal_unseen_data):
        df, config = sample_ordinal_unseen_data

        encoder = OrdinalEncoder(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*unseen)(?=.*Unseen')"):
            encoder.transform(df)

    def test_validate_columns(self, sample_ordinal_data):
        df, config = sample_ordinal_data
        config = config.copy()
        config['Fence'] = ['MnWw']

        encoder = OrdinalEncoder(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*OrdinalEncoder)(?=.*Fence)"):
            encoder.transform(df)

    def test_pass_through(self, sample_ordinal_data):
        df, config = sample_ordinal_data
        encoder = OrdinalEncoder(config)
        res = encoder.transform(df)
        assert res.dtypes.apply(ptypes.is_numeric_dtype).all()