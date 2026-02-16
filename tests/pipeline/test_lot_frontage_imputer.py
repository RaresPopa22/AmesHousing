import pandas as pd
import pytest

from src.pipeline.lot_frontage_imputer import LotFrontageImputer


class TestLotFrontageImputer:
    
    def test_missing_configuration(self):
        with pytest.raises(ValueError, match=r"(?i)(?=.*neighborhood)(?=.*lot_frontage)"):
            handler = LotFrontageImputer({})

    def test_validate_columns(self):
        df = pd.DataFrame({})
        config = {'neighborhood': 'Neighborhood', 'lot_frontage': 'Different Lot Frontage'}
        handler = LotFrontageImputer(config)

        with pytest.raises(ValueError, match=r"(?i)(?=.*LotFrontageImputer)(?=.*Different Lot Frontage)"):
            res = handler.fit(df)

    def test_fit_happy_path(self, sample_lot_frontage):
        data, config = sample_lot_frontage
        imputer = LotFrontageImputer(config)
        imputer.fit(data)

        assert imputer.medians_['BrkSide'] == 100
        assert imputer.medians_['NoRidge'] == 200
        assert imputer.global_median_ == 150

    def test_transform_happy_path(self, sample_lot_frontage_with_nan):
        data, config = sample_lot_frontage_with_nan
        imputer = LotFrontageImputer(config)
        imputer.fit(data)

        nan_mask = data['Lot Frontage'].isna()
        res = imputer.transform(data)

        assert not res.isnull().values.any()

        expected = data.loc[nan_mask, 'Neighborhood'].map(imputer.medians_)
        expected = expected.fillna(imputer.global_median_)
        pd.testing.assert_series_equal(
            res.loc[nan_mask, 'Lot Frontage'],
            expected,
            check_names=False
        )

    






