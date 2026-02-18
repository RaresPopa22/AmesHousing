import pytest
from src.pipeline.multicollinearity import MultiCollinearity


class TestMultiCollinearity:

    def test_fit_happy_path(self, sample_multicollinearity_data):
        X, y = sample_multicollinearity_data
        multicol = MultiCollinearity()
        multicol.fit(X, y)

        assert set(multicol.cols_to_drop_) == set(['A_2', 'B', 'C_83'])

    def test_fit_custom_threshold(self, sample_multicollinearity_data):
        X, y = sample_multicollinearity_data
        multicol = MultiCollinearity(0.84)
        multicol.fit(X, y)

        assert set(multicol.cols_to_drop_) == set(['A_2', 'B'])

    def test_fit_no_collinearity(self, sample_multicollinearity_data):
        X, y = sample_multicollinearity_data
        multicol = MultiCollinearity(1.1)
        multicol.fit(X, y)

        assert multicol.cols_to_drop_ == []

    def test_fit_y_corr_tiebreaker(self, sample_multicollinearity_tie_data):
        X, y = sample_multicollinearity_tie_data
        multicol = MultiCollinearity(0.8)
        multicol.fit(X, y)

        assert multicol.cols_to_drop_ == ['A']

    def test_transform_happy_path(self, sample_multicollinearity_data):
        X, y = sample_multicollinearity_data
        multicol = MultiCollinearity()
        multicol.fit(X, y)
        res = multicol.transform(X)

        assert not {'A_2', 'B', 'C_83'}.issubset(res)

    def test_transform_columns_changed_at_inference(self, sample_multicollinearity_data_changed):
        X, y, X_c = sample_multicollinearity_data_changed
        multicol = MultiCollinearity()
        multicol.fit(X, y)

        with pytest.raises(ValueError, match=r"(?i)(?=.*MultiCollinearity)(?=.*A_2)(?=.*B)"):
            multicol.transform(X_c)

