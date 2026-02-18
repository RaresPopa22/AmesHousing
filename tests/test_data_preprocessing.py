import pandas as pd

from src.data_processing import handle_outliers_iqr


class TestDataPreprocessing:

    def test_handle_outliers_iqr(self):
        config = {
            'outliers': {
                'features': ['A']
            }
        }

        X = pd.DataFrame({
            'A': [1, 5 , 41, 42, 43, 44, 45, 77, 100]
        })
        y = pd.Series([20, 30, 34, 50, 60, 99, 100, 10, 1])

        X_actual, y_actual = handle_outliers_iqr(X, y, config)

        assert X_actual.index.equals(y_actual.index)

        assert not {1, 5, 77, 100}.intersection(X_actual['A'].values)
        assert {41, 42, 43, 44, 45}.issubset(X_actual['A'].values)

        assert not {20, 30, 10, 1}.intersection(set(y_actual.to_list()))
        assert {34, 50, 60, 99, 100}.issubset(set(y_actual.to_list()))