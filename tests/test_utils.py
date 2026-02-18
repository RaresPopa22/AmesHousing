import pandas as pd
from src.utils import drop_columns, get_metrics_and_print, load_test_data, read_config, read_configs, save_test_data


class TestUtils:
    
    def test_read_config_happy_path(self, tmp_path):
        yaml_path = tmp_path / 'mock.yaml'
        yaml_path.write_text("model_name: ridge\ncv: 5\n")

        result = read_config(yaml_path)
        assert result == {'model_name': 'ridge', 'cv': 5}

    def test_read_configs_happy_path(self, tmp_path):
        base_config = tmp_path / "base.yaml"
        base_config.write_text("data_paths:\n  raw_data: 'data/raw/AmesHousing.csv'")

        specific_config = tmp_path / "specific.yaml"
        specific_config.write_text("model_name: 'lasso_cv'")

        result = read_configs(base_config, specific_config)

        assert result['data_paths']['raw_data'] == 'data/raw/AmesHousing.csv'
        assert result['model_name'] == 'lasso_cv'

    def test_get_metrics_and_print_r2(self):
        y_predicted = [1, 2, 3]
        y_true = [1, 2, 3]

        r2, _, _ = get_metrics_and_print(y_predicted, y_true) 
        assert r2 == 1

    def test_get_metrics_and_and_print_mae(self):
        y_predicted = [3, 4, 5]
        y_true = [1, 2, 3]

        _, mae, _ = get_metrics_and_print(y_predicted, y_true) 
        assert mae == 2

    def test_get_metrics_and_and_print_rmse(self):
        y_predicted = [4, 5, 6]
        y_true = [1, 2, 3]

        _, _, rmse = get_metrics_and_print(y_predicted, y_true) 
        assert rmse == 3

    def test_save_and_load_data(self, tmp_path):
        X = pd.DataFrame({'A': [1, 2, 3]})
        y = pd.Series([1, 0, 1])

        config = {
            'data_paths': {
                'X_test_job': tmp_path / 'X_test.joblib',
                'y_test_job': tmp_path / 'y_test.joblib'
            }
        }

        save_test_data(config, X, y)
        X_actual, y_actual = load_test_data(config)

        pd.testing.assert_frame_equal(X, X_actual)
        pd.testing.assert_series_equal(y, y_actual)

    def test_drop_columns(self):
        X = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [1, 2, 3],
            'C': [1, 2, 3],
            })
        
        res = drop_columns(X, ['B', 'C'])
        assert not {'B', 'C'}.issubset(res.columns)

