from src.utils import read_config, read_configs


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

    
    
