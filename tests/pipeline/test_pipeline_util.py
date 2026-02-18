import pytest
from src.pipeline.pipeline_util import validate_columns


class TestPipelineUtil:

    def test_raise_error(self, sample_validate_data, sample_validate_columns):
        with pytest.raises(ValueError, match=r"(?=.*test_transformer)(?=.*extra)"):
            expected_columns_test = sample_validate_columns.copy()
            expected_columns_test.append('extra')
            validate_columns(sample_validate_data, expected_columns_test, 'test_transformer')

    def test_happy_path(self, sample_validate_data, sample_validate_columns):
        
        expected_columns_test = sample_validate_columns.copy()
        validate_columns(sample_validate_data, expected_columns_test, 'test_transformer')

    def test_empty_expected_columns_list(self, sample_validate_data):
        validate_columns(sample_validate_data, [], 'test_transformer')
        