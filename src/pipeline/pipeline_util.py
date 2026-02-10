def validate_columns(X, expected_columns, transformer_name):
    expected_set = set(expected_columns)
    if not expected_set.issubset(X.columns):
        missing = expected_set - X.columns
        raise ValueError(f'{transformer_name} missing columns {missing} from input. Available: {list(X.columns)}')