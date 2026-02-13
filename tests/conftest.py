import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_validate_data():
    np.random.seed(1)
    n_samples = 2

    return pd.DataFrame({
        'Overall Qual': np.random.randint(0, 4987, n_samples), 
        'Total SF': np.random.randint(0, 4987, n_samples),
        'House Age': np.random.randint(0, 4987, n_samples), 
        'Gr Liv Area': np.random.randint(0, 4987, n_samples)    
    })

@pytest.fixture
def sample_validate_columns(sample_validate_data):
    return list(sample_validate_data.columns.values)
    
@pytest.fixture
def sample_categorical_data():
    np.random.seed(1)
    n_samples = 5

    return pd.DataFrame({
        'Pool QC': np.random.choice(['Fa', 'TA', 'Gd', 'Ex'], n_samples),
        'Fence': np.random.choice(['MnWw', 'GdWo', 'MnPrv', 'GdPrv'], n_samples),
        'Misc Feature': [np.nan] * n_samples
    })

@pytest.fixture
def sample_numerical_data():
    np.random.seed(1)
    n_samples = 5

    return pd.DataFrame({
        'Mas Vnr Area': np.random.randint(0, 4987, n_samples),
        'Bsmt Full Bath': np.random.randint(0, 4987, n_samples),
        'Bsmt Half Bath': [np.nan] * n_samples
    })

@pytest.fixture
def sample_cat_and_num_data(sample_categorical_data, sample_numerical_data):
    categorical_cols = sample_categorical_data.columns.to_list()
    numerical_cols = sample_numerical_data.columns.to_list()
    df = pd.concat([sample_categorical_data, sample_numerical_data], axis=1)
    config = {'categorical': categorical_cols, 'numerical': numerical_cols}

    return df, config

@pytest.fixture
def sample_clean_cat_and_num_data():
    n_samples = 2

    cat = pd.DataFrame({
        'Pool QC': np.random.choice(['Fa', 'TA', 'Gd', 'Ex'], n_samples),
        'Fence': np.random.choice(['MnWw', 'GdWo', 'MnPrv', 'GdPrv'], n_samples),
    })
    num = pd.DataFrame({
        'Mas Vnr Area': np.random.randint(0, 4987, n_samples),
        'Bsmt Full Bath': np.random.randint(0, 4987, n_samples),
    })

    categorical_cols = cat.columns.to_list()
    numerical_cols = num.columns.to_list()
    df = pd.concat([cat, num], axis=1)
    config = {'categorical': categorical_cols, 'numerical': numerical_cols}

    return df, config
