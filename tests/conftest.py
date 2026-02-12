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
    