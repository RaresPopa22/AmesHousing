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
    np.random.seed(1)
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


@pytest.fixture
def sample_ordinal_data():
    np.random.seed(1)
    n_samples = 5
    lot_shape_list =  ['IR3', 'IR2', 'IR1', 'Reg']
    land_slope_list = ['Gtl', 'Mod', 'Sev']

    config = {
        'Lot Shape': lot_shape_list,
        'Land Slope': land_slope_list
    }

    data = pd.DataFrame({
        'Lot Shape': np.random.choice(lot_shape_list, n_samples),
        'Land Slope': np.random.choice(land_slope_list, n_samples)
    })

    return data, config

@pytest.fixture
def sample_ordinal_unseen_data():
    np.random.seed(1)
    lot_shape_list =  ['IR3', 'IR2', 'IR1', 'Reg']
    land_slope_list = ['Gtl', 'Mod', 'Sev']

    config = {
        'Lot Shape': lot_shape_list,
        'Land Slope': land_slope_list
    }

    data = pd.DataFrame({
        'Lot Shape': ['IR3', 'IR2', 'IR1', 'Reg'],
        'Land Slope': ['Gtl', 'Mod', 'Mod', 'Unseen']
    })

    return data, config

@pytest.fixture
def sample_rare_category_data():
    np.random.seed(1)
    n_samples = 99
    config = {
        'values': ['Street']
    }

    streets = ['Main Street', 'Boulevard']

    data = pd.DataFrame({
        'Street': np.random.choice(streets, n_samples)
    })

    data.loc[len(data), 'Street'] = 'Tiny little street'
    data.loc[len(data), 'Street'] = 'Unknown street'

    return data, config

@pytest.fixture
def sample_rare_category_data_no_outlier():
    np.random.seed(1)
    n_samples = 100
    streets = ['Main Street', 'Boulevard']
    config = {
        'values': ['Street']
    }

    data = pd.DataFrame({
        'Street': np.random.choice(streets, n_samples)
    })

    return data, config

@pytest.fixture
def sample_lot_frontage():
    np.random.seed(1)
    n_samples = 100
    config = {
        'neighborhood': 'Neighborhood',
        'lot_frontage': 'Lot Frontage'
    }

    data = pd.DataFrame({
        'Neighborhood': ['BrkSide', 'BrkSide', 'BrkSide', 'NoRidge', 'NoRidge', 'NoRidge'],
        'Lot Frontage': [90, 100, 110, 190, 200, 210]
    })

    return data, config

@pytest.fixture
def sample_lot_frontage_with_nan():
    np.random.seed(1)
    n_samples = 100
    config = {
        'neighborhood': 'Neighborhood',
        'lot_frontage': 'Lot Frontage'
    }

    data = pd.DataFrame({
        'Neighborhood': ['BrkSide', 'BrkSide', 'BrkSide', 'BrkSide', 'NoRidge', 'NoRidge', 'NoRidge', 'NoRidge', 'Mitchel'],
        'Lot Frontage': [90, 100, 110, np.nan, 190, 200, 210, np.nan, np.nan]
    })

    return data, config
    
@pytest.fixture
def sample_feature_engineer():
    np.random.seed(1)
    n_samples = 100
    config = {
            'surface': {
                'op': 'sum',
                'new_name': 'Total SF',
                'old': ['Total Bsmt SF', '1st Flr SF']
            },
            'age': {
                'op': 'difference',
                'new_name': 'House Age',
                'minuend': 'Yr Sold',
                'subtrahend': 'Year Built'
            },
            'bath': {
                'op': 'weighted_sum',
                'new_name': 'Total Bath',
                'old': {'Full Bath': 1, 'Half Bath': 0.5}
            },
            'was_remod': {
                'op': 'unequal',
                'new_name': 'Was Remodeled',
                'first_operand': 'Year Remod/Add',
                'second_operand': 'Year Built'
            },
            'has_pool': {
                'op': 'greater_than_zero',
                'new_name': 'Has Pool',
                'operand': 'Pool Area'
            }
        }
    
    cols_to_drop = ['Total Bsmt SF', '1st Flr SF', 'Yr Sold', 'Year Built', 'Full Bath', 'Half Bath', 'Year Remod/Add', 'Year Built', 'Pool Area']
    new_names = ['Total SF', 'House Age', 'Total Bath', 'Was Remodeled', 'Has Pool']
    
    data = pd.DataFrame({
        'Total Bsmt SF': [5, 5, 5],
        '1st Flr SF': [3, 3, 4],
        'Yr Sold': [2020, 2010, 2000],
        'Year Built': [2015, 2000, 1970],
        'Full Bath': [10, 5, 1],
        'Half Bath': [4, 2, 0],
        'Year Remod/Add': [2015, 2005, 1990],
        'Pool Area': [15, 0, 10]
    })

    return data, config, cols_to_drop, new_names

@pytest.fixture
def sample_sum_data():
    config = {
            'op': 'sum',
            'new_name': 'Total SF',
            'old': ['Total Bsmt SF', '1st Flr SF']
        }
    
    data = pd.DataFrame({
        'Total Bsmt SF': [5, 5, 5],
        '1st Flr SF': [3, 3, 4],
    })

    return data, config, config.get('old')

@pytest.fixture
def sample_sum_with_nan_data():
    config = {
            'op': 'sum',
            'new_name': 'Total SF',
            'old': ['Total Bsmt SF', '1st Flr SF']
        }
    
    data = pd.DataFrame({
        'Total Bsmt SF': [5, 5, 5],
        '1st Flr SF': [3, 3, np.nan],
    })

    return data, config

@pytest.fixture
def sample_sum_with_one_col():
    config = {
        'op': 'sum',
        'new_name': 'Total SF',
        'old': ['Total Bsmt SF']
    }

    data = pd.DataFrame({
        'Total Bsmt SF': [5, 5, 5],
    })

    return data, config, config.get('old')

@pytest.fixture
def sample_weighted_sum_data():
    config = {
        'op': 'weighted_sum',
        'new_name': 'Total Bath',
        'old': {'Full Bath': 1, 'Half Bath': 0.5}
    }

    data = pd.DataFrame({
        'Full Bath': [10, 5, 1],
        'Half Bath': [4, 2, 0],
    })

    return data, config, [*config.get('old')]

@pytest.fixture
def sample_difference_data():
    config = {
        'op': 'difference',
        'new_name': 'House Age',
        'minuend': 'Yr Sold',
        'subtrahend': 'Year Built'
    }

    data = pd.DataFrame({
        'Yr Sold': [2020, 2010, 2000],
        'Year Built': [2015, 2000, 1970],
    })

    return data, config, [config.get('minuend'), config.get('subtrahend')]

@pytest.fixture
def sample_unequal_data():
    config = {
        'op': 'unequal',
        'new_name': 'Was Remodeled',
        'first_operand': 'Year Remod/Add',
        'second_operand': 'Year Built'
    }

    data = pd.DataFrame({
        'Year Built': [2015, 2000, 1970],
        'Year Remod/Add': [2015, 2005, 1990]
    })

    return data, config, [config.get('first_operand'), config.get('second_operand')]

@pytest.fixture
def sample_greater_than_zero_data():
    config = {
        'op': 'greater_than_zero',
        'new_name': 'Has Pool',
        'operand': 'Pool Area'
    }

    data = pd.DataFrame({
        'Pool Area': [15, 0, 10]
    })

    return data, config, [config.get('operand')]

@pytest.fixture
def sample_multicollinearity_data():
    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'A_2': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 6],
        'C_83': [1, 2, 3, 4, 3],
        'D_79': [1, 2, 2, 2, 7],
        'E': [7, 8, 1, 8, 0]
    })

    y = pd.Series([1, 2, 3, 4, 5])

    return X, y

@pytest.fixture
def sample_multicollinearity_tie_data():
    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1.1, 2.1, 3.1, 4.1, 5.1],
    })

    y = pd.Series([1, 2, 3, 4, 5])

    return X, y

@pytest.fixture
def sample_multicollinearity_data_changed():
    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'A_2': [1, 2, 3, 4, 5],
        'B': [1, 2, 3, 4, 6],
        'C_83': [1, 2, 3, 4, 3],
        'D_79': [1, 2, 2, 2, 7],
        'E': [7, 8, 1, 8, 0]
    })

    y = pd.Series([1, 2, 3, 4, 5])
    X_c = X.drop(columns=['A_2', 'B'])

    return X, y, X_c

@pytest.fixture
def sample_feature_selection():
    config = {
        'apply': True,
        'values': ['Total Bsmt SF', '1st Flr SF', 'Yr Sold']
    }

    data = pd.DataFrame({
        'Total Bsmt SF': [5, 5, 5],
        '1st Flr SF': [3, 3, 4],
        'Yr Sold': [2020, 2010, 2000],
        'Year Built': [2015, 2000, 1970],
        'Full Bath': [10, 5, 1]
    })

    return data, config


