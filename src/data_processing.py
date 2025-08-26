import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import load_dataset


def preprocess_data(config):
    data = load_dataset(config)
    data['SalePrice'] = np.log1p(data['SalePrice'])

    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_test = handle_multicollinearity(X_train, X_test, config)

    X_train = handle_missing_features(X_train, config)
    X_train, y_train = handle_outliers_iqr(X_train, y_train, config)
    X_test = handle_missing_features(X_test, config)

    X_train = handle_ordinal_features(X_train, config)
    X_test = handle_ordinal_features(X_test, config)

    X_train = handle_rare_categories(X_train, config)
    X_test = handle_rare_categories(X_test, config)

    X_train = feature_engineering(X_train, config)
    X_test = feature_engineering(X_test, config)

    preprocessor = get_preprocessor(X_train)

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    selector = VarianceThreshold(threshold=0.01)
    X_train_reduced = selector.fit_transform(X_train_scaled)
    X_test_reduced = selector.transform(X_test_scaled)

    return X_train_reduced, X_test_reduced, y_train, y_test, preprocessor


def handle_missing_features(data, config):
    data = data.drop(['PID'], axis=1)
    cols_to_fill = config['cols_to_fill']

    for feature_type in cols_to_fill:
        for feature in cols_to_fill[feature_type]:
            if feature_type == 'categorical':
                data[feature] = data[feature].fillna("None")
            elif feature_type == 'numerical':
                data[feature] = data[feature].fillna(0)

    data['Lot Frontage'] = data.groupby('Neighborhood')['Lot Frontage'].transform(
        lambda x: x.fillna(x.median())
    )

    return data


def handle_ordinal_features(data, config):
    ordinal_features_config = config['ordinal_features']
    for feature, ordering in ordinal_features_config.items():
        category_map = {category: i for i, category in enumerate(ordering)}
        data[feature] = data[feature].map(category_map)
    return data


def handle_rare_categories(data, config):
    rare_categories = config['rare_categories']
    for feature in rare_categories['values']:
        value_counts = data[feature].value_counts()
        low_freq_mask = value_counts / len(data) < 0.01
        low_freq = value_counts[low_freq_mask].index
        data[feature] = data[feature].replace(low_freq, 'Other')

    return data


def feature_engineering(data, config):
    feature_engineering_config = config['feature_engineering']
    cols_to_drop = []

    for feature in feature_engineering_config:
        feature_config = feature_engineering_config[feature]
        data, drop_cols = enhance_features(feature_config['op'], data, feature_config)
        cols_to_drop.extend(drop_cols)

    unique_cols_to_drop = set(cols_to_drop)
    data = data.drop(columns=unique_cols_to_drop)

    return data


def enhance_features(op_name, data, feature):
    op_dict = {
        'sum': _sum_features,
        'weighted_sum': _weighted_sum_features,
        'difference': _difference_features,
        'unequal': _unequal_features,
        'greater_than_zero': _greater_than_zero_features
    }

    if op_name in op_dict:
        return op_dict[op_name](data, feature)
    else:
        raise ValueError(f'The operation {op_name} is not yet supported')


def _get_new_col(feature):
    return feature['new_name']


def _sum_features(data, feature):
    old_columns = feature['old']
    new_col = _get_new_col(feature)
    data[new_col] = data[old_columns].sum(axis=1)
    return data, old_columns


def _weighted_sum_features(data, feature):
    weights = pd.Series(feature['old'])
    new_col = _get_new_col(feature)
    data[new_col] = (data[weights.index] * weights).sum(axis=1)
    return data, list(weights.index)


def _difference_features(data, feature):
    new_col = _get_new_col(feature)
    minuend_col = feature['minuend']
    subtrahend_col = feature['subtrahend']
    data[new_col] = data[minuend_col] - data[subtrahend_col]
    return data, [minuend_col, subtrahend_col]


def _unequal_features(data, feature):
    new_col = _get_new_col(feature)
    first_operand_col = feature['first_operand']
    second_operand_col = feature['second_operand']
    data[new_col] = (data[first_operand_col] != data[second_operand_col]).astype(int)
    return data, [first_operand_col, second_operand_col]


def _greater_than_zero_features(data, feature):
    new_col = _get_new_col(feature)
    operand_col = feature['operand']
    data[new_col] = (data[operand_col] > 0).astype(int)
    return data, [operand_col]


def get_preprocessor(X_train):
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_features = X_train.select_dtypes(include=['int64', 'float']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor


def handle_outliers_iqr(X_train, y_train, config):
    outlier_config = config['outliers']

    for feature in outlier_config['features']:
        Q1 = X_train[feature].quantile(0.25)
        Q3 = X_train[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (X_train[feature] >= lower_bound) & (X_train[feature] <= upper_bound)
        X_train = X_train[mask]
        y_train = y_train[mask]

    return X_train, y_train

def handle_multicollinearity(X_train, X_test, config):
    multicollinearity_config = config['multicollinearity']
    threshold = multicollinearity_config['threshold']

    correlation_matrix = X_train.corr(numeric_only=True).abs()
    upper_triangle = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    upper = correlation_matrix.where(upper_triangle)
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    return X_train, X_test


