import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def load_dataset(config):
    return pd.read_csv(config['data_paths']['raw_data'])

def preprocess_data(config):
    data = load_dataset(config)

    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train = handle_missing_features(X_train, config)
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

def enhance_features(op, data, feature):
    if op == 'sum':
        data[feature['new_name']] = data[feature['old']].sum(axis=1)
        drop_columns = feature['old']
    elif op == 'weighted_sum':
        weights = pd.Series(feature['old'])
        data[feature['new_name']] = (data[weights.index] * weights).sum(axis=1)
        drop_columns = weights.index
    elif op == 'difference':
        data[feature['new_name']] = data[feature['minuend']] - data[feature['subtrahend']]
        drop_columns = [feature['minuend'], feature['subtrahend']]
    elif op == 'unequal':
        data[feature['new_name']] = data[feature['first_operand']] != data[feature['second_operand']]
        drop_columns = [feature['first_operand'], feature['second_operand']]
    elif op == 'greater_than_zero':
        data[feature['new_name']] = data[feature['operand']] > 0
        drop_columns = [feature['operand']]
    else:
        raise ValueError("The operation is not yet supported.")

    return data, drop_columns

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
