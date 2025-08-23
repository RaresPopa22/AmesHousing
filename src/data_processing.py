import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    X_train = handle_missing_values(X_train, config)
    X_train = feature_engineering(X_train, config)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    print(X_train.head())


def handle_missing_values(data, config):
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
        data[feature['new']] = data[feature['old']].sum(axis=1)
        drop_columns = feature['old']
    elif op == 'weighted_sum':
        weights = pd.Series(feature['old'])
        data[feature['new']] = (data[weights.index] * weights).sum(axis=1)
        drop_columns = weights.index
    elif op == 'difference':
        data[feature['new']] = data[feature['minuend']] - data[feature['subtrahend']]
        drop_columns = [feature['minuend'], feature['subtrahend']]
    elif op == 'unequal':
        data[feature['new']] = data[feature['first_operand']] != data[feature['second_operand']]
        drop_columns = [feature['first_operand'], feature['second_operand']]
    elif op == 'greater_than_zero':
        data[feature['new']] = data[feature['operand']] > 0
        drop_columns = [feature['operand']]
    else:
        raise ValueError("The operation is not yet supported.")

    return data, drop_columns

if __name__ == '__main__':
    base_config = read_config("../config/base_config.yaml")
    preprocess_data(base_config)