import pandas as pd
import yaml


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def load_dataset(config):
    print(config['data_paths']['raw_data'])
    return pd.read_csv(config['data_paths']['raw_data'])

def preprocess_data(config):
    data = load_dataset(config)
    data = data.drop(['PID'], axis=1)
    data = handle_missing_values(data, config)

    print(data.head())


def handle_missing_values(data, config):
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

if __name__ == '__main__':
    base_config = read_config("../config/base_config.yaml")
    preprocess_data(base_config)