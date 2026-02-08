import logging

import numpy as np

from src.utils import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(config):
    logger.info('Starting data preprocessing...')
    data = load_dataset(config)
    data['SalePrice'] = np.log1p(data['SalePrice'])
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']

    return X, y


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
