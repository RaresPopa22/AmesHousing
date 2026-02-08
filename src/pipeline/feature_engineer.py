from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        columns_to_drop = []

        for feature, feature_config in self.config.items():
            X, drop_cols = self.enhance_features(X, feature_config)
            columns_to_drop.extend(drop_cols)

        unique_columns_to_drop = set(columns_to_drop)
        X = X.drop(columns=unique_columns_to_drop, axis=1)

        return X

    def enhance_features(self, X, feature_config):
        op_name = feature_config.get('op')

        op_dict = {
            'sum': self._sum_features,
            'weighted_sum': self._weighted_sum_features,
            'difference': self._difference_features,
            'unequal': self._unequal_features,
            'greater_than_zero': self._greater_than_zero_features
        }

        if op_name in op_dict:
            return op_dict[op_name](X, feature_config)
        else:
            raise ValueError(f'The operation {op_name} is not yet supported')

    def _get_new_col(self, feature):
        return feature['new_name']

    def _sum_features(self, data, feature):
        old_columns = feature['old']
        new_col = self._get_new_col(feature)
        data[new_col] = data[old_columns].sum(axis=1)
        return data, old_columns

    def _weighted_sum_features(self, data, feature):
        weights = pd.Series(feature['old'])
        new_col = self._get_new_col(feature)
        data[new_col] = (data[weights.index] * weights).sum(axis=1)
        return data, list(weights.index)

    def _difference_features(self, data, feature):
        new_col = self._get_new_col(feature)
        minuend_col = feature['minuend']
        subtrahend_col = feature['subtrahend']
        data[new_col] = data[minuend_col] - data[subtrahend_col]
        return data, [minuend_col, subtrahend_col]

    def _unequal_features(self, data, feature):
        new_col = self._get_new_col(feature)
        first_operand_col = feature['first_operand']
        second_operand_col = feature['second_operand']
        data[new_col] = (data[first_operand_col] != data[second_operand_col]).astype(int)
        return data, [first_operand_col, second_operand_col]

    def _greater_than_zero_features(self, data, feature):
        new_col = self._get_new_col(feature)
        operand_col = feature['operand']
        data[new_col] = (data[operand_col] > 0).astype(int)
        return data, [operand_col]
