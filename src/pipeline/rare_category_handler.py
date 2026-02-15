from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.pipeline_util import validate_columns


class RareCategoryHandler(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        if 'values' not in config:
            raise ValueError('RareCategoryHandler config missing required key: values')
        self.config = config

    def fit(self, X, y=None):
        self.low_freq_ = []
        validate_columns(X, self.config['values'], self.__class__.__name__)
        for feature in self.config['values']:
            value_counts = X[feature].value_counts()
            low_freq_mask = value_counts / len(X) < self.config.get("threshold", 0.01)
            if low_freq_mask.any():
                low_freq = value_counts[low_freq_mask].index
                self.low_freq_.append((feature, low_freq))

        return self

    def transform(self, X):
        X = X.copy()
        validate_columns(X, self.config['values'], self.__class__.__name__)
        for feature, low_freq in self.low_freq_:
            X[feature] = X[feature].replace(low_freq, 'Other')
        return X
