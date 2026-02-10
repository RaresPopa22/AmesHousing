from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.pipeline_util import validate_columns


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        validate_columns(X, list(self.config.keys()), self.__class__.__name__)
        for feature, ordering in self.config.items():
            category_map = {cat: i for i, cat in enumerate(ordering)}
            X[feature] = X[feature].map(category_map)

        return X
