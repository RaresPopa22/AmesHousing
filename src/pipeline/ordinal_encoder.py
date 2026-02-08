from sklearn.base import BaseEstimator, TransformerMixin


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature, ordering in self.config.items():
            category_map = {cat: i for i, cat in enumerate(ordering)}
            X[feature] = X[feature].map(category_map)

        return X
