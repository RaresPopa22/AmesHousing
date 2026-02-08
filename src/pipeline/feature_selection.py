from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.config['apply']:
            top_features = self.config.get('values', None)
            features_to_keep = top_features
            X = X[features_to_keep]

        return X
