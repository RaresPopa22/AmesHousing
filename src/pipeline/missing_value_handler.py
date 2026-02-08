from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature_type, features in self.config.items():
            for feature in features:
                if feature_type == 'categorical':
                    X[feature] = X[feature].fillna("None")
                elif feature_type == 'numerical':
                    X[feature] = X[feature].fillna(0)
                else:
                    raise ValueError(f'Unsupported feature type: {feature_type}')

        return X
