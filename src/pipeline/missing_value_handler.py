from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.pipeline_util import validate_columns


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        validate_columns(
            X,
            [*self.config.get("categorical"), *self.config.get("numerical")],
            self.__class__.__name__,
        )
        for feature_type, features in self.config.items():
            for feature in features:
                if feature_type == "categorical":
                    X[feature] = X[feature].fillna("None")
                elif feature_type == "numerical":
                    X[feature] = X[feature].fillna(0)
                else:
                    raise ValueError(f"Unsupported feature type: {feature_type}")

        return X
