from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.pipeline_util import validate_columns


class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def __init__(self, config) -> None:
        self.config = config
        self.neighborhood = self.config.get('neighborhood')
        self.lot_frontage = self.config.get('lot_frontage')

    def fit(self, X, y=None):
        validate_columns(X, [self.neighborhood, self.lot_frontage], self.__class__.__name__)
        self.medians_ = X.groupby(self.neighborhood)[self.lot_frontage].median()
        self.global_median_ = X[self.lot_frontage].median()
        return self

    def transform(self, X):
        X = X.copy()
        validate_columns(X, [self.neighborhood, self.lot_frontage], self.__class__.__name__)
        X[self.lot_frontage] = X[self.lot_frontage].fillna(
            X[self.neighborhood].map(self.medians_)
        )
        X[self.lot_frontage] = X[self.lot_frontage].fillna(self.global_median_)

        return X
