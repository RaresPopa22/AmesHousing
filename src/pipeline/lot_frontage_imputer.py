from sklearn.base import BaseEstimator, TransformerMixin


class LotFrontageImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.groupby('Neighborhood')['Lot Frontage'].median()
        self.global_median_ = X['Lot Frontage'].median()
        return self

    def transform(self, X):
        X = X.copy()
        X['Lot Frontage'] = X['Lot Frontage'].fillna(
            X['Neighborhood'].map(self.medians_)
        )
        X['Lot Frontage'] = X['Lot Frontage'].fillna(self.global_median_)

        return X
