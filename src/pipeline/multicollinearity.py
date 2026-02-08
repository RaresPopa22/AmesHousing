import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCollinearity(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        correlation_matrix = X.corr(numeric_only=True).abs()
        upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        upper_matrix = correlation_matrix.where(upper_triangle_mask)
        self.cols_to_drop_ = [c for c in upper_matrix.columns if any(upper_matrix[c] > self.threshold)]
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop_)
