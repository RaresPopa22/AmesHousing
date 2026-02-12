import logging

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline.pipeline_util import validate_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCollinearity(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def fit(self, X, y=None):
        cols_to_drop = set()
        
        y_corr = X.select_dtypes(include='number').corrwith(y).abs()

        corr_matrix = X.corr(numeric_only=True).abs()
        upper_triangle_mask = np.triu(corr_matrix, k=1).astype(bool)
        upper_matrix = corr_matrix.where(upper_triangle_mask)

        for i in range(len(upper_matrix.columns)):
            for j in range(1 + i, len(upper_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.threshold:
                    col_i = upper_matrix.columns[i]
                    col_j = upper_matrix.columns[j]
                    
                    if col_i in cols_to_drop or col_j in cols_to_drop:
                        continue
                    
                    to_drop = col_i if y_corr[col_i] < y_corr[col_j] else col_j

                    cols_to_drop.add(to_drop)
        
        self.cols_to_drop_ = list(cols_to_drop)

        return self

    def transform(self, X):
        X = X.copy()
        validate_columns(X, self.cols_to_drop_, self.__class__.__name__)
        return X.drop(columns=self.cols_to_drop_)
