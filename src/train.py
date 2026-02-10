import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from src.data_processing import preprocess_data, handle_outliers_iqr
from src.pipeline.feature_engineer import FeatureEngineer
from src.pipeline.feature_selection import FeatureSelection
from src.pipeline.lot_frontage_imputer import LotFrontageImputer
from src.pipeline.missing_value_handler import MissingValueHandler
from src.pipeline.multicollinearity import MultiCollinearity
from src.pipeline.ordinal_encoder import OrdinalEncoder
from src.pipeline.rare_category_handler import RareCategoryHandler
from src.utils import save_test_data, parse_args_and_get_config, drop_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model_and_evaluate(config):
    model_name = config["model_name"]
    logger.info(f'Starting training for model: {model_name}')
    X, y = preprocess_data(config)
    preprocessing_config = config.get('preprocessing', {})
    logspace_config = preprocessing_config.get('logspace', {})
    alphas_to_test = np.logspace(
        logspace_config.get('start', -3),
        logspace_config.get('end', 3),
        logspace_config.get('num', 100)
    )

    if model_name == 'ridge_cv':
        model = RidgeCV(alphas=alphas_to_test, cv=preprocessing_config.get('cv', 5))
    elif model_name == 'lasso_cv':
        model = LassoCV(alphas=alphas_to_test, cv=preprocessing_config.get('cv', 5), random_state=1)
    elif model_name == 'elastic_net_cv':
        model = ElasticNetCV(alphas=alphas_to_test, cv=preprocessing_config.get('cv', 5), random_state=1)
    else:
        raise ValueError(f'The model: {model_name} is not supported yet')

    columns_to_drop = config.get('cols_to_drop', [])

    full_pipeline = Pipeline([
        ('drop_cols', FunctionTransformer(drop_columns, kw_args={'cols': columns_to_drop})),
        ('feature_selection', FeatureSelection(config.get('top_20_features'))),
        ('multicollinearity', MultiCollinearity(config.get('multicollinearity', {}).get('threshold'))),
        ('missing', MissingValueHandler(config['cols_to_fill'])),
        ('lot_frontage', LotFrontageImputer(config['lot_frontage'])),
        ('ordinal', OrdinalEncoder(config['ordinal_features'])),
        ('feature_engineer', FeatureEngineer(config['feature_engineer'])),
        ('rare_category', RareCategoryHandler(config["rare_categories"])),
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('variance', VarianceThreshold(threshold=preprocessing_config.get('variance_threshold', 0.01))),
                ('scaler', StandardScaler())
            ]), make_column_selector(dtype_include=np.number)),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'),
             make_column_selector(dtype_include='object'))
        ])),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, y_train = handle_outliers_iqr(X_train, y_train, config)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                message="overflow encountered in matmul|invalid value encountered in matmul|divide by zero encountered in matmul",
                                category=RuntimeWarning)
        full_pipeline.fit(X_train, y_train)

    logger.info(f'Best alpha {model.alpha_}')
    root_dir = Path(__file__).parent.parent
    model_paths = config['model_output_paths']
    joblib.dump(full_pipeline, root_dir / model_paths['dir'] / model_paths['pipeline'])
    save_test_data(config, X_test, y_test)


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train_model_and_evaluate(config)
