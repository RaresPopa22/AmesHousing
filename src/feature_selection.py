import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from src.data_processing import preprocess_data
from src.utils import read_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_feature_importance(config):
    X_train, _, y_train, _, preprocessor, selector = preprocess_data(config)

    named_transformers = preprocessor.named_transformers_
    categorical_features = named_transformers['cat'].get_feature_names_out()
    numerical_features = named_transformers['num'].get_feature_names_out()
    original_feature_names = np.concatenate([numerical_features, categorical_features])
    selected_feature_names = original_feature_names[selector.get_support()]

    rf = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=1)
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        'feature': selected_feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f'Top 20 most important features: {importance.head(20)}')

    plot(importance)


def plot(importance):
    plt.figure(figsize=(10, 8))
    plt.barh(importance['feature'][:20], importance['importance'][:20])
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    config = read_config(Path(__file__).parent.parent / 'config' / 'base_config.yaml')
    get_feature_importance(config)
