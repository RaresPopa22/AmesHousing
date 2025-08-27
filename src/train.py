import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV

from src.data_processing import preprocess_data
from src.utils import save_model, save_test_data, parse_args_and_get_config


def train_model_and_evaluate(config):
    model_name = config["model_name"]
    X_train, X_test, y_train, y_test, prep = preprocess_data(config)
    alphas_to_test = np.logspace(-3, 3, 100)

    if model_name == 'ridge_cv':
        model = RidgeCV(alphas=alphas_to_test, cv=5)
    elif model_name == 'lasso_cv':
        model = LassoCV(alphas=alphas_to_test, cv=5, random_state=1)

    model.fit(X_train, y_train)
    save_model(config, model)
    save_test_data(config, X_test, y_test)

if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train_model_and_evaluate(config)
