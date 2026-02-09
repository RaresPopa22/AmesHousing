import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def drop_columns(X, cols):
    return X.drop(columns=cols)


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return {**base_config, **specific_config}


def load_dataset(config):
    return pd.read_csv(config['data_paths']['raw_data'])


def evaluate_pipeline(y_preds_original, y_test_original, model_names):
    for i, (y_pred, model_name) in enumerate(zip(y_preds_original, model_names)):
        logger.info(f"Evaluation Metric on Test Set for {model_name}")
        get_metrics_and_print(y_pred, y_test_original)

    plot_actual_vs_predicted(y_preds_original, y_test_original, model_names)


def plot_actual_vs_predicted(y_preds_original, y_test_original, model_names):
    plot_data = add_plot_data(model_names, y_preds_original, y_test_original)
    combined_df = pd.concat(plot_data, ignore_index=True)

    g = sns.lmplot(
        data=combined_df,
        x='Actual Sale Price ($)',
        y='Predicted Sale Price ($)',
        col='Model',
        hue='Model',
        height=6,
        aspect=1,
        scatter_kws={'alpha': 0.5},
        line_kws={'linestyle': '-'}
    )

    ax_add_text(g, model_names, y_preds_original, y_test_original)
    g.figure.suptitle('Actual vs. Predicted Sale Prices by model', y=1.03)
    plt.show()


def ax_add_text(g, model_names, y_preds_original, y_test_original):
    for i, model_name in enumerate(model_names):
        ax = g.axes[0, i]
        r2 = r2_score(y_test_original, y_preds_original[i])

        ax.text(
            0.05, 0.95, f'R^2 = {r2:.3f}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
        )


def add_plot_data(model_names, y_preds_original, y_test_original):
    plot_data = []
    for i, y_pred in enumerate(y_preds_original):
        model_name = model_names[i]
        temp_df = pd.DataFrame({
            'Actual Sale Price ($)': y_test_original,
            'Predicted Sale Price ($)': y_pred,
            'Model': model_name
        })

        plot_data.append(temp_df)
    return plot_data


def get_metrics_and_print(y_pred_original, y_test_original):
    r2 = r2_score(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

    logger.info(f"R-squared: {r2:.3f}")
    logger.info(f"Mean Absolute Error (MAE): ${mae:,.3f}")
    logger.info(f"Root Mean Squared Error (RMSE): ${rmse:,.3f}")


def save_test_data(config, X_test, y_test):
    joblib.dump(X_test, config['data_paths']['X_test_job'])
    joblib.dump(y_test, config['data_paths']['y_test_job'])


def load_test_data(config):
    X_test = joblib.load(config['data_paths']['X_test_job'])
    y_test = joblib.load(config['data_paths']['y_test_job'])

    return X_test, y_test.squeeze()


def check_for_invalid_data(full_pipeline, X, y):
    preprocessing_pipeline = full_pipeline[:-1]
    X_transformed = preprocessing_pipeline.fit_transform(X, y)

    logger.info(f'Shape {X_transformed.shape}')
    logger.info(f'NaN count: {np.isnan(X_transformed).sum()}')
    logger.info(f'Inf count: {np.isinf(X_transformed).sum()}')
    logger.info(f'Max value: {np.nanmax(X_transformed)}')
    logger.info(f'Min value: {np.nanmin(X_transformed)}')


def parse_args_and_get_config(stage):
    base_config = Path(__file__).parent.parent / 'config' / 'base_config.yaml'
    parser = argparse.ArgumentParser()

    if stage == 'train':
        parser.add_argument('--config', required=True, help='Path to configuration file')
        args = parser.parse_args()
        return read_configs(base_config, args.config)
    elif stage == 'evaluate':
        parser.add_argument('--models', nargs="+", required=True, help='Paths to model files')
        parser.add_argument('--x-test', required=True, help='Paths to X_test_scaled.csv')
        parser.add_argument('--y-test', required=True, help='Paths to y_test.csv')
        args = parser.parse_args()
        return read_config(base_config), args.models
    else:
        raise ValueError("Unknown stage. Only 'train' and 'evaluate' supported so far")
