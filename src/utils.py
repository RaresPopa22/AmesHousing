import argparse

import joblib
import os
import pandas as pd
import yaml
import numpy as np

import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from matplotlib import pyplot as plt


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


def save_model(config, model):
    os.makedirs(config['model_output_paths']['dir'], exist_ok=True)
    joblib.dump(model, config['model_output_paths']['model'])
    print(f"Model saved successfully to {config['model_output_paths']['model']}")

def load_model(path):
    return joblib.load(path)


def evaluate_model(y_preds_original, y_test_original, model_names):
    for i, (y_pred, model_name) in enumerate(zip(y_preds_original, model_names)):
        print(f"Evaluation Metric on Test Set for {model_name}")
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
            0.05, 0.95, f'R^2 = {r2:.2f}',
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

    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")


def save_test_data(config, X_test, y_test):
    np.save(config['data_paths']['X_test_npy'], X_test)
    np.save(config['data_paths']['y_test_npy'], y_test)


def load_test_data(config):
    X_test = np.load(config['data_paths']['X_test_npy'])
    y_test = np.load(config['data_paths']['y_test_npy'])

    return X_test, y_test.squeeze()


def parse_args_and_get_config(stage):
    base_config = "../config/base_config.yaml"
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


