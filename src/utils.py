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
        get_metrics_and_print(y_pred, y_test_original)

    plot_actual_vs_predicted(y_preds_original, y_test_original, model_names)


def plot_actual_vs_predicted(y_preds_original, y_test_original, model_names):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=y_test_original, y=y_test_original, color='gray', label='Actual Values', alpha=0.6)
    colors = sns.color_palette("deep", len(y_preds_original))

    for i, (y_pred, model_name) in enumerate(zip(y_preds_original, model_names)):
        sns.regplot(x=y_test_original, y=y_pred, scatter=False, line_kws={'linestyle': '--'}, color=colors[i], label=f'Predicted - {model_name}')

    plt.xlabel('Actual Sale Price ($)')
    plt.ylabel("Predicted Sale Price ($)")
    plt.title("Actual vs. Predicted Sale Prices")
    plt.legend()
    plt.grid(True)
    plt.show()


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


