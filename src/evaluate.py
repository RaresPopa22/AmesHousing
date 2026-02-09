import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_test_data, parse_args_and_get_config, evaluate_pipeline, check_for_invalid_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_models(base_config, model_paths):
    X_test, y_test = load_test_data(base_config)
    y_test_original = np.expm1(y_test)

    predictions = []
    model_names = []

    fig, axes = plt.subplots(len(model_paths), 1, figsize=(6, 3 * len(model_paths)), squeeze=False)
    axes = axes.flatten()

    for i, path in enumerate(model_paths):
        model_path = Path(path)
        base_config['model_name'] = model_path.stem
        if not model_path.exists():
            logger.info(f"Model file not found at {model_path}. Skipping...")
            continue

        pipeline = joblib.load(model_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y_pred_log = pipeline.predict(X_test)

        y_pred_original = np.expm1(y_pred_log)
        predictions.append(y_pred_original)
        model_names.append(model_path.stem)

        coef = pipeline.named_steps['model'].coef_
        features_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        features_names = [name.replace('num__', "").replace('cat__', '') for name in features_names]

        n = 10
        top_n_features_tuple = sorted(zip(features_names, coef), key=lambda x: abs(x[1]), reverse=True)[:n]
        top_n_features, top_n_coef = zip(*top_n_features_tuple)

        bars = axes[i].barh(top_n_features, top_n_coef, align='center')
        axes[i].bar_label(bars, fmt='%.3f', padding=3)
        axes[i].margins(x=0.15)
        axes[i].invert_yaxis()
        axes[i].set_xlabel("Coefficient")
        axes[i].set_title(f"Top {n} features - {model_path.stem}")

    plt.tight_layout()
    plt.show()

    evaluate_pipeline(predictions, y_test_original, model_names)


if __name__ == '__main__':
    config, model_paths = parse_args_and_get_config("evaluate")
    evaluate_models(config, model_paths)
