import logging

import joblib
import numpy as np

from pathlib import Path

from src.utils import load_test_data, evaluate_pipeline, parse_args_and_get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_models(base_config, model_paths):
    X_test, y_test = load_test_data(base_config)
    y_test_original = np.expm1(y_test)

    predictions = []
    model_names = []

    for path in model_paths:
        model_path = Path(path)
        base_config['model_name'] = model_path.stem
        if not model_path.exists():
            logger.info(f"Model file not found at {model_path}. Skipping...")
            continue

        pipeline = joblib.load(model_path)
        y_pred_log = pipeline.predict(X_test)

        y_pred_original = np.expm1(y_pred_log)
        predictions.append(y_pred_original)
        model_names.append(model_path.stem)

    evaluate_pipeline(predictions, y_test_original, model_names)


if __name__ == '__main__':
    config, model_paths = parse_args_and_get_config("evaluate")
    evaluate_models(config, model_paths)
