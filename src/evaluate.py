import numpy as np

from pathlib import Path

from src.utils import load_test_data, load_model, evaluate_model, parse_args_and_get_config


def evaluate_models(base_config, model_paths):
    X_test, y_test = load_test_data(base_config)

    for path in model_paths:
        model_path = Path(path)
        base_config['model_name'] = model_path.stem
        if not model_path.exists():
            print(f"Model file not found at {model_path}. Skipping...")
            continue

        model_path = Path(path)
        base_config['model_name'] = model_path.stem

        model = load_model(model_path)
        y_pred_log = model.predict(X_test)

        y_test_original = np.expm1(y_test)
        y_pred_original = np.expm1(y_pred_log)

        evaluate_model(y_pred_original, y_test_original)


if __name__ == '__main__':
    config, model_paths = parse_args_and_get_config("evaluate")
    evaluate_models(config, model_paths)