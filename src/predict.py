import argparse
import joblib
import pandas as pd
import numpy as np

from src.utils import parse_args_and_get_config


def predict_from_dataframe(model, data: pd.DataFrame):
    data = data.copy()
    y_pred_log = model.predict(data)
    y_pred_original = np.expm1(y_pred_log)
    return y_pred_original


def predict(model_path, input_path):
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)

    if 'SalePrice' in data:
        data = data.drop('SalePrice', axis=1)

    return predict_from_dataframe(model, data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run house price prediction on new data")
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--input', required=True, help='Path to input')
    parser.add_argument('--output', required=True, help='Path to save prediction in CSV format')
    args = parser.parse_args()
    
    predictions = predict(args.model, args.input)
    results = pd.DataFrame({
        'predictions': predictions
    })
    results.to_csv(args.output, index=False)
    