import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error

from src.data_processing import read_config, preprocess_data


def train_model_and_evaluate(config):
    X_train, X_test, y_train, y_test, prep = preprocess_data(config)

    alphas_to_test = np.logspace(-3, 3, 100)
    model = RidgeCV(alphas=alphas_to_test, cv=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_absolute_error(y_test, y_pred))

    print("Evaluation Metric on Test Set")
    print(f"R-squared: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.xlabel('Actual Sale Price ($)')
    plt.ylabel("Predicted Sale Price ($)")
    plt.title("Actual vs. Predicted Sale Prices")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    base_config = read_config("../config/base_config.yaml")
    train_model_and_evaluate(base_config)