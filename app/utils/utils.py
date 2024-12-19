import os
from datetime import timedelta, datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def metrics(test, prediction)->pd.DataFrame:

    print('def metrics')

    mae = mean_absolute_error(test, prediction)
    mse = mean_squared_error(test, prediction)
    r2 = r2_score(test, prediction)

    data = {'eval':['MAE','MSE','R2'], 'score':[mae,mse,r2]}
    results = pd.DataFrame(data)
    return results

def plot_arima_results(train, test, prediction, target_column):

    print('plot_arima')

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Test', color='green')
    plt.plot(test.index, prediction, label='Predicted', color='red', linestyle='dashed')
    plt.title(f"Optimized ARIMA Prediction ({target_column})")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.show();

def plot_TSMixer_results(predicted_prices_df, date, actual, predicted):

    print('plot TSMixer')

    plt.figure(figsize=(12, 6))
    plt.plot(predicted_prices_df[date], predicted_prices_df[actual], label='Actual Close Price')
    plt.plot(predicted_prices_df[date], predicted_prices_df[predicted], label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_predictions_LSTM(y_test, predictions, title="Prediction vs True Values"):

    print('plot LSTM')

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="True Values", color="blue")
    plt.plot(predictions, label="Predictions", color="orange")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.show()