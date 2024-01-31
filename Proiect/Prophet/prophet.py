from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from prophet import Prophet

import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np 
import sys

def get_timeseries(ticker):
    yfin = yf.Ticker(ticker)
    data = yfin.history(period="max")
    return data

def preprocess_data(data):
    data = data[['Close']]
    data.reset_index(level=0, inplace=True)
    data.rename({'Date': 'ds', 'Close': 'y'}, axis='columns', inplace=True)
    print(data.head())
    data['ds'] = data['ds'].dt.tz_localize(None)
    return data

def run_prophet(data, days):
    m = Prophet(changepoint_range=0.8, changepoint_prior_scale=0.1)
    m.add_country_holidays(country_name='US')

    m.fit(data)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    m.plot(forecast)
    plt.show()
    return forecast

def is_anomaly(error, uncertainty, factor):
    return np.abs(error) > factor * uncertainty

def detect_anomalies(forecast, data, factor):
    data_with_forecasts = pd.merge(forecast, data, how='inner', left_on = 'ds', right_on = 'ds')
 
    data_with_forecasts['error'] = data_with_forecasts['y'] - data_with_forecasts['yhat']
    data_with_forecasts['uncertainty'] = data_with_forecasts['yhat_upper'] - data_with_forecasts['yhat_lower']

    data_with_forecasts['anomaly'] = data_with_forecasts.apply(lambda x: is_anomaly(x['error'], x['uncertainty'], factor), axis = 1)
    
    color_discrete_map = {True: 'red', False: 'blue'}

    data_with_forecasts = data_with_forecasts[data_with_forecasts['ds'] > '2018-01-01']
    evaluate_model(data_with_forecasts)

    plt.figure(figsize=(10, 6))
    for category, color in color_discrete_map.items():
        filtered_data = data_with_forecasts[data_with_forecasts['anomaly'] == category]
        is_anom = 'Yes' if category else 'No'
        plt.scatter(filtered_data['ds'], filtered_data['y'], color=color, label=is_anom)

    plt.title("Anomaly (factor = {})".format(factor))
    plt.xlabel('ds')
    plt.ylabel('price')
    plt.legend()
    plt.show()

def evaluate_model(forecasting_final):
    MAE = mean_absolute_error(forecasting_final['yhat'],forecasting_final['y'])
    print('MAE = ' + str(np.round(MAE, 2)))

    MSE = mean_squared_error(forecasting_final['yhat'],forecasting_final['y'])
    print('MSE = ' + str(np.round(MSE, 2)))

    MAPE = mean_absolute_percentage_error(forecasting_final['yhat'],forecasting_final['y'])
    print('MAPE = ' + str(np.round(MAPE, 2)) + ' %')

if len(sys.argv) != 4:
    print("Usage: python prophet.py <ticker> <days> <factor>")
else:
    print(sys.argv)
    ticker = sys.argv[1]
    days = int(sys.argv[2])
    factor = float(sys.argv[3])

    data = get_timeseries(ticker)
    processed_data = preprocess_data(data)
    forecast = run_prophet(processed_data, days)
    detect_anomalies(forecast, processed_data, factor)
