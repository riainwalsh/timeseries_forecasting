
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras import layers, models

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    s = df["Close"].dropna()
    return s

def arima_forecast(series, order=(5,1,0), steps=5):
    model = ARIMA(series, order=order)
    res = model.fit()
    fc = res.forecast(steps=steps)
    return fc

def make_lstm_dataset(arr, lookback=30):
    X, y = [], []
    for i in range(len(arr)-lookback):
        X.append(arr[i:i+lookback])
        y.append(arr[i+lookback])
    X = np.array(X)[...,None]
    y = np.array(y)
    return X, y

def lstm_forecast(series, lookback=30, epochs=3, steps=5):
    arr = series.values.astype("float32")
    X, y = make_lstm_dataset(arr, lookback)
    model = models.Sequential([
        layers.Input(shape=(lookback,1)),
        layers.LSTM(32),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0)
    last = arr[-lookback:].reshape(1,lookback,1)
    preds = []
    cur = last.copy()
    for _ in range(steps):
        p = model.predict(cur, verbose=0)[0,0]
        preds.append(p)
        cur = np.roll(cur, -1, axis=1)
        cur[0,-1,0] = p
    return pd.Series(preds, index=pd.RangeIndex(len(series), len(series)+steps))

def main(ticker, start, end):
    s = load_data(ticker, start, end)
    arima_fc = arima_forecast(s, steps=5)
    lstm_fc = lstm_forecast(s, steps=5)
    print("ARIMA forecast:\n", arima_fc)
    print("LSTM forecast:\n", lstm_fc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2024-12-31")
    args = ap.parse_args()
    main(args.ticker, args.start, args.end)
