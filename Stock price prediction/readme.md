# Stock Price Prediction Using LSTM

## Project Overview
This project predicts stock prices using a Long Short-Term Memory (LSTM) neural network. Historical stock data is fetched using the yfinance library, and the model is trained to forecast future prices based on past trends.

### Key Features:
1. Fetches historical stock prices for a specified ticker (e.g., AAPL).
2. Preprocesses data, including normalization and sequence creation.
3. Builds an LSTM-based neural network for time-series forecasting.
4. Evaluates the model's performance using RMSE on training and testing data.
5. Visualizes predictions alongside actual stock prices.

## Requirements
Install the required Python packages using:
```bash
pip install -r requirements.txt
