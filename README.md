# Algo-Trading-ML-Model

The goal of this project is to identify optimal buy, sell, and hold points for any stock through a trained machine learning model. 

The main class - `MLAlgoStrat.py` - loads the daily price data of a stock from a specified start date and end date (see `SPYMLDataset.csv`) and includes values from various technical indicators such as the moving averages, exponential moving averages, RSI, and more. 

In order to train the machine learning model, I manually identified buy, sell, and hold points for SPY (see `SPYClassifiedBuyPoints.csv`) and fed the data into a Random Forest model.  

The program can also be used to test user-defined algorithmic trading strategies, and view each trade and its performance metrics.
