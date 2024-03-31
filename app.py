import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')

#%matplotlib inline

# Disable warning about passing a figure to st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)
# Define the stock symbols
stock_symbols = { 'Google':'GOOGL', 'Apple':'AAPL',
                  'Microsoft': 'MSFT', 
                 'Amazon': 'AMZN', 
                 'Facebook': 'FB', 
                 'Tesla': 'TSLA', 
                 'Alphabet': 'GOOG', 
                 'Netflix': 'NFLX', 
                 'Nvidia': 'NVDA', 
                 'Adobe': 'ADBE', 
                 'Intel': 'INTC', 
                 'PayPal': 'PYPL',
                'Johnson & Johnson': 'JNJ',
                 'Visa': 'V',
                 'JPMorgan Chase': 'JPM',
                 'Walmart': 'WMT',
                 'Procter & Gamble': 'PG',
                 'Bank of America': 'BAC',
                 'Mastercard': 'MA',
                 'Verizon': 'VZ',
                 'Coca-Cola': 'KO',
                 'Disney': 'DIS'}



# Function to download stock data
@st.cache_data
def load_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

# Function to visualize data
def visualize_data(data,key_):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Close Price')
    ax.set_title(f'{key_} Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    plt.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    from statsmodels.tsa.seasonal import STL
    decomposition = STL(data['Close'], period=12).fit()
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True,  figsize=(10,8))
    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observed')
    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Trend')
    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Seasonal')
    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')
    #plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)


def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str) -> list:

    total_len = train_len + horizon

    if method == 'mean':
        pred_mean = []

        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))

        return pred_mean

    elif method == 'last':
        pred_last_value = []

        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value

    elif method == 'ARIMA':
        pred_ARIMA = []

        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(2,1,3))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_ARIMA.extend(oos_pred)

        return pred_ARIMA

# Function for forecasting
def get_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # If the value is not found

# Main function to create the app
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Download and Visualize', 'Forecasting'])

    if page == 'Download and Visualize':
        st.title('Download and Visualize Data')
        selected_stock = st.sidebar.selectbox('Select stock symbol:', list(stock_symbols.items()), format_func=lambda x: x[0])
        stock_symbol = selected_stock[1]
        key_ = get_key(stock_symbols, stock_symbol)
        #stock_symbol = st.sidebar.text_input('Enter stock symbol (e.g., AAPL for Apple):')
        start_date = st.sidebar.date_input('Select start date:')
        end_date = st.sidebar.date_input('Select end date:')
        
        if st.sidebar.button('Download Data'):
            data = load_data(stock_symbol, start_date, end_date)
            visualize_data(data,key_)

    
        
        
            

    
        
    elif page == 'Forecasting':
        st.title('Forecasting')
        #stock_symbol = st.sidebar.text_input('Enter stock symbol (e.g., AAPL for Apple):')
        selected_symbol = st.sidebar.selectbox('Select stock symbol:', list(stock_symbols.items()), format_func=lambda x: x[0])
        stock_symbol = selected_symbol[1]
        start_date = st.sidebar.date_input('Select start date:')
        end_date = st.sidebar.date_input('Select end date:')
        
        
        
        if st.sidebar.button('Forecast'):
            with st.spinner("Forecasting in progress..."):
                df = yf.download(stock_symbol, '2014-01-01', end=end_date)
                del df['Open']
                del df['High']
                del df['Low']
                del df['Adj Close']
                del df['Volume']
                size = int(0.99*(len(df['Close'])))
                train = df[:size]
                test =  df[size:]
                #st.write(test.tail())
                #HORIZON = st.sidebar.number_input('Enter forecast horizon (in days):')
                TRAIN_LEN = len(train)
                HORIZON = len(test)
                WINDOW = 2


                pred_ARIMA = rolling_forecast(df["Close"], TRAIN_LEN, HORIZON, WINDOW, 'ARIMA')

                test.loc[:, 'pred_ARIMA'] = pred_ARIMA[:]
                st.write(test.tail())
                key_ = get_key(stock_symbols, stock_symbol)
                # Plot original data and forecasted values
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df['Close'][TRAIN_LEN:], label='Original Data')
                ax.plot(test['pred_ARIMA'], 'k--', label='ARIMA')
                ax.set_title(f'Forecast for {key_}')
                #ax.set_title(f'Forecast for {stock_symbol}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (USD)')
                ax.xticks(rotation=45) 
                ax.legend()
                st.pyplot(fig)
        
if __name__ == "__main__":
    main()
