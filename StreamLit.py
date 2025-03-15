import streamlit as st
import pandas as pd
import datetime
import numpy as np
import yfinance as yf
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()

# #get hsi components
# import get_hsi
# get_hsi.get_hsi_components()

# get hang seng index components fr csv file
df = pd.read_csv('hsi_components.csv')
df.columns = ['Security','Symbol']


# define function for downloading data
@st.cache
def yf_download(ticker, start_date='2020-12-31', end_date='2025-03-15', interval='1d'):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, multi_level_index=False, interval=interval)
    df['rtn'] = df.Close.pct_change()
    df['log_rtn'] = np.log(df.Close/df.Close.shift(1))
    df.dropna(inplace=True)
    # Format only rtn and log_rtn columns as percentages
    return df


# define function for storring csv
@st.cache
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')


# define the part of the sidebar used for selecting the ticker and the time period
st.sidebar.header('Select Ticker and Time Period')

available_tickers = df['Symbol'].tolist()
tickers_companies_dict = dict(zip(df['Symbol'], df['Security']))

ticker = st.sidebar.selectbox('Select Ticker', available_tickers, format_func=lambda x: tickers_companies_dict[x])
start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 12, 31))
end_date = st.sidebar.date_input('End Date', datetime.date.today().strftime('%Y-%m-%d'))
if start_date > end_date or end_date > datetime.date.today():
    st.sidebar.error('Start date must be before end date')

# Add volume flag
st.sidebar.header('Technical Analysis Parameters')
volume_flag = st.sidebar.checkbox(label='Add Volume')

# Add expander with SMA
exp_sma = st.sidebar.expander('SMA')
sma_flag = exp_sma.checkbox(label='Add SMA')
sma_period = exp_sma.slider('SMA Period', min_value=1, max_value=50, value=20, step=1)

# Add expander with Bollinger Bands
exp_bb = st.sidebar.expander('Bollinger Bands')
bb_flag = exp_bb.checkbox(label='Add Bollinger Bands')
bb_period = exp_bb.slider('Bollinger Bands Period', min_value=1, max_value=50, value=20, step=1)
bb_std = exp_bb.slider('Bollinger Bands Std', min_value=1, max_value=4, value=2, step=1)

#add expander with RSI
exp_rsi = st.sidebar.expander('RSI')
rsi_flag = exp_rsi.checkbox(label='Add RSI')
rsi_period = exp_rsi.slider('RSI Period', min_value=1, max_value=28, value=14, step=1)
rsi_upper = exp_rsi.slider('RSI Upper', min_value=0, max_value=100, value=70, step=1)
rsi_lower = exp_rsi.slider('RSI Lower', min_value=0, max_value=100, value=30, step=1)

#add title and text
st.title('HSI Components')
st.text('Data Source: Yahoo Finance')
st.write('You can select HSI components')

# download the data from yahoo finance
df = yf_download(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

# add expander with preview data
data_exp = st.expander('Preview Data')
available_col = df.columns.tolist()
columns_to_show = data_exp.multiselect('Select Columns to Show', available_col, default=available_col)
data_exp.dataframe(df[columns_to_show])

csv_file = convert_df_to_csv(df)
data_exp.download_button(label='Download Data as CSV', data=csv_file, file_name=ticker + '_data.csv', mime='text/csv')

# create candlestick chart with the selected tenical analysis parameters
title_str = f'{tickers_companies_dict[ticker]}\'s price'
qf = cf.QuantFig(df, title=title_str, legend='top', name=ticker)

if volume_flag:
    qf.add_volume(up_color='orange', down_color='blue')

if sma_flag:
    qf.add_sma(sma_period)

if bb_flag:
    qf.add_bollinger_bands(bb_period, bb_std)

if rsi_flag:
    qf.add_rsi(rsi_period, rsi_upper, rsi_lower)

fig = qf.iplot(asFigure=True)

st.plotly_chart(fig)

