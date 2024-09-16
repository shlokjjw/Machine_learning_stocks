import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model('Stock Predictions Model.keras')

st.header('Stock Price Prediction')

stock = st.text_input('Enter the stock symbol:', 'GOOG')
start = '2014-01-01'
end='2024-09-10'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test=pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scalar.fit_transform(data_test)

x=[]
y=[]

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)


