#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Stock Price Prediction using LSTM and BERT

import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# === 1. Download Stock Price ===
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']]
    return data

# === 2. Mock News Headlines (Replace with actual API for real use) ===
def get_news_data(dates):
    headlines = {
        str(date.date()): ["Company is doing great!", "Stock is going up."]
        for date in dates
    }
    return headlines

# === 3. BERT Sentiment Analysis ===
def get_sentiment_scores(news_dict):
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    sentiment_scores = []
    for date, headlines in news_dict.items():
        total_score = 0
        for headline in headlines:
            inputs = tokenizer.encode_plus(headline, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            score = torch.argmax(probs).item() + 1
            total_score += score
        avg_score = total_score / len(headlines)
        sentiment_scores.append((date, avg_score))

    return pd.DataFrame(sentiment_scores, columns=["Date", "Sentiment"]).set_index("Date")

# === 4. Combine Stock + Sentiment ===
def prepare_data(stock_data, sentiment_data):
    data = stock_data.copy()
    sentiment_data.index = pd.to_datetime(sentiment_data.index)
    data = data.join(sentiment_data, how='left')
    data['Sentiment'].fillna(method='ffill', inplace=True)
    return data

# === 5. LSTM Model Preparation ===
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

# === Main Pipeline ===
ticker = 'AAPL'
start = '2022-01-01'
end = '2023-01-01'

stock_data = get_stock_data(ticker, start, end)
dates = stock_data.index
news_data = get_news_data(dates)
sentiment_df = get_sentiment_scores(news_data)
full_data = prepare_data(stock_data, sentiment_df)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(full_data)

seq_len = 10
X, y = create_sequences(scaled_data, seq_len)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=16)

predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(np.concatenate((predicted, X[:, -1, 1:]), axis=1))[:, 0]

plt.plot(stock_data.index[seq_len:], full_data['Close'].values[seq_len:], label='Actual Price')
plt.plot(stock_data.index[seq_len:], predicted_prices, label='Predicted Price')
plt.legend()
plt.title(f'{ticker} Stock Price Prediction with BERT+LSTM')
plt.show()

