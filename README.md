# Stock-Price-with-LSTM-and-BERT


# Stock Price Prediction using LSTM and BERT

This project predicts stock prices by combining:

- **LSTM (Long Short-Term Memory)** networks for modeling historical stock prices
- **BERT (Bidirectional Encoder Representations from Transformers)** for extracting sentiment from financial news headlines

The model uses historical data from Yahoo Finance and mock news data (which can be replaced by real news API).

---

## 📌 Features
- Fetches historical stock data using `yfinance`
- Performs sentiment analysis using a pre-trained BERT model
- Combines sentiment scores with stock prices
- Trains an LSTM neural network on combined data
- Visualizes actual vs. predicted stock prices

---

## 🧠 Technologies Used
- Python
- LSTM (TensorFlow/Keras)
- BERT (HuggingFace Transformers)
- yFinance for stock data
- MinMaxScaler (sklearn)

---

## 📂 Project Structure
```
├── stock_prediction_lstm_bert.py  # Main pipeline
├── README.md                      # Project description
```

---

## ⚙️ Installation
```bash
pip install yfinance transformers scikit-learn pandas numpy matplotlib torch tensorflow
```

---

## 🚀 How to Run
```bash
python stock_prediction_lstm_bert.py
```

---

## 📰 News Data (Optional)
You can replace the mock news headlines with real headlines from NewsAPI or another news provider. Ensure the news is dated and matched correctly to stock trading dates.

---

## 📈 Output
The script plots a graph of:
- Actual stock prices
- Predicted prices using LSTM + Sentiment data

---

## 🔮 Future Improvements
- Integrate live news headlines using NewsAPI
- Use technical indicators (RSI, MACD) as additional features
- Hyperparameter tuning for LSTM
- Use a Transformer-based time-series model

---

## 🧑‍💻 Author
Arsalan Taassob

---

## 📜 License
MIT License

