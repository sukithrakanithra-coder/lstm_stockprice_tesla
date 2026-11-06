# Tesla Stock Price Forecasting using LSTM

## Overview
LSTM model that predicts Tesla's Open stock price from historical data. The model looks at the previous 60 days of Open prices to predict the next day's price.

## Dataset
- Source: [Tesla.csv](https://drive.google.com/file/d/1tMT6klV1ISO_VZxdbTuDnFjBhT6-369I/view) (Tesla historical stock prices)
- 1692 rows, 6 columns, indexed by Date
- Period: 29-06-2010 to 17-03-2017
- Only the Open price is used for prediction

## Preprocessing
- Checked for missing values (`df.isna()`)
- Selected the Open price as the feature
- Scaled data with MinMaxScaler (0 to 1)
- Created sequences of 60 past days to predict the next day
- Split into 75% train and 25% test

## Model Architecture
- LSTM (50 units, return_sequences=True)
- LSTM (64 units)
- Dense (32)
- Dense (16)
- Dense (1)
- Optimizer: Adam, Loss: MSE
- Epochs: 100 (with EarlyStopping, patience 10), Batch size: 32

## How to Run
1. Clone the repo: `git clone https://github.com/sukithrakanithra-coder/tesla_stockprice`
2. Install packages: `pip install tensorflow pandas numpy matplotlib scikit-learn seaborn`
3. Open `lstm_stockprice.ipynb` and run all cells

## Results
| Metric | Value |
|--------|-------|
| MAE | 4.91 |
| RMSE | 6.47 |
| MSE | 41.86 |

The model follows the overall price trend closely but lags during sudden spikes.

![Predictions vs Actual](Screenshot%20(24).png)
![Open Price History](Screenshot%20(25).png)

## Challenges
- Stock prices are volatile and hard to predict
- Choosing the sequence length and number of epochs

## Future Improvements
- Add Dropout layers to reduce overfitting
- Add more features (volume, moving averages)
- Try GRU or Bidirectional LSTM
- Tune hyperparameters
