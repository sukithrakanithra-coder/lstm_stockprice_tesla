# Tesla Stock Price Forecasting using LSTM

## Overview
LSTM model that predicts Tesla's stock closing price from historical data.

## Dataset
- Source: (Yahoo Finance / Kaggle, add link)
- Columns used: Open, High, Low, Close, Volume
- Period: (start date to end date)

## Preprocessing
- Handled missing values (dropna / fill)
- Selected the Close price as the feature
- Scaled data with MinMaxScaler (0 to 1)
- Created sequences of 60 past days to predict the next day
- Split into 80% train and 20% test

## Model Architecture
- LSTM (50 units, return_sequences=True)
- Dropout (0.2)
- LSTM (50 units)
- Dropout (0.2)
- Dense (1)
- Optimizer: Adam, Loss: MSE
- Epochs: __, Batch size: __

## How to Run
1. Clone the repo: `git clone <your-repo-link>`
2. Install packages: `pip install tensorflow pandas numpy matplotlib scikit-learn seaborn`
3. Open `your_file.ipynb` and run all cells

## Results
| Metric | Value |
|--------|-------|
| MAE | __ |
| RMSE | __ |
| MSE | __ |

The model follows the overall trend well but lags during sudden spikes.

![Predictions vs Actual](Screenshot%20(24).png)
![Loss Curve](Screenshot%20(25).png)

## Challenges
- Stock prices are volatile
- Overfitting risk, handled with Dropout

## Future Improvements
- Add more features (volume, moving averages)
- Try GRU or Bidirectional LSTM
- Tune hyperparameters
