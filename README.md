# Tesla Stock Price Forecasting Using LSTM

## Overview
This project uses a Long Short-Term Memory (LSTM) deep learning model to predict Tesla's next-day Open stock price using historical stock market data.

The model uses the previous 60 days of Open prices to predict the next day's price.

## Dataset
- **Source:** Tesla.csv (included in this repository)
- **Rows:** 1692
- **Columns:** 6
- **Index:** Date
- **Period:** 29-06-2010 to 17-03-2017
- Only the Open price is used for prediction.

## Preprocessing
- Checked for missing values using `df.isna()`.
- Selected the Open price as the input feature.
- Split the dataset chronologically into 75% training data and 25% testing data.
- Applied MinMaxScaler (0 to 1), fitting the scaler only on training data to prevent data leakage.
- Created sequences using the previous 60 days to predict the next day's price.

## Model Architecture
- LSTM (50 units, return_sequences=True)
- LSTM (64 units)
- Dense (32 units)
- Dense (16 units)
- Dense (1 unit)

**Model Configuration:**
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Maximum Epochs: 100
- EarlyStopping: Patience of 10 epochs, monitoring validation loss
- Batch Size: 32

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/sukithrakanithra-coder/tesla_stockprice.git
   ```

2. Install the required packages:

   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn seaborn
   ```

3. Ensure `Tesla.csv` is in the same directory as the notebook.

4. Open `lstm_stockprice.ipynb` using Jupyter Notebook or JupyterLab.

5. Run all cells to train the model and generate predictions.

## Results

The model performance is evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

| Metric | Value |
|---|---:|
| MAE | 4.91 |
| RMSE | 6.47 |
| MSE | 41.86 |

The model follows the overall stock price trend closely but may lag during sudden price fluctuations.

*Note: These metrics should be updated after rerunning the corrected notebook.*

## Visualizations

### Predictions vs Actual Stock Prices
![Predictions vs Actual](Screenshot%20(24).png)

### Tesla Open Price History
![Open Price History](Screenshot%20(25).png)

## Challenges
- Stock prices are volatile and difficult to predict accurately.
- Selecting the appropriate sequence length.
- Choosing the optimal number of epochs.
- Preventing overfitting during model training.

## Future Improvements
- Add Dropout layers to reduce overfitting.
- Include additional features such as Volume and Moving Averages.
- Experiment with GRU and Bidirectional LSTM models.
- Perform hyperparameter tuning to improve prediction accuracy.
