Flood Forecasting with Deep Learning (Time Series Modeling)

This project focuses on flood (river discharge) forecasting using multiple deep learning models on multivariate time-series data.
I implemented and compared different architectures (ANN, CNN, LSTM, GRU, CNN-LSTM) using a consistent preprocessing + evaluation pipeline.

Models Implemented

Each model is implemented as a separate notebook:

ANN.ipynb → Feed-forward Neural Network (with hyperparameter tuning)

CNN.ipynb → 1D CNN for time-series regression

LSTM.ipynb → LSTM sequence model

GRU.ipynb → GRU sequence model

CNN_LSTM.ipynb → Hybrid CNN + LSTM model

CNN_GRU.ipynb → (Notebook present, but currently minimal/incomplete)

Dataset & Problem Setup

Input data is read from: final.csv

From the dataset, the project uses 8 input variables (df.columns[1:9]) as multivariate signals.

Data is scaled using MinMaxScaler.

Time-series is converted into a supervised learning format using a custom function (series_to_supervised) with different lag windows per feature (to capture delayed effects).

The target is constructed from the supervised frame (forecasting the next step output).

✅ To run locally: keep final.csv in the repository root (same folder as notebooks).

Workflow (Common Across Notebooks)

Load data (final.csv)

Drop missing values

Feature selection (8 columns)

Scaling (MinMaxScaler)

Lag feature engineering using series_to_supervised

Train / Validation / Test split

Train deep learning model

Evaluate using hydrology + error metrics

Plot predicted vs observed

Export results to CSV

Evaluation Metrics

Metrics are computed using hydroeval (commonly used in hydrology):

NSE (Nash–Sutcliffe Efficiency) — higher is better

RMSE — lower is better

MARE (Mean Absolute Relative Error) — lower is better

PBIAS (Percent Bias) — closer to 0 is better
