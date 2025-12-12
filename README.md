# Flood Forecasting with Deep Learning (Time Series Modeling)

**Deep learning–based flood (river discharge) forecasting using multivariate time-series data.**  
This repository implements, tunes, and compares multiple deep learning architectures using a **consistent preprocessing, training, and evaluation pipeline**, with a strong focus on **hydrology-specific performance metrics**.

✔ Temporal feature engineering with lagged variables  
✔ Automated hyperparameter search using Python scripts  
✔ ANN, CNN, LSTM, GRU, CNN–LSTM model comparison  
✔ NSE, RMSE, MARE, PBIAS evaluation (Hydrology standard)

---

## 📌 Project Overview (Quick Summary for Recruiters)

- Built and benchmarked multiple deep learning models for flood forecasting  
- Converted raw hydrological time-series into supervised learning format  
- Automated hyperparameter tuning using GridSearch-style Python scripts  
- Evaluated models using domain-specific hydrological metrics  
- Identified CNN–LSTM as the most effective architecture for the dataset  

---

## 🧠 Models Implemented

Each model is implemented as an **independent Jupyter Notebook** to ensure fair comparison and reproducibility:

- **ANN.ipynb**  
  Feed-forward Artificial Neural Network (ANN)

- **CNN.ipynb**  
  1D Convolutional Neural Network for time-series regression

- **LSTM.ipynb**  
  Long Short-Term Memory (LSTM) sequence model

- **GRU.ipynb**  
  Gated Recurrent Unit (GRU) sequence model

- **CNN_LSTM.ipynb**  
  Hybrid CNN + LSTM model for spatial–temporal feature learning

- **CNN_GRU.ipynb**  
  Hybrid CNN + GRU model  
  *(Present for completeness; currently minimal / exploratory)*

---

## 📊 Dataset & Problem Setup

- Dataset file: **`final.csv`**
- Uses **8 input variables**:



- Preprocessing steps:
- Missing values removed
- Feature scaling using **MinMaxScaler**
- Time-series converted to supervised learning format
- Variable-specific lag windows applied to capture delayed hydrological effects
- Target:
- **Next-step river discharge forecasting**

> ✅ **To run locally:**  
> Keep `final.csv` in the repository root (same folder as notebooks).

---

## 🔁 Common Workflow (Across All Models)

1. Load dataset (`final.csv`)
2. Drop missing values
3. Select input features (8 variables)
4. Scale data using MinMaxScaler
5. Generate lag features using `series_to_supervised`
6. Split data into Train / Validation / Test sets
7. Train deep learning model
8. Evaluate performance
9. Plot Observed vs Predicted discharge
10. Export predictions and metrics to CSV

---

## ⚙️ Automated Hyperparameter Search (Important)

The folder **`automated_py_scripts/`** contains Python scripts used to **automatically explore the best hyperparameter combinations** before final model evaluation.

These scripts:
- Perform GridSearch-style experimentation
- Iterate over:
- Lag window sizes
- Number of hidden units
- Network depth
- Batch sizes
- Epochs
- Optimizers and activation functions
- Identify the **best-performing configurations** based on validation metrics

📁 Example scripts:
- `ann_3.py`
- `cnn_2.py`
- `lstm_3.py`
- `gru_3.py`
- `cnn_lstm_3.py`
- `cnn_gru_3.py`

➡️ Results from these scripts were used to finalize model architectures in the notebooks.

---

## 📐 Evaluation Metrics

Model performance is evaluated using **hydrology-standard metrics** via the `hydroeval` library:

- **NSE (Nash–Sutcliffe Efficiency)**  
Higher values indicate better performance

- **RMSE (Root Mean Squared Error)**  
Lower values indicate better accuracy

- **MARE (Mean Absolute Relative Error)**  
Lower values indicate better prediction quality

- **PBIAS (Percent Bias)**  
Values closer to **0** indicate minimal bias

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- HydroEval
- Matplotlib

---

