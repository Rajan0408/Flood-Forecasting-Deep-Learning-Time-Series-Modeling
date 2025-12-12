# Flood Forecasting with Deep Learning (Time Series Modeling)

This project focuses on **flood (river discharge) forecasting** using **deep learning models** on **multivariate time-series data**.  
Multiple neural network architectures are implemented and compared using a **consistent preprocessing, training, and evaluation pipeline**.

The objective is to analyze how different deep learning models perform on hydrological time-series forecasting tasks and identify the most effective architecture.

---

## 📌 Models Implemented

Each model is implemented as a **separate Jupyter Notebook** for clarity and fair comparison:

- **ANN.ipynb**  
  Feed-forward Artificial Neural Network with hyperparameter tuning

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
  *(Notebook present but currently minimal / incomplete)*

---

## 📊 Dataset & Problem Setup

- Input data file: **`final.csv`**
- Uses **8 input variables** from the dataset:


- Data preprocessing steps:
- Missing values removed
- Feature scaling using **MinMaxScaler**
- Conversion of time-series into supervised learning format using a custom  
  `series_to_supervised()` function
- Different lag windows are applied to capture **delayed hydrological effects**
- Target variable is constructed from the supervised frame to predict **next-step discharge**

> ✅ **To run locally:**  
> Keep `final.csv` in the repository root (same directory as the notebooks).

---

## 🔁 Common Workflow (Across All Notebooks)

Each notebook follows the same structured pipeline:

1. Load dataset (`final.csv`)
2. Drop missing values
3. Select input features (8 variables)
4. Scale data using `MinMaxScaler`
5. Generate lag features using `series_to_supervised`
6. Split data into Train / Validation / Test sets
7. Train deep learning model
8. Evaluate performance using hydrology-specific metrics
9. Plot **Observed vs Predicted** discharge
10. Export prediction results to CSV

---

## 📐 Evaluation Metrics

Model performance is evaluated using **hydrology-standard metrics** via the `hydroeval` library:

- **NSE (Nash–Sutcliffe Efficiency)**  
Higher values indicate better predictive performance

- **RMSE (Root Mean Squared Error)**  
Lower values indicate better accuracy

- **MARE (Mean Absolute Relative Error)**  
Lower values indicate better performance

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

## 🎯 Key Takeaways

- Provides a **fair comparison** of multiple deep learning architectures
- Demonstrates the impact of **temporal feature engineering**
- Uses **domain-specific hydrological evaluation metrics**
- Designed for **research reproducibility and model benchmarking**

---

## 📎 Notes

- The `CNN_GRU.ipynb` notebook is included for completeness but is currently not fully implemented.
- All other models are fully trained, evaluated, and visualized.

---

⭐ If you find this project useful, feel free to star the repository!
