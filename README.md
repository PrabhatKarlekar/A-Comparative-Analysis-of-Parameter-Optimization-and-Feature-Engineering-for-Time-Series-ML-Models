# A Comparative Analysis of Parameter Optimization and Feature Engineering for Time-Series ML Models

This repository contains the implementation of the research paper titled "A Comparative Analysis of Parameter Optimization and Feature Engineering for Time-Series ML Models" by Prabhat Karlekar, Pranali Nikam, and Puneet Bakshi, conducted at the Centre for Development of Advanced Computing (C-DAC), Pune, India.

## Project Overview

This project presents a systematic analysis of hyperparameter optimization and feature engineering techniques for time-series machine learning models, specifically in the context of federated learning (FL). It investigates how configurations such as window size, batch size, learning rate, and model architecture affect forecasting performance across heterogeneous clients using models like **LSTM** and **GRU**.

## Key Contributions

- Comparative evaluation of **LSTM** and **GRU** architectures on real-world financial time-series data.
- Exploration of hyperparameters including window size, batch size, learning rate, dropout, and number of layers.
- Implementation of various **feature engineering strategies**, such as:
  - Lag features
  - Moving averages
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
- Experiments performed on **PARAM Siddhi-AI**, a high-performance computing cluster at C-DAC Pune.
- Comprehensive evaluation using **MSE, RMSE, MAE, R², MAPE**, and **PMAE**.


## Methodology

1. **Data Collection**  
   - Historical stock data collected from Yahoo Finance.
   - Cleaned, normalized, and split into train/validation/test sets.

2. **Modeling**  
   - LSTM and GRU models implemented using **TensorFlow/Keras**.
   - Experiments with different architectures and dropout configurations.

3. **Hyperparameter Tuning**  
   - Manual tuning of window size, learning rate, epochs, and batch size.
   - Use of **early stopping** to prevent overfitting.

4. **Evaluation Metrics**  
   - Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE), R² Score, MAPE, PMAE
  
## Key Findings

- Smaller window sizes (e.g., 60) often lead to better generalization in LSTM models.
- GRU models provide lightweight alternatives with similar or better performance in some cases.
- Feature engineering such as RSI and lag features significantly improves temporal pattern learning.
- Consistent preprocessing and parameter settings lead to better aggregation outcomes in federated settings.

## Requirements

- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas
- scikit-learn
- matplotlib, seaborn

## How To Run

python lstm.py
python GRU.py


## Acknowledgements

This research was supported by the Centre for Development of Advanced Computing (C-DAC) and the National Supercomputing Mission (NSM), Government of India, utilizing resources from PARAM Siddhi-AI.

