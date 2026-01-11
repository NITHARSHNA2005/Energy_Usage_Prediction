# Project Structure Documentation

## Directory Overview

### Root Directory
- `app.py` - Main Streamlit web application
- `requirements.txt` - Python package dependencies
- `README.md` - Project documentation and setup guide
- `setup.sh` - Automated setup script for Unix/Linux/macOS
- `.gitignore` - Git ignore patterns

### `/data/`
Contains all datasets used for training and evaluation:
- `synthetic_campus_energy_2022_2025.csv` - Main energy consumption dataset

### `/models/`
Stores trained models and preprocessing scalers:
- `saved_cnn_bilstm_model/` - CNN+BiLSTM model artifacts
- `saved_lstm_model/` - Basic LSTM model artifacts  
- `saved_lstm_model_improved/` - Enhanced LSTM model artifacts

Each model directory contains:
- `*.h5` - Trained Keras model
- `scaler_X.pkl` - Input feature scaler
- `scaler_y.pkl` - Target variable scaler
- `*_performance.csv` - Model evaluation metrics

### `/src/`
Source code for model training, evaluation, and utilities:

#### Model Training Scripts
- `cnn_bilstm_training.py` - Train CNN+BiLSTM model
- `lstm_training.py` - Train basic LSTM model
- `lstm_improved_training.py` - Train enhanced LSTM model
- `traditional.py` - Traditional ML baseline models

#### Model Evaluation Scripts
- `cnn_bilstm_evaluation.py` - Evaluate CNN+BiLSTM performance
- `lstm_evaluation.py` - Evaluate LSTM performance
- `performance_metrics.py` - Metrics calculation utilities

#### Prediction Scripts
- `cnn_bilstm_prediction.py` - Make predictions with CNN+BiLSTM
- `prediction.py` - General prediction utilities

#### Application Scripts
- `streamlit_app.py` - Alternative Streamlit interface
- `run_app.py` - Application runner script
- `final.py` - Complete pipeline script

### `/notebooks/`
Jupyter notebooks for exploratory data analysis and experimentation (empty initially)

### `/docs/`
Additional documentation and guides (empty initially)

## Quick Start

1. **Setup Environment:**
  
   ./setup.sh
   

2. **Run Application:**
 
   source .venv/bin/activate
   streamlit run app.py
   

3. **Train New Models:**
   
   python src/cnn_bilstm_training.py
  

4. **Evaluate Models:**
  
   python src/cnn_bilstm_evaluation.py
   

## Model Pipeline

1. **Data Preprocessing** → Feature engineering and scaling
2. **Model Training** → CNN+BiLSTM, LSTM, or traditional ML
3. **Model Evaluation** → Performance metrics and validation
4. **Prediction** → Real-time energy usage forecasting
5. **Web Interface** → User-friendly Streamlit application