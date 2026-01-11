# Energy Usage Prediction System

A deep learning-based system for predicting electricity consumption in campus rooms using CNN + BiLSTM neural networks.

## Features

- **Real-time Prediction**: Predict energy usage for specific hours or entire days
- **Multiple Models**: LSTM, CNN+BiLSTM, and traditional ML approaches
- **Interactive Web App**: Streamlit-based user interface
- **Carbon Footprint**: Calculate environmental impact
- **Room-specific**: Supports different room types and configurations

## Project Structure


├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Dataset storage
│   └── synthetic_campus_energy_2022_2025.csv
├── models/               # Trained models and scalers
│   ├── saved_cnn_bilstm_model/
│   ├── saved_lstm_model/
│   └── saved_lstm_model_improved/
├── src/                  # Source code
│   ├── model1.py         # LSTM model evaluation
│   ├── model2.py         # Model training scripts
│   ├── LSTM+CNN.py       # CNN+BiLSTM implementation
│   ├── prediction.py     # Prediction utilities
│   └── performance_metrics.py
├── notebooks/            # Jupyter notebooks (if any)
└── docs/                # Documentation


## Installation

1. Clone the repository:

git clone <repository-url>
cd energy-prediction-system


2. Create virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt


## Usage

### Run the Web Application

streamlit run app.py

### Model Training

python src/LSTM+CNN.py


### Model Evaluation

python src/model1.py


## Models

- **CNN+BiLSTM**: Primary model combining convolutional and bidirectional LSTM layers
- **LSTM**: Traditional LSTM approach for time series prediction
- **Traditional ML**: Baseline models for comparison

## Features Used

- Hour of day, day of week, weekend flag
- Occupancy levels
- Outdoor temperature and solar irradiance
- Room characteristics (area, type)
- Exam and event flags
- Lag features and rolling averages

##  Performance

The CNN+BiLSTM model achieves:
- MAE: ~X.XX kWh
- RMSE: ~X.XX kWh
- R²: ~0.XX
