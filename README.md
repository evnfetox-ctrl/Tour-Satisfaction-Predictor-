# Travel Satisfaction Predictor & Fare Guide

A Streamlit web application that predicts travel satisfaction and provides fare guidance using a machine learning model.

## Features

- Predict travel satisfaction based on trip details
- Interactive web interface built with Streamlit
- Uses KNN (K-Nearest Neighbors) machine learning model
- Input parameters include age, gender, destination, transport mode, costs, etc.

## Installation

1. Clone or download the project files.

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the virtual environment is activated.

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501`

4. Enter your trip details in the sidebar and click "Predict Satisfaction" to get the prediction.

## Files

- `app.py`: Main Streamlit application
- `knn_model_pipeline.joblib`: Trained KNN model pipeline
- `df_1.csv`, `df_2.csv`, `df_3.csv`: Sample data files
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Model Details

The application uses a KNN classifier trained on travel data to predict satisfaction levels. The model includes preprocessing steps like scaling and encoding.

## Requirements

- Python 3.8+
- See `requirements.txt` for specific package versions

## License

[Add license information if applicable]