# Flask Machine Learning Application

This repository contains a Flask application that allows users to upload datasets, preprocess them, train a machine learning model using TPOT, and download the trained model.

## Features

- Upload CSV datasets
- Preprocess data with OneHotEncoder and StandardScaler
- Train a model using TPOT
- Evaluate model accuracy
- Download the trained model

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone [repository URL]
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

1. Start the Flask application:
    ```bash
    python app.py
    ```
2. Navigate to http://localhost:5000/ in your web browser.
3. Upload a CSV file.
4. Specify feature and target columns for the model.
5. Train the model and view the results.
6. Download the trained model.

## Structure
- app.py: Main Flask application file.
- templates/: Folder containing HTML templates for the application.
- tmp/: Temporary folder for uploaded files and trained models.

## Dependencies
- Flask
- Pandas
- NumPy
- scikit-learn
- TPOT
- Werkzeug

## Note: 
When a file is uploaded and the feature columns along with the target column are specified, the model training process begins. This stage may appear to be unresponsive or 'stuck', but in reality, it's just that the model training requires a considerable amount of time to complete.
