# Importing necessary libraries
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import tpot2
from werkzeug.utils import secure_filename
import os
import sklearn
import pickle

# Initialize Flask app
app = Flask(__name__,template_folder='templates')

# Configuration for file upload
UPLOAD_FOLDER = 'tmp/'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Secret key for Flask session
app.secret_key = 'Anusha123'

# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function for preprocessing the data
def preprocess_data(X_train, X_test):
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    # Create pipelines for categorical and numerical feature processing
    categorical_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numeric_pipe = Pipeline([('scaler', StandardScaler())])

    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipe, categorical_cols),
            ('num', numeric_pipe, X_train.columns.difference(categorical_cols))
        ])

    # Apply the preprocessing to the training and test data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return X_train_transformed, X_test_transformed

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', filepath=filepath, filename=filename)
    return render_template('index.html', error="Invalid file type")

# Route for training the model
@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Extracting file path and column details from form data
        file_path = request.form['file_path']
        feature_columns = request.form['feature_columns'].split(',')
        target_column = request.form['target_column']

        # Read the dataset from the uploaded file
        dataset = pd.read_csv(file_path)
        if not all(column in dataset.columns for column in feature_columns):
            return render_template('index.html', error="Column not found in dataset")

        # Splitting dataset into features and target
        features = dataset[feature_columns]
        target = dataset[target_column]

        # Splitting data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Preprocess the data
        X_train_transformed, X_test_transformed = preprocess_data(X_train, X_test)

        # Initialize and train TPOT estimator
        tpot = tpot2.TPOTEstimator(population_size=50, generations=5, scorers=['accuracy'], verbose=1, n_jobs=32, scorers_weights=[1], classification=True, early_stop=5)
        tpot.fit(X_train_transformed, y_train)

        # Evaluate model accuracy
        scorer = sklearn.metrics.get_scorer('accuracy')
        accuracy = scorer(tpot, X_test_transformed, y_test)
        model_type = str(tpot.fitted_pipeline_)

        # Save model details in session
        session['model_details'] = {'accuracy': accuracy, 'model_type': model_type}

        # Save the trained model to a file
        model_path = os.path.join('tmp/', 'tpot_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(tpot.fitted_pipeline_, f)
        session['model_path'] = model_path

        # Redirect to the results page
        return redirect(url_for('display_results'))
    except Exception as e:
        # Handle exceptions and return error message
        return render_template('index.html', error=str(e))

# Route to display the model results
@app.route('/results', methods=['GET'])
def display_results():
    # Retrieve model details from session
    model_details = session.get('model_details', {})
    return render_template('results.html', model_details=model_details)

# Route to download the trained model
@app.route('/download_model', methods=['GET'])
def download_model():
    # Retrieve model path from session and send file if exists
    model_path = session.get('model_path', '')
    if model_path and os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return render_template('index.html', error="Model file not found.")

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
