from fastapi import FastAPI, File, UploadFile, Request, HTTPException, BackgroundTasks, Form
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from fastapi.templating import Jinja2Templates
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle
import tpot2
import os

app = FastAPI()
UPLOAD_FOLDER = "tmp/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Function for preprocessing the data
def preprocess_data(X_train, X_test):
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    categorical_pipe = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numeric_pipe = Pipeline([('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipe, categorical_cols),
            ('num', numeric_pipe, X_train.columns.difference(categorical_cols))
        ])
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return X_train_transformed, X_test_transformed

# Route to render the upload page
@app.get("/")
async def upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# File upload endpoint
@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid file format. Only CSV is allowed."})
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    dataset = pd.read_csv(file_path)
    column_names = list(dataset.columns)
    return templates.TemplateResponse("index.html", {"request": request, "filepath": file_path, "filename": file.filename, "columns": column_names})

# API Endpoint to get column names from uploaded CSV
@app.get("/get_columns")
async def get_columns(file_path: str):
    try:
        dataset = pd.read_csv(file_path)
        return {"columns": list(dataset.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Asynchronous training function
def train_model_task(file_path, feature_columns, target_column):
    dataset = pd.read_csv(file_path)
    feature_columns = feature_columns.split(',')
    
    if target_column not in dataset.columns or any(col not in dataset.columns for col in feature_columns):
        raise ValueError("One or more selected columns are not in the dataset")

    features = dataset[feature_columns]
    target = dataset[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    X_train_transformed, X_test_transformed = preprocess_data(X_train, X_test)

    # Initialize and train TPOT estimator
    tpot = tpot2.TPOTEstimator(population_size=50, generations=5, scorers=['accuracy'], verbose=1, n_jobs=32, scorers_weights=[1], classification=True, early_stop=5)
    tpot.fit(X_train_transformed, y_train)
    predictions = tpot.predict(X_test_transformed)

    # Use sklearn's accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    model_type = str(tpot.fitted_pipeline_)
    clf_report = classification_report(y_test, predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_test, predictions).tolist()

    # Save the trained model
    model_path = os.path.join(UPLOAD_FOLDER, 'tpot_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(tpot.fitted_pipeline_, f)
    
    # Save model details for UI display
    results_path = os.path.join(UPLOAD_FOLDER, 'model_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            "accuracy": accuracy,
            "classification_report": clf_report,
            "confusion_matrix": conf_matrix,
            "model_type": model_type
        }, f)

# API Endpoint: Train the Model Asynchronously
@app.post("/train")
async def train_model(
    request: Request,
    file_path: str = Form(...),
    feature_columns: str = Form(...),
    target_column: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    background_tasks.add_task(train_model_task, file_path, feature_columns, target_column)
    return RedirectResponse(url="/results", status_code=303)

# API Endpoint: Display Model Results
@app.get("/results", response_class=HTMLResponse)
async def display_results(request: Request):
    results_path = os.path.join(UPLOAD_FOLDER, 'model_results.pkl')
    if not os.path.exists(results_path):
        return templates.TemplateResponse("results.html", {"request": request, "error": "Model results not found."})

    with open(results_path, 'rb') as f:
        model_results = pickle.load(f)

    return templates.TemplateResponse("results.html", {"request": request, "model_details": model_results})

# API Endpoint: Download Trained Model
@app.get("/download_model")
def download_model():
    model_path = os.path.join(UPLOAD_FOLDER, 'tpot_model.pkl')
    if os.path.exists(model_path):
        return FileResponse(model_path, filename="tpot_model.pkl", media_type="application/octet-stream")
    else:
        raise HTTPException(status_code=404, detail="Model file not found.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)