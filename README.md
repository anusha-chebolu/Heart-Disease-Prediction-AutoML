# AutoML Heart Disease Prediction Web Application with MLOps

## Overview
This project presents a FastAPI-based web application designed to predict heart disease risk. The application harnesses AutoML capabilities through TPOT to automatically identify and optimize the best machine learning pipeline, achieving an impressive recall of 0.88. Furthermore, it leverages MLflow to streamline deployment and track model experiments, ensuring reproducibility and efficient MLOps practices.

## Features
- **FastAPI Web Application**: Provides a RESTful API for heart disease prediction.
- **AutoML with TPOT**: Automatically selects and tunes the optimal model pipeline, reaching a recall of 0.88.
- **MLflow Integration**: Logs experiments, model versions, and metrics for seamless tracking and reproducibility.
- **Streamlined Deployment**: Designed for easy integration into production environments.

## Getting Started

### Prerequisites
- Python 3.8+
- FastAPI
- TPOT
- MLflow
- Uvicorn (for running the FastAPI server)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/heart-disease-prediction.git
   cd heart-disease-prediction
