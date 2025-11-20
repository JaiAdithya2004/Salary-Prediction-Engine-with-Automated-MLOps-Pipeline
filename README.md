# 💼 Salary Prediction Engine with MLOps Automation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

An end-to-end machine learning system for predicting employee salaries with automated retraining, drift detection, and real-time monitoring capabilities. Built with production-ready features including REST API, interactive UI, and automated pipeline orchestration.

## 🌟 Key Features

- **Dual Interface**: RESTful API (FastAPI) + Interactive Web UI (Streamlit)
- **Automated ML Pipeline**: Drift detection → Data merge → Retraining → Evaluation
- **Intelligent Monitoring**: Statistical drift detection with email notifications
- **Production Ready**: Docker deployment, health checks, batch processing
- **Robust Processing**: Multi-encoding CSV support, automatic data cleaning
- **Real-time Predictions**: Single and batch inference endpoints

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Automated Pipeline](#-automated-pipeline)
- [Configuration](#-configuration)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)


## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   New Data      │────▶│  Drift Detection │────▶│  Retraining     │
│   (new_data/)   │     │  (KS Test + TVD) │     │  Pipeline       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                          │
                                ▼                          ▼
                        ┌──────────────┐         ┌─────────────────┐
                        │ Email Alerts │         │   model.pkl     │
                        └──────────────┘         │   metrics.json  │
                                                 └─────────────────┘
                                                          │
                        ┌─────────────────────────────────┴──────┐
                        │                                        │
                        ▼                                        ▼
                ┌───────────────┐                       ┌──────────────┐
                │  FastAPI      │◀──────────────────────│  Streamlit   │
                │  REST API     │                       │  Web UI      │
                └───────────────┘                       └──────────────┘
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core ML** | scikit-learn, NumPy, Pandas, SciPy |
| **API & Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Deployment** | Docker, Docker Compose |
| **Monitoring** | Custom drift detection (KS Test, TVD) |
| **Notifications** | SMTP email alerts |
| **Visualization** | Matplotlib, Seaborn |

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip or conda
- (Optional) Docker for containerized deployment

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/salary-prediction-ml-platform.git
cd salary-prediction-ml-platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file for email configuration (optional)
cp .env.example .env
# Edit .env with your SMTP credentials
```

### Train Initial Model

```bash
python -m src.train_model
```

Expected output:
```
✓ Model trained successfully
✓ Saved to: model.pkl
✓ Metrics: MAE=5243.21, R²=0.89, RMSE=6891.45
```

## 📖 Usage

### 1. Run FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access API documentation: `http://localhost:8000/docs`

### 2. Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

Access web interface: `http://localhost:8501`

### 3. Automated Retraining Pipeline

```bash
# Place new data in new_data/latest.csv
python automated_pipeline.py
```

The pipeline will:
1. ✅ Detect data drift
2. ✅ Merge new data with historical dataset
3. ✅ Retrain model automatically
4. ✅ Evaluate performance
5. ✅ Send email notifications

### 4. Continuous Monitoring

```bash
python -m src.drift_detector --monitor
```

Monitors `new_data/` directory for new CSV files and triggers drift detection automatically.

## 🔌 API Documentation

### Endpoints

#### `GET /`
Returns API status and available endpoints.

#### `GET /health`
Health check endpoint with model availability status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-20T10:30:00"
}
```

#### `POST /predict`
Single salary prediction.

**Request Body:**
```json
{
  "Age": 35,
  "Gender": "Male",
  "Education_Level": "Bachelor's",
  "Job_Title": "Software Engineer",
  "Years_of_Experience": 8.5
}
```

**Response:**
```json
{
  "predicted_salary": 95000.50,
  "confidence": "high",
  "model_version": "1.0"
}
```

#### `POST /predict/batch`
Batch predictions for multiple records.

**Request Body:**
```json
{
  "records": [
    {
      "Age": 35,
      "Gender": "Male",
      "Education_Level": "Bachelor's",
      "Job_Title": "Software Engineer",
      "Years_of_Experience": 8.5
    }
  ]
}
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "Age": 30,
        "Gender": "Female",
        "Education_Level": "Master's",
        "Job_Title": "Data Scientist",
        "Years_of_Experience": 5.0
    }
)
print(response.json())

# Batch prediction
with open("batch_data.csv", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/batch",
        files={"file": f}
    )
print(response.json())
```

## 🔄 Automated Pipeline

The automated pipeline (`automated_pipeline.py`) orchestrates the complete ML lifecycle:

```
┌─────────────────────────────────────────────────────────┐
│  1. Drift Detection                                     │
│     → KS Test for numerical features                    │
│     → TVD for categorical features                      │
│     → Threshold: p-value < 0.05, TVD > 0.2              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  2. Data Preprocessing & Merge                          │
│     → Clean new data (remove duplicates, handle nulls)  │
│     → Merge with historical dataset                     │
│     → Normalize column names                            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  3. Model Training                                      │
│     → RandomForest (200 estimators, max_depth=15)       │
│     → Train-test split (80/20)                          │
│     → Save to model.pkl                                 │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  4. Evaluation & Persistence                            │
│     → Calculate MAE, RMSE, R²                           │
│     → Save metrics to metrics.json                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  5. Notifications                                       │
│     → Email alerts on drift/training status             │
│     → HTML formatted reports                            │
└─────────────────────────────────────────────────────────┘
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Email Configuration (for notifications)
EMAIL_FROM=your-email@gmail.com
EMAIL_PASSWORD=your-app-specific-password
EMAIL_TO=recipient@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Notification Settings
SEND_TRAINING_EMAIL=true
NOTIFY_ON_NO_DRIFT=false

# API Configuration
API_URL=http://localhost:8000
```

### Gmail Setup (for notifications)

1. Enable 2-factor authentication on your Google account
2. Generate an [App Password](https://myaccount.google.com/apppasswords)
3. Use the app password in `EMAIL_PASSWORD`

### Model Configuration

Edit `src/train_model.py` to customize:

```python
# Model hyperparameters
model = RandomForestRegressor(
    n_estimators=200,      # Number of trees
    max_depth=15,          # Maximum tree depth
    min_samples_split=5,   # Min samples to split node
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop all services
docker-compose down
```

Services will be available at:
- **API**: http://localhost:8000
- **Streamlit UI**: http://localhost:8501

### Individual Containers

```bash
# Build API image
docker build -t salary-predictor-api -f Dockerfile .

# Run API container
docker run -p 8000:8000 salary-predictor-api

# Build Streamlit image
docker build -t salary-predictor-ui -f Dockerfile.streamlit .

# Run Streamlit container
docker run -p 8501:8501 salary-predictor-ui
```

## 📁 Project Structure

```
salary-prediction-ml-platform/
├── app.py                      # FastAPI application
├── streamlit_app.py            # Streamlit web interface
├── automated_pipeline.py       # ML pipeline orchestrator
├── requirements.txt            # Python dependencies
├── Dockerfile                  # API container definition
├── Dockerfile.streamlit        # UI container definition
├── docker-compose.yml          # Multi-container orchestration
├── render.yaml                 # Render.com deployment config
├── .env.example                # Environment variables template
├── salary_data.csv             # Historical training data
├── model.pkl                   # Trained model (generated)
├── metrics.json                # Evaluation metrics (generated)
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── train_model.py          # Training pipeline
│   ├── preprocess.py           # Data preprocessing
│   ├── evaluate.py             # Model evaluation
│   ├── drift_detector.py       # Data drift detection
│   ├── notify.py               # Email notifications
│   └── utils.py                # Utility functions
│
├── new_data/                   # Drop new CSV files here
│   └── latest.csv
│
└── plots/                      # EDA visualizations
    ├── correlation_matrix.png
    ├── salary_distribution.png
    └── feature_importance.png
```

## 🎯 Model Details

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Employee age (18-65) |
| Gender | Categorical | Male/Female/Other |
| Education_Level | Categorical | High School/Bachelor's/Master's/PhD |
| Job_Title | Categorical | Job position title |
| Years_of_Experience | Numeric | Years of work experience (0-40) |

### Target

- **Salary**: Annual salary in USD (continuous variable)

### Preprocessing Pipeline

1. **Column Normalization**: Standardize column names (remove BOM, spaces → underscores)
2. **Categorical Encoding**: OneHotEncoder with unknown handling
3. **Numeric Scaling**: StandardScaler for numeric features
4. **Missing Value Handling**: Drop rows with missing target, optional imputation for features

### Model Performance

Current model metrics (example):
```json
{
  "mae": 5243.21,
  "r2": 0.8934,
  "rmse": 6891.45
}
```

### Drift Detection Thresholds

- **Numerical Features**: Kolmogorov-Smirnov test, p-value < 0.05
- **Categorical Features**: Total Variation Distance (TVD) > 0.2



### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests (coming soon)
pytest tests/

# Format code
black src/ app.py streamlit_app.py

# Lint code
flake8 src/ app.py streamlit_app.py
```


<img width="1911" height="1051" alt="Screenshot 2025-11-20 220436" src="https://github.com/user-attachments/assets/df233910-4891-4234-a2bd-e540ceb39da6" />





<img width="1909" height="1034" alt="Screenshot 2025-11-20 220445" src="https://github.com/user-attachments/assets/529d01ed-f1e0-4ca1-a97e-aa74f9198722" />



