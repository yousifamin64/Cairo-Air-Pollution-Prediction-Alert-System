# Cairo Air Pollution Prediction & Smart Energy Optimizer
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24-orange)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.101-green)](https://fastapi.tiangolo.com/)

This project predicts air pollution levels and optimizes energy consumption in Cairo using machine learning. It provides both an API and an interactive Streamlit dashboard for visualization. The project is structured for modularity and Docker deployment.
# Project Structure
Cairo-Air-Pollution-Prediction-Alert-System/
├─ assets/ # Screenshots and images
├─ data/
│  ├─ raw/                     # Raw input datasets
│  ├─ processed/               # Preprocessed 
│  └─ metrics/ # Model training metrics logs
features and predictions
│  └─ metrics/                 # Model training metrics logs
├─ models/                     # Saved ML models
├─ src/
│  ├─ scripts/
│  │  └─ api_app.py            # FastAPI app   with /, /predict, /train, /metrics endpoints
│  └─ gui/
│     └─ dashboard.py          # Streamlit interactive dashboard
├─ requirements.txt            # Python dependencies
└─ README.md                   # This file
# Setup
# Clone the repository:
```
git clone  https://github.com/yousifamin64/Cairo-Air-Pollution-Prediction-Alert-System

cd Cairo-Air-Pollution-Prediction-Alert-System
```
# Create and activate virtual environment:
```
python -m venv .venv
& .venv/Scripts/Activate.ps1      # Windows PowerShell
```
# 3. Install dependencies:
```
pip install -r requirements.txt
```
# Requirements (requirements.txt)
```
fastapi
uvicorn
pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
seaborn
plotly
```
# Running FastAPI API
1. Start the API server:
```
uvicorn src.scripts.api_app:app --host 0.0.0.0 --port 8000
```
# Endpoints:
- `/` : Health check
- `/predict` : Generate energy predictions from processed features
- `/train` : Train new model and save metrics
- `/metrics` : View training metrics logs
# Running Streamlit Dashboard
```
streamlit run src/gui/dashboard.py --server.port 8010
```
- Select chart type (line, bar, scatter, area) from sidebar
- Select feature to visualize
- View predictions and metrics interactively
# Docker Usage
1. Build Docker image:
```
docker-compose up --build

```
2. Run container:
```
docker run -p 8000:8000 cairo-energy
```
- The API runs inside Docker, accessible at `http://localhost:8000`
- For Streamlit, run inside container interactively:
```
docker run -it -p 8010:8010 cairo-energy /bin/bash
streamlit run src/gui/dashboard.py --server.port 8010
```
# Download Data & models
The datasets and trained models are too large to include in this repository.  
You can download them from the following links:
[Download Data $ Models](https://mega.nz/file/i5wUSCRb#unBUJ8sNOkCYkaLEZHP5NXc9Q-8TryvCTYkGm5OovGI)

After downloading, place the files in the following folders in your local repository:
data/raw/
data/processed/
models/
This ensures the code and dashboard can access the files correctly.

# Notes

All models are trained using historical Cairo energy and pollution datasets.

Metrics are logged automatically after every retrain.

Dashboard supports dynamic chart selection and metric visualization.

# Author

Yousif Amin
 
 Cairo, Egypt

Mechatronics Engineer & AI Developer

