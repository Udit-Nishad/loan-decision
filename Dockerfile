# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies including git-lfs to resolve large model files
RUN apt-get update && apt-get install -y git git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ONLY the necessary directories and files
COPY frontend/ ./frontend/
COPY src/ ./src/
COPY data/models/default_model.joblib ./data/models/
COPY data/models/thin_file_model.joblib ./data/models/
COPY data/models/final_ensemble.joblib ./data/models/
COPY data/models/feature_cols.joblib ./data/models/
COPY interest_engine.py .
COPY serve.py .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Run the application using Gunicorn for production
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 120 --workers 1 serve:app
