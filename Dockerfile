# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install dependencies first (this caches the heavy downloads)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy ONLY the necessary directories and files
# (The .dockerignore file will make sure we don't copy the 1GB of junk)
COPY frontend/ ./frontend/
COPY src/ ./src/
COPY data/models/default_model.joblib ./data/models/
COPY data/models/thin_file_model.joblib ./data/models/
COPY data/models/final_ensemble.joblib ./data/models/
COPY interest_engine.py .
COPY serve.py .

# 5. Expose the port the app runs on
EXPOSE 8000

# 6. Run the application using Gunicorn for production
# We set a 120-second timeout because large models can take a moment to load
CMD gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 120 serve:app
