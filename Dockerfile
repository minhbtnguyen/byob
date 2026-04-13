FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG KAGGLE_API_TOKEN
RUN if [ -n "$KAGGLE_API_TOKEN" ]; then \
      KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN \
      python -c "import kagglehub; kagglehub.dataset_download('arashnic/microsoft-geolife-gps-trajectory-dataset')"; \
    fi

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableWebsocketCompression=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
