FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs Docker containers as UID 1000 - create a matching
# non-root user and hand it ownership so runtime writes (e.g. Streamlit's
# config/cache dir) don't hit permission errors.
RUN useradd -m -u 1000 user
COPY --chown=user . .
USER user
ENV HOME=/home/user

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableWebsocketCompression=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
