# Custom Airflow image with ML dependencies
FROM apache/airflow:2.8.1-python3.11

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Create outputs directory
RUN mkdir -p /opt/airflow/outputs
