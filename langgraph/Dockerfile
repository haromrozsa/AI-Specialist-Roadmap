FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with compatible versions
RUN pip install --no-cache-dir \
    langchain>=0.3.0 \
    langchain-core>=0.3.0 \
    langgraph>=0.2.0 \
    langchain-huggingface>=0.1.0 \
    transformers>=4.40.0 \
    torch \
    accelerate \
    sentencepiece

# Copy your script
COPY query_classifier.py .

# Run the script
CMD ["python", "query_classifier.py"]