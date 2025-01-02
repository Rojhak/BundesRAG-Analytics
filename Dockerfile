FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download de_core_news_sm

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs vector_store

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 