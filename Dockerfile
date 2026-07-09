# Base image with CUDA 11.8 + Python 3.11 — matches RunPod GPU environment
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace base model weights at build time
# so startup is faster at runtime (these don't change)
RUN python -c "from transformers import BertTokenizer, BertModel; \
    BertTokenizer.from_pretrained('bert-base-uncased'); \
    BertModel.from_pretrained('bert-base-uncased'); \
    print('BERT cached ✅')"

RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('xlm-roberta-base'); \
    AutoModel.from_pretrained('xlm-roberta-base'); \
    print('XLM-RoBERTa cached ✅')"

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
