FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Force the app directory into the Python path.
ENV PYTHONPATH=/app

# Set the Hugging Face home directory to a standard location
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# ---------------------------------------------------------
# Install system dependencies (Git, C++ Compilers, CMake)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# THE FIX: Tell Git to ignore ownership mismatches system-wide
RUN git config --system --add safe.directory '*'

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir huggingface_hub

# ---------------------------------------------------------
# BAKE THE MODEL INTO THE DOCKER IMAGE
# Using Python directly avoids all 'command not found' errors
# ---------------------------------------------------------
# ARG MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_NAME}')"


# PROOF OF STORAGE: This will print the downloaded model files in your Render build logs!
# RUN echo "=== PROOF OF MODEL IN STORAGE ===" && \
#     ls -lh /root/.cache/huggingface/hub/
    
COPY . /app
RUN chmod +x /app/start.sh

EXPOSE 7860

CMD ["/app/start.sh"]