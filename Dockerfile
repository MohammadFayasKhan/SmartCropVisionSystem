# Dockerfile for SmartCropVisionSystem
# Supports standard container runtime and Hugging Face Spaces Docker SDK

FROM python:3.10-slim

# Install system dependencies for OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# Expose default port 8000 for local deployment and 7860 for Hugging Face Spaces
EXPOSE 8000 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
