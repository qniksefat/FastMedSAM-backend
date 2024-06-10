# Use an NVIDIA CUDA base image
FROM paperspace/fastapi-app-base:2023-06-14

# Install Python 3.9, wget, and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    wget \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip for Python 3.9
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy the entire backend directory
COPY backend/ ./backend/

# Copy the inferences directory
COPY inferences/ ./inferences/
COPY segment_anything/ ./segment_anything/
COPY ./*.py ./
COPY ./lite_medsam.pth ./

# Copy the Dockerfile, setup.py, and other required files
COPY Dockerfile .
COPY setup.py .
COPY LICENSE .
COPY README.md .

# Install dependencies
RUN pip install --no-cache-dir .

# Expose the port FastAPI is running on
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

