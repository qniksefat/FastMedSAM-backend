# Use the official Python base image
FROM python:3.9-slim

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

# Install build tools and development headers
RUN apt-get update && \
    apt-get install -y gcc python3-dev

# Install dependencies
RUN pip install --no-cache-dir .

# Expose the port FastAPI is running on
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]

