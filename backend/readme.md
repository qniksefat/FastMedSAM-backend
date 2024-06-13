# Fast MedSAM Backend Documentation

## Overview

The Fast MedSAM backend is built using FastAPI and PyTorch. It provides endpoints for uploading medical images and performing segmentation using a custom MedSAM_Lite model. The backend handles image preprocessing, model inference, and returns the segmented images.

> The GPU backend (off by default) is running on Paperspace on an NVIDIA P4000 GPU. The CPU backend is running on `172.177.216.138:8000` on Azure.

## Project Structure

The backend codebase consists of several files organized to separate concerns and improve maintainability:

- `inference.py`: Contains the `Inference` class, which handles loading the model, preprocessing images, and performing inference.
- `model.py`: Contains the `MedSAM_Lite` class and related model components.
- `utils.py`: Contains utility functions for image processing.
- `app.py`: Sets up the FastAPI application, defines the API endpoints, and handles file uploads and segmentation requests.

## Model Improvements

I have tried a few different approaches to improve inference speed, but they were not successful when running on single 3D images. I have tried `torch.jit.script`, `torch.quantization`, and `torch.autocast`. Here, for example, is a comparison between the original model and the amp model:

![compare processing times](static/amp_vs_original.png)


## Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone -b backend-fast-api --single-branch https://github.com/qniksefat/FastMedSAM-backend
   cd FastMedSAM-backend
   ```

2. **Install Dependencies:**
   ```bash
   pip install -e .
   ```

3. **Run the Application:**
   ```bash
   uvicorn backend.app:app --port 8080
   ```

## Usage

### Uploading an Image

- **Endpoint:** `/upload/`
- **Method:** `POST`
- **Description:** Uploads an `.npz` file containing the medical image.
- **Parameters:**
  - `file`: The image file to be uploaded.
- **Response:** Returns a JSON object containing the filename of the uploaded image.

### Segmenting an Image

- **Endpoint:** `/segment/`
- **Method:** `GET`
- **Description:** Performs segmentation on the uploaded image.
- **Parameters:**
  - `filename`: The name of the file to be segmented.
- **Response:** Returns the segmented image as a binary stream.

## Explanation of Components

### Inference Class

#### Initialization

- Initializes the `Inference` class with the model checkpoint path, device, and bounding box shift value.

#### Model Loading

- Loads the MedSAM_Lite model from the checkpoint, prepares it for quantization, and moves it to the specified device.

> **Note:** Comparing the performance of the model improvements was super hard due to the lack of on premise GPU. I used Google Colab, AWS, Azure, and finally Paperspace by DigitalOcean to test the model. 

#### Image Preprocessing

- Resizes and normalizes the input image, pads it to the required size, and converts it to a tensor.

#### Inference

- Performs inference on the preprocessed image and returns the segmented mask.

#### File Processing

- Processes the input file, performs segmentation on each slice, and saves the segmented results.

### FastAPI Endpoints

#### Upload Endpoint

- **`/upload/`** Handles file uploads and stores them in a temporary directory.

#### Segment Endpoint

- **`/segment/`** Processes the uploaded file, performs segmentation, and returns the segmented image.

## Deployment

#### Running in Development Mode

```bash
uvicorn backend.app:app --port 8080
```

### Docker

Two Dockerfiles are provided for deployment on different hardware:

- `Dockerfile`: For Nvidia GPU machines.
- `DockerfileCPU`: For CPU-only machines.

#### Building Docker Image

1. **Navigate to the project directory:**
   ```bash
   cd FastMedSAM-backend
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t fastmedsam -f Dockerfile .
   ```

#### Pulling GPU Docker Image from Docker Hub

I have already pushed the GPU version of the Docker image to Docker Hub. You can pull it directly using the following command:

```bash
docker pull mrab72/anbar:fastmedsam
```

#### Running the Docker Container

```bash
docker run -p 8080:8000 fastmedsam
```

### Paperspace

You can now make use of either CPU or GPU-based images on any server. The application is running on Paperspace by DigitalOcean. Sign in to your Paperspace account and create a new Gradient deployment. Choose a machine type based on your requirements (CPU or GPU) and select the image from Docker Hub. You can then deploy the container and access the FastAPI backend.

> **Note:** The CPU version is currently running and the website is connected to it. To test the GPU version, please contact me to turn on the GPU server, as it is costly to keep running all the time.

## Logging

The application uses Python's logging module to record information, warnings, and errors during execution. Logs are written to the console for easy monitoring and debugging.

## Future Improvements

### Testing Documentation

To ensure the accuracy and usability of the documentation, it is essential to test the endpoints and verify that the expected responses are returned. This can be achieved through automated testing using tools like Pytest. Given the time constraints, this aspect was not covered in the current version.
