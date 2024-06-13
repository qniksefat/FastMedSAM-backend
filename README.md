# Fast MedSAM Documentation

<p align="center">
  <img src="static/fastmedsam.gif"
  alt="Web Interface" align="middle" width="85%">
</p>

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
   - [Backend](#backend)
   - [Frontend](#frontend)
3. [Installation Instructions](#installation-instructions)
   - [Backend](#backend-installation)
   - [Frontend](#frontend-installation)
4. [Usage](#usage)
   - [Backend Endpoints](#backend-endpoints)
   - [Frontend Interface](#frontend-interface)
5. [Deployment](#deployment)
   - [Backend](#backend-deployment)
   - [Frontend](#frontend-deployment)
6. [Logging](#logging)
7. [Model Improvements](#model-improvements)
8. [Future Work](#future-work)
   - [Backend](#backend-future-work)
   - [Frontend](#frontend-future-work)
9. [Conclusion](#conclusion)

## Overview

Fast MedSAM is a comprehensive solution for rapid medical image segmentation, combining a FastAPI backend with a Streamlit frontend. The project aims to provide an efficient and user-friendly interface for clinicians to upload medical images, perform segmentation using a custom MedSAM_Lite model, and download the segmented results. This documentation details the project's purpose, structure, methods, tools, and provides installation, usage, and contribution instructions.

> An interface for the Fast MedSAM project, built using Streamlit, is available at [Fast MedSAM Frontend](http
The GPU backend (off by default) is running on Paperspace on an NVIDIA P4000 GPU. The CPU backend is running on `172.177.216.138:8000` on Azure.

## Project Structure

### Backend

The backend is built using FastAPI and PyTorch, providing endpoints for uploading medical images and performing segmentation. It handles image preprocessing, model inference, and returns the segmented images.

#### Key Files

- `inference.py`: Contains the `Inference` class, which handles loading the model, preprocessing images, and performing inference.
- `model.py`: Contains the `MedSAM_Lite` class and related model components.
- `utils.py`: Contains utility functions for image processing.
- `app.py`: Sets up the FastAPI application, defines the API endpoints, and handles file uploads and segmentation requests.

For more detailed information, refer to the [backend readme](backend/readme.md).

### Frontend

The frontend is built using Streamlit, providing a web interface for users to interact with the backend services. It allows users to upload images, view original and segmented results, and download outcomes.

#### Key Files

- `streamlit_app.py`: Main Streamlit application file.
    - `View` class responsible for rendering the user interface components.
    - `Controller` class that manages the application logic and controls the flow of the application.

For more detailed information, refer to the [frontend readme](frontend/readme.md).

> **Note:** For ease of readability, I will also add the frontend directory to the backend repository and provide a single repository for the entire project. But you can track my activity on the repo.

## Installation Instructions

### Backend Installation

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

### Frontend Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/qniksefat/FastMedSAM-front.git
   cd FastMedSAM-front
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set Up Backend Endpoint:** if you do not run the backend locally at `http://localhost:8080`.
   - Create a `.streamlit` directory in the root of the project.
   - Create a `secrets.toml` file in the `.streamlit` directory.
   - Add your backend endpoint to the `secrets.toml` file:
     ```toml
     ENDPOINT = "http://your-backend-endpoint" 
     ```

4. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Backend Endpoints

#### Uploading an Image

- **Endpoint:** `/upload/`
- **Method:** `POST`
- **Description:** Uploads an `.npz` file containing the medical image.
- **Parameters:**
  - `file`: The image file to be uploaded.
- **Response:** Returns a JSON object containing the filename of the uploaded image.

#### Segmenting an Image

- **Endpoint:** `/segment/`
- **Method:** `GET`
- **Description:** Performs segmentation on the uploaded image.
- **Parameters:**
  - `filename`: The name of the file to be segmented.
- **Response:** Returns the segmented image as a binary stream.

### Frontend Interface

#### Uploading an Image

1. **Upload Section:** Users can upload an `.npz` file containing the medical image through the sidebar's "Upload Your Image" section.

2. **Sample Images:** Alternatively, users can select from predefined sample images provided in the "Or Choose a Sample Image" section.

#### Viewing Images

- **Original Image:** The original image is displayed in the main interface. Users can use a slider to select different slices of the image for visualization.

- **Segmented Image:** Once the image is segmented, the segmented image is displayed alongside the original image.

#### Segmentation and Download

- **Segment Image:** Users can start the segmentation process by clicking the "Segment Image" button in the sidebar.

- **Download Segmented Image:** After segmentation, users can download the segmented image using the "Download Segmented Image" button.

#### Refresh

- **Refresh Button:** Users can reset the application to try another image by clicking the "Refresh" button in the sidebar.

## Deployment

### Backend Deployment

#### Running in Development Mode

```bash
uvicorn backend.app:app --port 8080
```

### Docker

Two Dockerfiles are provided for deployment on different hardware:

- `Dockerfile`: For Nvidia GPU machines.
- `Dockerfile.cpu`: For CPU-only machines.

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

I have already pushed the GPU version of the Docker image to Docker Hub. 
Same for the CPU version is on `fastmedsamcpu`.
You can pull it directly using the following command:

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

### Frontend Deployment

The easiest way to deploy your Streamlit UI app is to go to [Streamlit Sharing](https://share.streamlit.io/) and follow the instructions to deploy your app. You should first fork the frontend repository to your GitHub account.

## Logging

The backend application uses Python 's logging module to record information, warnings, and errors during execution. Logs are written to the console for easy monitoring and debugging.

## Model Improvements

I have tried a few different approaches to improve inference speed, but they were not successful when running on single 3D images. I have tried `torch.jit.script`, `torch.quantization`, and `torch.autocast`. Here, for example, is a comparison between the original model and the amp model:

![compare processing times](static/amp_vs_original.png)

I have not tried pruning yet, but I think it could be a good approach to reduce the model size and improve inference speed.

## Future Work

### Backend

- **Testing Documentation:** To ensure the accuracy and usability of the documentation, it is essential to test the endpoints and verify that the expected responses are returned. This can be achieved through automated testing using tools like Pytest.

### Frontend

- **Advanced Visualization:** Adding more advanced visualization tools, such as 3D rendering by Plotly or VTK, to provide better insights into the segmented images.
- **User Authentication:** Implementing user authentication to secure the application and manage user-specific data.
- **Interactive Segmentation:** Streamlit supports inputting point cursors, so in the future, handling input points and one 2D image could be implemented for more interactive segmentation.

## Conclusion

Fast MedSAM provides a robust and efficient solution for medical image segmentation, combining a powerful backend with an intuitive frontend. The project is designed to be easily deployable and configurable, making it accessible for various use cases in medical imaging.
