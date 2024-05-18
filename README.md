# Face Recognition and Liveness Detection System

This project is a Flask-based web application that performs real-time face recognition and liveness detection using a pre-trained deep learning model. The system can identify known faces and determine if the detected face is real or fake.

## Features

- **Real-time Face Recognition:** Identifies known faces using encoded face data.
- **Liveness Detection:** Determines if the detected face is real or fake using a trained deep learning model.
- **Web Interface:** Displays the video feed with recognition and liveness detection results.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/face-recognition-liveness.git
    cd face-recognition-liveness
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the model and data:**
    - Place the liveness detection model (`liveness.model`) and label encoder (`label_encoder.pickle`) in the `model/liveness/` directory.
    - Place the face detector model (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) in the `model/face_detector/` directory.
    - Place the encoded face data (`encoded_faces.pickle`) in the `model/face_detector/` directory.

2. **Run the Flask application:**
    ```bash
    python app.py
    ```

3. **Open your browser and navigate to:**
    ```
    http://127.0.0.1:5000/
    ```

## Code Overview

- **`app.py`:** The main Flask application file that sets up the web server, video feed, and face recognition/liveness detection logic.
- **`templates/index.html`:** The HTML template for the web interface.

### Core Functions

- **`recognition_liveness(model_path, le_path, detector_folder, encodings, confidence=0.5)`:** Handles the face recognition and liveness detection process.
- **`index()`:** Renders the main web page.
- **`video_feed()`:** Provides the video feed with recognition and liveness detection results.

## Dependencies

- Flask
- TensorFlow
- OpenCV
- imutils
- face_recognition
- numpy

