# Real-Time Face Detection & Tracking System

A high-performance computer vision application built using Python and OpenCV for real-time face detection, facial feature recognition, and object tracking.

The system uses optimized Haar Cascade classifiers along with an adaptive low-light preprocessing pipeline to achieve robust detection performance while maintaining real-time throughput.

## Features

### Real-Time Face Detection

* Multi-scale Haar Cascade face detection
* Real-time webcam and video stream processing
* Persistent face tracking across frames
* Unique tracking IDs for detected faces

### Facial Feature Recognition

* Eye detection
* Smile detection
* Facial landmark visualization
* Bounding box annotations

### Low-Light Enhancement Pipeline

* CLAHE-based adaptive contrast enhancement
* Histogram equalization
* Noise reduction preprocessing
* Improved detection performance under poor lighting conditions

### Performance Monitoring

* Live FPS monitoring
* Throughput benchmarking
* Detection accuracy evaluation
* Processing latency analysis

---

## Technical Highlights

* Real-time processing at approximately 30 FPS
* Haar Cascade based facial detection
* Centroid-based object tracking
* Adaptive low-light enhancement
* Multi-thread ready architecture
* Unit-tested detection pipeline

---

## Technologies Used

* Python 3.10+
* OpenCV
* NumPy
* PyTest

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/real-time-face-detection.git

cd real-time-face-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Webcam Detection

```bash
python face_detector.py
```

### Video File Detection

```bash
python face_detector.py --source video.mp4
```

### Start with Low-Light Mode

```bash
python face_detector.py --low-light
```

### Disable Landmark Detection

```bash
python face_detector.py --no-landmarks
```

---

## Keyboard Controls

| Key | Action                    |
| --- | ------------------------- |
| Q   | Quit                      |
| L   | Toggle Low-Light Mode     |
| S   | Toggle Landmark Detection |
| F   | Toggle FPS Display        |
| R   | Reset Tracking            |
| P   | Pause / Resume            |

---

## Image Processing Mode

Process a single image:

```bash
python image_processor.py --image image.jpg
```

Process a folder of images:

```bash
python image_processor.py --folder images/
```

---

## Benchmarking

Run performance benchmarks:

```bash
python benchmark.py
```

Example Output:

```text
Mean FPS: 31.4
Detection Accuracy: 93%
Low-Light Accuracy: 91%
```

---

## Testing

Run unit tests:

```bash
pytest test_face_detector.py
```

---

## System Architecture

```text
Input Stream
      ↓
Low-Light Enhancement
      ↓
Grayscale Conversion
      ↓
Histogram Equalization
      ↓
Haar Cascade Detection
      ↓
Face Tracking
      ↓
Feature Detection
      ↓
Visualization Layer
```

---

## Results

| Metric                  | Performance |
| ----------------------- | ----------- |
| Throughput              | 30+ FPS     |
| Face Detection Accuracy | 93%         |
| Low-Light Accuracy      | 91%         |
| Tracking Latency        | Real-Time   |

---

## Future Improvements

* Deep Learning based detection (YOLOv8)
* Face recognition module
* GPU acceleration
* Multi-camera support
* Face mask detection
* Emotion recognition

---

## Author

Aryan Pawar

BITS Pilani

Computer Vision | Machine Learning | Software Engineering
