# 🎭 Real-Time Face Emotion Detection

A real-time Face Emotion Detection system built using **TensorFlow** and **OpenCV**.  
This project detects human faces from webcam input and predicts emotions using a trained **Convolutional Neural Network (CNN)** model.

---

## 📷 Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/8345f156-a954-4c2c-bf26-2ec907a6ee89" width="600"/>
</p>

---

## 🚀 Features

- Real-time webcam face detection
- Emotion classification using CNN
- 7 emotion categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- Live emotion label overlay
- Press `q` to exit webcam

---

## 🛠️ Tech Stack

- Python
- TensorFlow (Keras API)
- OpenCV
- NumPy
- Scikit-learn

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/404-ERROR-Striver/face-emotion-detection.git
```

### 2️⃣ Navigate to project folder

```bash
cd face-emotion-detection
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python realtimedetection.py
```

The webcam will open automatically.  
Press **`q`** to close the application.

---

## 📂 Project Structure

```
face-emotion-detection/
│
├── realtimedetection.py
├── emotion_model.h5
├── requirements.txt
└── README.md
```

---

## 🧠 Model Details

- Model Type: Convolutional Neural Network (CNN)
- Input Size: 48x48 grayscale images
- Output Classes: 7 emotions
- Face Detection: Haar Cascade (OpenCV)

---

## 🐍 Python Version

Recommended: **Python 3.10**

---

## 📈 Future Improvements

- Deploy as a web app using Streamlit
- Improve model accuracy
- Add emotion analytics dashboard
- Support multiple face detection

---
