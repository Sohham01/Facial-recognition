#  Real-Time Facial Emotion Recognition with Meme Recommendation

A real-time facial emotion recognition system built using **TensorFlow**, **OpenCV**, and a **Convolutional Neural Network (CNN)** trained on the **FER-2013** dataset. The application detects facial expressions through a webcam, predicts the user's emotion, displays a corresponding meme, and visualizes emotion frequency in real time.

---

##  Features

-  Real-time face detection using OpenCV Haar Cascade
-  CNN-based facial emotion recognition
-  Detects 7 emotions:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
  -  Displays emotion-specific memes
  - Live emotion frequency bar chart
  - Lightweight and easy to run

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Dataset | FER-2013 |

---

##  Project Structure

```
Facial-Emotion-Recognition/
│
├── app.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── model/
│   └── emotion_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── memes/
    ├── angry.jpeg
    ├── disgust.jpeg
    ├── fear.jpeg
    ├── happy.jpeg
    ├── neutral.jpeg
    ├── sad.jpeg
    └── surprise.jpeg

```

---

##  Model Training

The CNN model was trained on the **FER-2013** dataset using **Kaggle GPU**.

The training notebook is available in:

```
notebooks/training.ipynb
```

The trained model is included in the repository:

```
model/emotion_model.h5
```

This allows users to run the application immediately without retraining the model.

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Facial-Emotion-Recognition.git
```

Move into the project directory

```bash
cd Facial-Emotion-Recognition
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Application

```bash
python app.py
```

A webcam window will open.

Press **Q** to exit the application.

---


## Workflow

```
FER-2013 Dataset
        │
        ▼
CNN Training (TensorFlow/Keras)
        │
        ▼
emotion_model.h5
        │
        ▼
Real-Time Webcam Feed
        │
        ▼
Face Detection
        │
        ▼
Emotion Prediction
        │
        ▼
Meme Recommendation
        │
        ▼
Live Emotion Statistics
```

---

##  Future Improvements

- Confidence score for predictions
- Multiple face detection
- Streamlit/Gradio web interface
- Emotion history dashboard
- Emotion logging to CSV
- Transfer learning using EfficientNet or ResNet
- Better face detector (MTCNN or MediaPipe)

---

##  Authors

**Sohham Choudhary**  
BS Data Science & Artificial Intelligence  
Indian Institute of Management Sambalpur

**Apurv Singh**  
BS Data Science & Artificial Intelligence  
Indian Institute of Management Sambalpur

---

