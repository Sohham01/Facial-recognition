#import libraries
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

#load the model
model = load_model("model/emotion_model.h5")
#emotion labels and meme paths
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
meme_paths = {'Angry': 'memes/angry.jpeg',
    'Disgust': 'memes/disgust.jpeg',
    'Fear': 'memes/fear.jpeg',
    'Happy': 'memes/happy.jpeg',
    'Sad': 'memes/sad.jpeg',
    'Surprise': 'memes/surprise.jpeg',
    'Neutral': 'memes/neutral.jpeg'}
# ==========================================================
# Variables for Emotion Tracking
# ==========================================================
# Stores the number of times each emotion is detected
emotion_count = {e: 0 for e in emotion_labels}
# Used to avoid repeatedly displaying the same meme
last_emotion = None
last_change_time = 0
# Delay (in seconds) before changing the displayed meme
DELAY = 2

# Initialize Face Detector and Webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Main Application Loop
while True:
    # Capture one frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
     # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Process every detected face
    for (x,y,w,h) in faces:
         # Extract and preprocess face region
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi.reshape(1,48,48,1) / 255.0
         # Predict facial emotion using the trained CNN
        preds = model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        # Update emotion frequency
        emotion_count[emotion] += 1
        # Draw bounding box and predicted emotion
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, emotion, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
        current_time = time.time()
        # Display meme only if emotion changes
        # and the delay period has passed
        if emotion != last_emotion and current_time - last_change_time > DELAY:
            meme = cv2.imread(meme_paths[emotion])
            if meme is not None:
                meme = cv2.resize(meme, (350,350))
                cv2.imshow("Emotion Meme", meme)
            last_emotion = emotion
            last_change_time = current_time
    #Live Emotion Frequency Chart
    chart = np.zeros((300,420,3), dtype=np.uint8)
    max_val = max(emotion_count.values()) + 1
    for i, emo in enumerate(emotion_labels):
        bar_height = int((emotion_count[emo] / max_val) * 250)
        cv2.rectangle(chart,
                      (i*60+20, 280),
                      (i*60+50, 280-bar_height),
                      (255,255,255), -1)
        cv2.putText(chart, emo[:3],
                    (i*60+15,295),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
    # Display application windows
    cv2.imshow("Emotion Frequency", chart)
    cv2.imshow("Webcam", frame)
    # Press 'Q' to exit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release Resources
cap.release()
cv2.destroyAllWindows()