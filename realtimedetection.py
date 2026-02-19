import cv2
import numpy as np
from tensorflow.keras.models import load_model
model = load_model("emotion_model.h5")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0



webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit...")

while True:
    ret, frame = webcam.read()

    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))

        img = extract_features(face)
        pred = model.predict(img, verbose=0)
        prediction_label = labels[np.argmax(pred)]

       
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        
        cv2.putText(frame,
                    prediction_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2)

    cv2.imshow("Emotion Detector", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
