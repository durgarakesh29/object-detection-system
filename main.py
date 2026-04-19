import cv2
import numpy as np
import pyttsx3
import time

print("Starting Object Detection...")

# Voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load model
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

# Start camera
cap = cv2.VideoCapture(0)

last_spoken = {}
cooldown = 3  # seconds

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    h, w = frame.shape[:2]

    # Convert to blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    current_time = time.time()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw box
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

            # Put label
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Speak with cooldown
            if label not in last_spoken or current_time - last_spoken[label] > cooldown:
                engine.say(label)
                engine.runAndWait()
                last_spoken[label] = current_time

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()