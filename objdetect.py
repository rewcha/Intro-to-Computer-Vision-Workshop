from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Convert to RGB for Matplotlib display
    plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.pause(0.001)   # Small pause to update frame (like real-time feed)
    plt.clf()          # Clear previous frame before showing the next one

    # Optional: break loop manually if needed
    # You can’t press 'q' here, so just stop with Ctrl+C
    # or set a fixed number of frames if preferred

cap.release()
plt.close()
