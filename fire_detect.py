import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# Load YOLO model
model = YOLO("fire_detector_v1.pt")

# Initialize Pi Camera 2
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (1280, 720)})
picam2.configure(camera_config)
picam2.start()

# HSV Color Range for Fire Detection
lower_fire = np.array([0, 100, 150])
upper_fire = np.array([30, 255, 255])

cv2.namedWindow("🔥 Fire Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🔥 Fire Detection", 1280, 720)

print("Press 'q' to quit")

while True:
    # Capture frame from Pi Camera
    frame = picam2.capture_array()

    # YOLO Detection
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Color-based Detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_fire_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 8)
            cv2.rectangle(annotated_frame, (x, y-40), (x + 300, y), (0, 0, 255), -1)
            cv2.putText(annotated_frame, 'FIRE DETECTED HERE!', (x + 5, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.circle(annotated_frame, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (x+w, y), 8, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (x, y+h), 8, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (x+w, y+h), 8, (0, 0, 255), -1)
            color_fire_detected = True

    # Check YOLO detections
    yolo_fire_detected = len(results[0].boxes) > 0 if results[0].boxes is not None else False

    # Display status
    if color_fire_detected or yolo_fire_detected:
        cv2.circle(annotated_frame, (30, 30), 20, (0, 0, 255), -1)
        cv2.putText(annotated_frame, 'FIRE DETECTED!', (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated_frame, '⚠️ FIRE ALERT ⚠️', (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        if color_fire_detected:
            print("🔥 Color fire detected at location!")
        if yolo_fire_detected:
            print("🔥 YOLO fire detected!")
    else:
        cv2.circle(annotated_frame, (30, 30), 20, (0, 255, 0), -1)
        cv2.putText(annotated_frame, 'NO FIRE', (70, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show window
    cv2.imshow("🔥 Fire Detection", annotated_frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
