import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model = YOLO("fire_detector_v1.pt")

cap = cv2.VideoCapture(0)
# Try to bump up resolution hope it work 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Video recording setup for now 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from XVID to mp4v
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"fire_detection_{timestamp}.mp4"  # Changed to .mp4
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (1280, 720))

# Refined HSV Color Range for Bright Fire Detection
lower_fire = np.array([0, 100, 150])  # Red to Orange hues, high saturation, high brightness
upper_fire = np.array([30, 255, 255])  # Red to Yellow, high saturation and value

cv2.namedWindow("🔥 Fire Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🔥 Fire Detection", 1280, 720)

print(f"Recording started: {output_filename}")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLO Detection using my custom model 
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    
    # Color-based Detection for extra precision
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    
    # Debug windows
    hsv_display = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('HSV Display', hsv_display)
    cv2.imshow('Fire Mask', mask)
    
    # Find contours for color detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color_fire_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            # Draw VERY thick red rectangle around fire area for video visibility
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 8)  # Thick RED box
            # Add fire label with background for better visibility
            cv2.rectangle(annotated_frame, (x, y-40), (x + 300, y), (0, 0, 255), -1)  # Red background
            cv2.putText(annotated_frame, 'FIRE DETECTED HERE!', (x + 5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # Add corner markers
            cv2.circle(annotated_frame, (x, y), 8, (0, 0, 255), -1)  # Top-left corner
            cv2.circle(annotated_frame, (x+w, y), 8, (0, 0, 255), -1)  # Top-right corner  
            cv2.circle(annotated_frame, (x, y+h), 8, (0, 0, 255), -1)  # Bottom-left corner
            cv2.circle(annotated_frame, (x+w, y+h), 8, (0, 0, 255), -1)  # Bottom-right corner
            color_fire_detected = True
    
    # Check YOLO detections
    yolo_fire_detected = len(results[0].boxes) > 0 if results[0].boxes is not None else False
    
    # Display status with better visibility
    if color_fire_detected or yolo_fire_detected:
        # Big red circle indicator
        cv2.circle(annotated_frame, (30, 30), 20, (0, 0, 255), -1)  # Bigger red circle
        cv2.putText(annotated_frame, 'FIRE DETECTED!', (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Add warning text at top of screen
        cv2.putText(annotated_frame, '⚠️ FIRE ALERT ⚠️', (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        if color_fire_detected:
            print("🔥 Color fire detected at location!")
        if yolo_fire_detected:
            print("🔥 YOLO fire detected!")
    else:
        cv2.circle(annotated_frame, (30, 30), 20, (0, 255, 0), -1)  # Bigger green circle
        cv2.putText(annotated_frame, 'NO FIRE', (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Write the color frame to video file (not the mask!)
    out.write(annotated_frame)
    
    cv2.imshow('🔥 Fire Detection', annotated_frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved as: {output_filename}")