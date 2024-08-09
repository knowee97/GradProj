import time
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# Load YOLO model
model_path = r"C:\Users\12108\fiftyone\detect\MODEL\yolov8sModel_ncnn_model"
model = YOLO(model_path)

# Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# FPS calculation (Start time)
prev_frame_time = time.time()
prev_positions = defaultdict(lambda: None)

pixels_to_meters = 0.05                    # Conversion factor: pixels to meters

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Couldn't read frame from Webcam.")
        break

    # Resize frame
    frame_resized = cv2.resize(frame, (640, 480))

    # Convert the frame to RGB (YOLO in RGB format)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(frame_rgb)

    # Process each detection: box coordinates, class ID, and confidence score
    for result in results:
        boxes = result.boxes

        for i in range(len(boxes)):
            box = boxes.xyxy[i].numpy().astype(int)        # Coordinates: results.boxes.xyxy
            confidence = float(boxes.conf[i].numpy())      # Score: confidence of detection
            class_id = int(boxes.cls[i].numpy())           # Class ID: results.boxes.cls
            label = model.names[class_id]

            if label in ['car', 'truck', 'person', 'bus]:
                x1, y1, x2, y2 = box                       # Box Coordinates
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_id = (label, i)                      # ID from label and index
                
                if prev_positions[track_id] is not None:
                    prev_center = prev_positions[track_id]
                    distance_pixels = euclidean_distance(center, prev_center)
                    distance_meters = distance_pixels * pixels_to_meters
                    new_frame_time = time.time()
                    time_elapsed = new_frame_time - prev_frame_time
                    fps = 1 / time_elapsed if time_elapsed > 0 else 1
                    
                    speed_mps = distance_meters * fps           # (units/s)
                    speed_mph = speed_mps * 2.23694             # Conversion 3600s in 1 hour, 1609m in 1mi (3600/1609)
                    
                    cv2.putText(frame_resized, f"Speed: {speed_mph:.2f} mph", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Update previous positions
                prev_positions[track_id] = center
                prev_frame_time = new_frame_time

                # Draw bounding box and label on frame
                color = (0, 0, 255)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate FPS
    new_frame_time = time.time()
    time_elapsed = new_frame_time - prev_frame_time
    fps = 1 / time_elapsed if time_elapsed > 0 else 1
    prev_frame_time = new_frame_time

    # Display FPS on frame
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Webcam YOLO Detection", frame_resized)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
