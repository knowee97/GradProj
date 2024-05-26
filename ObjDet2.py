from ultralytics import YOLO
import argparse
import os
import cv2
import numpy as np
import argparse
import sys
import time
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--model', help = 'Folder where .tflite file is located.', required = True)
parser.add_argument('--graph', help = 'Name of .tflite file', default = 'yolov8sModel_float32.tflite')

args = parser.parse_args()

MODEL = args.model
GRAPH = args.graph 

wd_path = os.getcwd()
model_path = os.path.join(wd_path, MODEL, GRAPH)

# Should go here: graph_path = r"C:\Users\12108\fiftyone\detect\MODEL\yolov8sModel_float32.tflite"

# Load YOLO.tflite model
tflite_model = YOLO(graph_path)

# Initialize Webcam
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# FPS calculation (Start time)
prev_frame_time = 0
new_frame_time = 0
    
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Couldn't read frame from Webcam.")
        break

    # Resize frame to a smaller size for faster processing
    frame_resized = cv2.resize(frame, (640, 480))
    
    # Convert the frame to RGB (YOLO expects images in RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference
    results = tflite_model(frame_rgb)

    # Determine box coordinates, class ID, and confidence score
    for result in results:
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].numpy().astype(int)        # Coordinates: results.boxes.xyxy
            confidence = float(boxes.conf[i].numpy())
            class_id = int(boxes.cls[i].numpy())           # Class ID: results.boxes.cls
            label = tflite_model.names[class_id]
            
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box

            # BBox and Class ID on frame
            color = (0, 0, 255)
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
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
