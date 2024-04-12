# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

#-------------------------------------------------------------------------------------------------------------------------------------------------

## WILL COME BACK TO UNDERSTAND. Source Link in Trello
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Helps call/find files from the command prompt/terminal
parser = argparse.ArgumentParser()

# Positional Argument
parser.add_argument('--model', help='Folder .tflite file is located.', required=True)		# parser.add_argument('model', help=...)
# Optional Arguments
parser.add_argument('--graph', help='Name of .tflite file', default = 'detect.tflite')
parser.add_argument('--label', help='Name of the labelmap file', default = 'labelmap.txt')
parser.add_argument('--resolution', help='Webcam resolution in WxH.', default='1280x720')
parser.add_argument('--threshold', help='Minimum threshold to display detected objects', default= 0.5)

# Breaks arguments
args = parser.parse_args()

MODEL= args.model                                    # MODEL = parser.parser_args().model
GRAPH = args.graph
LABELMAP = args.label
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
min_threshold = float(args.threshold)

#-------------------------------------------------------------------------------------------------------------------------------------------------

# WILL COME BACK TO UNDERSTAND
# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate
      
#-------------------------------------------------------------------------------------------------------------------------------------------------

# Path to current working directory
CWD_PATH = os.getcwd()
# Path to .tflite file containing model 
CKPT_PATH = os.path.join(CWD_PATH, MODEL, GRAPH)                    # Current dir to Model dir, finds model file 
# Path to label map file
LABELS_PATH = os.path.join(CWD_PATH, MODEL, LABELMAP)               # Current dir to Model dir, finds label file

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Opens and Loads the label file 
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
  
# A weird fix for label map if using the COCO "starter model". First label is '???', which has to be removed. 
# Not sure if COCO starter model will be used.
if labels[0] == '???':
    del(labels[0])

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Load the TFLite model to use.
# tflite.Interpreter()
interpreter = Interpreter(model_path = CKPT_PATH)      

# Neds to be called. TFLite pre-plans tensor allocations to optimize inference
interpreter.allocate_tensors()

# Get model details
# Each item as a dictionary with details (name, index, shape, dtype) 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#-------------------------------------------------------------------------------------------------------------------------------------------------

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    # Grab frame from video stream
    frame1 = videostream.read()
    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] 	# Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] 	# Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] 	# Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] 							# Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) 					# Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)		# Get font size
            label_ymin = max(ymin, labelSize[1] + 10) 							# Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

#-------------------------------------------------------------------------------------------------------------------------------------------------

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
