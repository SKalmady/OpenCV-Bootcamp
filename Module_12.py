"""
# Module 12: Real-Time Face Detection using SSD

Unlike Haar Cascades (which use older, fragile contrast-detection math), modern computer vision relies on Deep Learning for robust face tracking. 
In this script, we deploy a Single Shot MultiBox Detector (SSD) paired with a lightweight ResNet-10 backbone. 
This combination is incredibly fast and highly accurate, allowing us to process live high-definition webcam feeds in real-time without needing a massive graphics card. 


### 1. Fetching Network Weights & Initializing Hardware
To run this inference pipeline, we need two components from the downloaded assets:
**`deploy.prototxt` (Architecture): The structural blueprint of the ResNet-10 SSD.
**`res10_300x300_ssd_iter_140000_fp16.caffemodel` (Weights): The trained mathematical parameters, optimized in 16-bit floating-point format for faster computation.

Once the network is compiled into memory via `cv2.dnn.readNetFromCaffe`, we initialize our system's primary webcam using `cv2.VideoCapture(0)` and prepare a designated OpenCV window to render the output stream.

### 2. Frame Extraction and Blob Normalization
Live video is just a rapid sequence of static images. We use an infinite `while` loop to capture these frames one by one. 
Before passing a frame to the AI, we must standardize it:
1. Mirroring: We flip the frame horizontally (`cv2.flip`) so it acts like a mirror, making the visual experience intuitive for the user.
2. Blob Conversion: The AI expects a 300x300 pixel matrix. We use `cv2.dnn.blobFromImage` to resize the frame and perform Mean Subtraction. 
We subtract the values `(104, 117, 123)` from the Blue, Green, and Red channels respectively to remove global lighting biases.

### 3. Forward Pass & Matrix Slicing
We feed the standardized blob into our model and trigger `net.forward()`. 
The model returns a highly dense 4D array containing all its predictions. 

We iterate through the third dimension (`detections.shape[2]`), which represents every individual face the AI thinks it found. 
For every potential face, we extract the Confidence Score. 
To eliminate false positives and background noise, we enforce a strict 70% Confidence Threshold (`conf_threshold = 0.7`). 
Any prediction that is less than 70% certain is immediately ignored.

### 4. Telemetry Rendering & Inference Profiling
If a face passes our confidence check, the network provides its location as normalized coordinates (percentages between 0.0 and 1.0). 
We multiply these percentages by our webcam's actual pixel width and height to calculate the exact top-left and bottom-right pixel corners.

Finally, we overlay our visual telemetry:
* Bounding Boxes: A thick green rectangle (`cv2.rectangle`) is drawn around the mathematical corners.
* Confidence Labels: The exact certainty percentage is printed on a highly visible white background above the face.
* Hardware Profiling: We use `net.getPerfProfile()` and `cv2.getTickFrequency()` to calculate exactly how many milliseconds it took the CPU to process the frame, displaying the speed in real-time in the top-left corner of the stream.
"""

import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve

# ========================-Downloading Assets-========================
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_12.zip")

# Download if asset ZIP does not exist
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================

# Set video source to default webcam (0)
source = cv2.VideoCapture(0)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

print("Starting webcam stream... Press 'ESC' to exit.")

while cv2.waitKey(1) != 27:  # 27 is the ASCII code for the ESC key
    has_frame, frame = source.read()
    if not has_frame:
        print("Failed to grab frame. Exiting...")
        break
        
    # Flip the frame horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    
    # Run the forward pass
    net.setInput(blob)
    detections = net.forward()

    # Loop through all detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Only draw if confidence is above our 70% threshold
        if confidence > conf_threshold:
            x_top_left = int(detections[0, 0, i, 3] * frame_width)
            y_top_left = int(detections[0, 0, i, 4] * frame_height)
            x_bottom_right = int(detections[0, 0, i, 5] * frame_width)
            y_bottom_right = int(detections[0, 0, i, 6] * frame_height)

            # Draw the green bounding box
            cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0), 2)
            
            # Format the confidence label
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw the white background rectangle for the text
            cv2.rectangle(
                frame,
                (x_top_left, y_top_left - label_size[1] - 5),
                (x_top_left + label_size[0], y_top_left + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            
            # Draw the text itself
            cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Calculate and display real-time inference speed
    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Render the final frame to the popup window
    cv2.imshow(win_name, frame)

# Cleanup resources once the loop breaks
source.release()
cv2.destroyAllWindows()