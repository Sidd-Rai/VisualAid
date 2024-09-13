import cv2
import numpy as np

# Replace with paths to downloaded YOLOv3 files
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
names_path = "coco.names"

# Define object classes to avoid (modify as needed)
objects_to_avoid = ["person", "car", "bicycle"]

# Function to calculate safe movement angle
def calculate_safe_angle(center_x, image_width):
    # This is a basic example. You can implement more sophisticated logic
    # based on object size, distance, and desired avoidance behavior.
    center_offset = (center_x - image_width / 2) / (image_width / 2)
    safe_angle = -90 * center_offset  # Adjust multiplier for finer control
    return int(safe_angle)

def detect_and_avoid(cap):
    # Load YOLOv3 model
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Load class names
    with open(names_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Initialize start time for FPS calculation (optional)
    start_time = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame was captured successfully
        if not ret:
            print("Error: Unable to capture frame from video stream.")
            break

        # Get image dimensions
        (h, w) = frame.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Set input to the network
        net.setInput(blob)

        # Get output layers names
        ln = net.getLayerNames()
        output_layer_names = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # Forward pass through the network
        layers_outputs = net.forward(output_layer_names)

        # Initialize lists for detected bounding boxes and confidences
        boxes = []
        confidences = []
        class_ids = []

        # Loop over each output layer
        for output in layers_outputs:
            # Loop over each detection
            for detection in output:
                # Extract scores (confidences) for each class in the prediction
                scores = detection[5:]
                most_confident_id = np.argmax(scores)
                confidence = scores[most_confident_id]

                # If class confidence is greater than threshold and the class is in the avoid list
                if confidence > 0.5 and class_names[most_confident_id] in objects_to_avoid:
                    # Scale bounding box coordinates based on the image size
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Extract top left coordinates
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Update lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(most_confident_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.1)

        # If any objects were detected
        if len(idxs) > 0:
            # Loop over remaining indices
            for i in idxs.flatten():
                # Extract bounding box coordinates
                (x, y, w, h) = boxes[i]

                # Calculate safe movement angle
                safe_angle = calculate_safe_angle(x + w // 2, w)

                # Print safe angle to terminal
                print(f"Safe movement angle: {safe_angle} degrees")

                # Optional: Draw bounding box and label (for visualization)
                color = (0, 0, 255)  # Red for objects to avoid
                

