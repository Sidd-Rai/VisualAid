import cv2
import numpy as np
import time
import requests
import threading


weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
names_path = "coco.names"


server_url = "http://192.168.119.126"


def ping_server_with_angle(server_url, angle):
    url = f"{server_url}/angle?angle={angle}"


    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"Ping to {url} successful.")
    except requests.RequestException as e:
        print(f"Failed to ping {url}: {e}")


def calculate_safe_angle(center_x, image_width):
    center_offset = (center_x - image_width / 2) / (image_width / 2)
    safe_angle = (center_offset + 1) * 90
    return int(safe_angle)


def detect_objects(cap, net, ping_server_event):
    cumulative_angles = []
    while True:
        ret, frame = cap.read()


        if not ret:
            print("Error: Unable to capture frame from video stream.")
            break


        (h, w) = frame.shape[:2]


        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (100, 100), swapRB=True, crop=False)
        net.setInput(blob)
        ln = net.getLayerNames()
        output_layer_names = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
        layers_outputs = net.forward(output_layer_names)


        boxes = []
        confidences = []


        for output in layers_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]


                if confidence > 0.3:  # Adjust this threshold as needed
                    # Scale bounding box coordinates based on the image size
                    box = detection[0:4] * np.array([w, h, w, h])
                    (x, y, width, height) = box.astype("int")


                    # Calculate top-left and bottom-right coordinates of the bounding box
                    x_min = int(x - (width / 2))
                    y_min = int(y - (height / 2))
                    x_max = int(x + (width / 2))
                    y_max = int(y + (height / 2))


                    boxes.append([x_min, y_min, x_max, y_max])
                    confidences.append(float(confidence))


        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.5)


        if len(indices) > 0:
            for i in indices.flatten():
                (x_min, y_min, x_max, y_max) = boxes[i]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                safe_angle = calculate_safe_angle((x_min + x_max) / 2, w)
                cumulative_angles.append(safe_angle)


        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if cumulative_angles and ping_server_event.is_set():
            collective_angle = sum(cumulative_angles) / len(cumulative_angles)
            print(f"Collective angle: {collective_angle} degrees")
            angle = collective_angle
            if not ping_server_with_angle(server_url, angle):
                print("Failed to ping server. Retrying...")
            cumulative_angles = []


def main():
    cap = cv2.VideoCapture(2)
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)


    ping_server_event = threading.Event()
    ping_server_event.set()


    threading.Thread(target=detect_objects, args=(cap, net, ping_server_event)).start()


    while True:
        command = input("Press 'p' to ping server or 'q' to quit: ")
        if command == 'q':
            break
        elif command == 'p':
            ping_server_event.set()


    ping_server_event.clear()  # stop server pinging
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()