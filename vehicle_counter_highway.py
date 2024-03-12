import cv2
import numpy as np
from sort import Sort
from ultralytics import YOLO
import math

def enhance_frame(frame):
    """Enhancing frame visibility through histogram equalization."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return enhanced_frame

video_capture = cv2.VideoCapture('cars2.mp4')
yolo_model = YOLO('yolov8n.pt')
with open('classes.txt', 'r') as file:
    class_names = file.read().splitlines()

vehicle_tracker = Sort(max_age=20)
vehicle_counter = []

# Defining the detection line
frame_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
line_position = frame_height // 2
detection_line = [(0, int(line_position)), (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(line_position))]


while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Applying frame enhancement
    frame = enhance_frame(frame)

    detections = np.empty((0, 5))
    detection_results = yolo_model(frame)
    for detection in detection_results:
        for bbox in detection.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            confidence = bbox.conf[0]
            class_id = int(bbox.cls[0])
            confidence_percentage = math.ceil(confidence * 100)
            detected_object = class_names[class_id]

            box_area = (x_max - x_min) * (y_max - y_min)

            if detected_object in ['car', 'bus', 'truck'] and confidence_percentage > 50:
                detections = np.vstack((detections, np.array([x_min, y_min, x_max, y_max, confidence_percentage])))

    #  tracker
    tracking_results = vehicle_tracker.update(detections)

    # Drawing the detection line
    cv2.line(frame, detection_line[0], detection_line[1], (255, 0, 0), 2)

    for track in tracking_results:
        x_min, y_min, x_max, y_max, track_id = map(int, track)
        center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

        if line_position - 10 < center_y < line_position + 10:  # Vehicle crosses the line
            if track_id not in vehicle_counter:
                vehicle_counter.append(track_id)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        cv2.putText(frame, f'ID: {track_id}', (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Displaying the vehicle count
    cv2.putText(frame, f'Vehicle Count: {len(vehicle_counter)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Vehicle Detection and Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
