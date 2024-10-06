from collections import defaultdict
import cv2
import numpy as np
from time import time
from ultralytics import YOLO

# Load the YOLOv8 models
model = YOLO("yolov8s.pt")
model_ct = YOLO("yolov8_1000.pt")

# Open the video file
video_path = "./video.mp4"
cap = cv2.VideoCapture(video_path)

# Global variables
track_history = defaultdict(lambda: [])
classification_data = defaultdict(lambda: {
    "start_time": None,
    "predictions": [],
    "confidences": [],
    "final_class": "Loading",
    "final_confidence": 0.0,
    "best_class": None,
    "best_confidence": 0.0,
    "class_counts": defaultdict(int)
})
track_ids = []

# Time threshold for updating the classification
time_threshold = 3  # seconds

# Confidence threshold for high accuracy
confidence_threshold = 0.8

# Number of intervals a class needs to outperform the best class to replace it
outperform_threshold = 3


# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 video codec
out = cv2.VideoWriter('predicted_output.mp4', fourcc, 30.0, (frame_width, frame_height))



def get_best_class(predictions, confidences):
    if not predictions:
        return "Unknown", 0, 0.0
    class_counts = defaultdict(int)
    class_total_conf = defaultdict(float)
    for pred, conf in zip(predictions, confidences):
        class_counts[pred] += 1
        class_total_conf[pred] += conf

    best_class = max(class_counts, key=class_counts.get)
    best_count = class_counts[best_class]
    avg_confidence = class_total_conf[best_class] / \
        best_count if best_count > 0 else 0.0

    return best_class, best_count, avg_confidence


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        new_track_ids = results[0].boxes.id.int().cpu(
        ).tolist() if results[0].boxes.id is not None else []
        classes = results[0].boxes.cls.cpu().tolist()

        # Update global track IDs
        for new_id in new_track_ids:
            if new_id not in track_ids:
                track_ids.append(new_id)

        # Filter for persons (class 0)
        person_indices = [i for i, cls in enumerate(classes) if cls == 0]
        boxes = boxes[person_indices]
        current_track_ids = [new_track_ids[i]
                             for i in person_indices] if new_track_ids else []

        # Create a clean frame to draw on
        annotated_frame = frame.copy()

        # Plot the filtered boxes and tracks
        for box, track_id in zip(boxes, current_track_ids):
            x, y, w, h = box

            # Initialize classification data if not already
            if classification_data[track_id]["start_time"] is None:
                classification_data[track_id]["start_time"] = time()

            # Extract the person from the frame
            person = frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

            # Classify the person using model_ct
            if person.size > 0:
                classification_results = model_ct(person)[0]
                if len(classification_results.boxes) > 0:
                    class_name = classification_results.names[int(
                        classification_results.boxes.cls[0])]
                    confidence = float(classification_results.boxes.conf[0])
                else:
                    class_name = "Unclassified"
                    confidence = 0.0
            else:
                class_name = "Unknown"
                confidence = 0.0

            # Store the classification result
            classification_data[track_id]["predictions"].append(class_name)
            classification_data[track_id]["confidences"].append(confidence)

            # Check if 3 seconds have passed since the start time
            elapsed_time = time() - classification_data[track_id]["start_time"]
            if elapsed_time >= time_threshold:
                # Get the best class from the last 3 seconds
                interval_best_class, interval_best_count, interval_avg_confidence = get_best_class(
                    classification_data[track_id]["predictions"],
                    classification_data[track_id]["confidences"]
                )

                # Update class counts
                classification_data[track_id]["class_counts"][interval_best_class] += 1

                # If this is the first interval, set it as the best class
                if classification_data[track_id]["best_class"] is None:
                    classification_data[track_id]["best_class"] = interval_best_class
                    classification_data[track_id]["best_confidence"] = interval_avg_confidence
                    classification_data[track_id]["final_class"] = interval_best_class
                    classification_data[track_id]["final_confidence"] = interval_avg_confidence
                else:
                    # Check if the interval best class has outperformed the overall best class
                    if (classification_data[track_id]["class_counts"][interval_best_class] >
                            classification_data[track_id]["class_counts"][classification_data[track_id]["best_class"]] + outperform_threshold):
                        classification_data[track_id]["best_class"] = interval_best_class
                        classification_data[track_id]["best_confidence"] = interval_avg_confidence

                    # Update final class and confidence
                    classification_data[track_id]["final_class"] = classification_data[track_id]["best_class"]
                    classification_data[track_id]["final_confidence"] = classification_data[track_id]["best_confidence"]

                # Reset the start time and predictions for the next interval
                classification_data[track_id]["start_time"] = time()
                classification_data[track_id]["predictions"] = []
                classification_data[track_id]["confidences"] = []

            # Get the final class to display
            display_class = classification_data[track_id]["final_class"]
            display_confidence = classification_data[track_id]["final_confidence"]

            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)),
                          (int(x+w/2), int(y+h/2)), (0, 255, 150), 3)

            # Draw track ID and classification
            label = f"ID: {track_id}, Class: {display_class}"
            cv2.putText(annotated_frame, label, (int(x-w/2), int(y-h/2)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # # Update and draw the tracking lines
            # track = track_history[track_id]
            # track.append((float(x), float(y)))  # x, y center point
            # if len(track) > 50:  # retain 50 tracks for 50 frames
            #     track.pop(0)

            # # Draw the tracking lines
            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(annotated_frame, [points], isClosed=False, color=(
            #     230, 230, 230), thickness=2)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Person Tracking and Classification", annotated_frame)
        out.write(annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
out.release()
cap.release()
cv2.destroyAllWindows()
