import cv2
import numpy as np
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set up the standard RGB camera (default camera)
camera = cv2.VideoCapture(0)  # Change the index if using an external camera

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Main loop
while True:
    # Capture frame-by-frame from the camera
    ret, color_image = camera.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Detect objects using YOLOv5
    results = model(color_image)

    # Process the results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result

        # Create a label with the detected object's name and confidence score
        label = f"{model.names[int(class_id)]} {confidence:.2f}"

        # Draw a rectangle around the object
        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (252, 119, 30), 2)

        # Add label with background for visibility
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_width, text_height = text_size
        text_offset_x, text_offset_y = int(x1), int(y1) - 10
        box_coords = ((text_offset_x, text_offset_y - text_height - 5), (text_offset_x + text_width, text_offset_y + 5))
        cv2.rectangle(color_image, box_coords[0], box_coords[1], (252, 119, 30), cv2.FILLED)
        cv2.putText(color_image, label, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Print the object's class and confidence
        print(f"{model.names[int(class_id)]}: {confidence:.2f}")

    # Display the image with detected objects
    cv2.imshow("Color Image", color_image)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

