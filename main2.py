import numpy as np
import cv2

def get_red_limits():
    # Define the HSV thresholds for a highly restricted range of vivid red colors
    lower_red = np.array([0, 170, 120])  # Lower limit for vivid red in HSV
    upper_red = np.array([20, 255, 255])  # Upper limit for vivid red in HSV

    return lower_red, upper_red

# Function to get bounding boxes for contours
def get_contour_boxes(mask, min_contour_area=300):  # Set a minimum contour area (adjust as needed)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, x + w, y + h))
    return boxes

# Open the webcam
cap = cv2.VideoCapture(0)  # Use '0' for default webcam

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to capture a frame")
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV

    lower_red, upper_red = get_red_limits()
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)  # Create a mask for vivid red hues

    # Get bounding boxes around the significant red objects
    boxes = get_contour_boxes(mask)

    # Draw bounding boxes on the frame
    for box in boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow('Original Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

