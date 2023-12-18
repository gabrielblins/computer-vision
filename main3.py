import numpy as np
import cv2

# Define color ranges in HSV for different colors
color_ranges = {
    'red': ([0, 120, 80], [5, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'blue': ([90, 100, 100], [130, 255, 255]),
    'green': ([50, 100, 100], [70, 255, 255]),
    'purple': ([130, 100, 100], [160, 255, 255]),
    # Add more colors and their corresponding HSV ranges as needed
}

# Function to get lower and upper HSV limits based on color name
def get_color_limits(color_name):
    if color_name.lower() in color_ranges:
        return np.array(color_ranges[color_name.lower()][0]), np.array(color_ranges[color_name.lower()][1])
    else:
        print("Color not found in the defined ranges.")
        return None, None

# Function to get bounding boxes for contours
def get_contour_boxes(mask, min_contour_area=500):
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
chosen_color = input("Enter a color (e.g., red, yellow, blue, green, purple): ")
while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to capture a frame")
        break


    # Get lower and upper HSV limits based on user input
    lower_limit, upper_limit = get_color_limits(chosen_color)

    if lower_limit is not None and upper_limit is not None:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV

        mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)  # Create a mask for the specified color

        # Get bounding boxes around the significant colored objects
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

