import cv2
import numpy as np

# Set up the camera
cap = cv2.VideoCapture(0)

# Define the region of interest (ROI) where the line is expected to be
roi_top = 100  # top boundary of the ROI
roi_bottom = 300  # bottom boundary of the ROI
roi_left = 0  # left boundary of the ROI
roi_right = 640  # right boundary of the ROI

# Define the threshold values for detecting the line
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# Define the thresholds for turning left or right
threshold_left = roi_left + (roi_right - roi_left) * 0.4
threshold_right = roi_left + (roi_right - roi_left) * 0.6

# Loop over frames from the camera
while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define the region of interest (ROI)
    roi = gray[roi_top:roi_bottom, roi_left:roi_right]

    # Threshold the ROI to get the black line
    _, thresh = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Determine the direction to turn based on the position of the line
    if len(contours) == 0:
        direction = "no line"
    else:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            direction = "no line"
        else:
            cx = int(M["m10"] / M["m00"])
            if cx < threshold_left:
                direction = "left"
            elif cx > threshold_right:
                direction = "right"
            else:
                direction = "straight"

        # Draw the line on the black path
        #c = c + np.array([roi_left, roi_top])  # shift the contour to match the original frame
        #leftmost = tuple(c[c[:, :, 0].argmin()][0])
        #rightmost = tuple(c[c[:, :, 0].argmax()][0])
        #cv2.line(frame, leftmost, rightmost, (0, 255, 0), 3)

    # Draw a rectangle around the ROI on the original frame
    cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 255), 2)

    # Display the direction on the frame
    cv2.putText(frame, direction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Line Following Robot', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

