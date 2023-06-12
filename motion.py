import cv2

# Initialize video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the path to a video file

# Read the first frame
ret, frame1 = video_capture.read()
previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Define motion detection parameters
motion_threshold = 30  # Adjust this value as per your requirement

while True:
    # Read the current frame
    ret, frame2 = video_capture.read()
    current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current and previous frames
    frame_delta = cv2.absdiff(previous_frame, current_frame)

    # Apply thresholding to convert the difference into binary image
    _, threshold = cv2.threshold(frame_delta, motion_threshold, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise
    threshold = cv2.dilate(threshold, None, iterations=2)
    threshold = cv2.erode(threshold, None, iterations=2)

    # Find contours of the threshold image
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and find motion regions
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust this value as per your requirement
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Display the resulting frame with motion regions
    cv2.imshow('Motion Detection', frame2)

    # Update the previous frame
    previous_frame = current_frame.copy()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv2.destroyAllWindows()
