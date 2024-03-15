import cv2
import os

# Load pre-trained hand gesture detection model
hand_cascade = cv2.CascadeClassifier('Haarcascades/face.xml')

# Load video file
cap = cv2.VideoCapture('sample2.mp4')

# Initialize video writer for saving zoomed-in hand gesture screenshots
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (frame_width, frame_height))

# Initialize screenshot counter
screenshot_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for hand detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Check if the cascade classifier is loaded successfully
    if hand_cascade.empty():
        print("Error: Failed to load the cascade classifier.")
        exit()

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Zoom into detected hand gestures and save screenshots
    for (x, y, w, h) in hands:
        # Highlight detected hand gesture
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Zoom in by cropping region around hand gesture
        zoomed_in = frame[max(0, y-50):min(frame_height, y+h+50), max(0, x-50):min(frame_width, x+w+50)]

        # Save screenshot
        screenshot_count += 1
        screenshot_name = f'screenshot_{screenshot_count}.jpg'

        # Save zoomed-in screenshot
        filename = os.path.join('sample2_smile_results/', screenshot_name)
        cv2.imwrite(filename, zoomed_in)

    # Write the frame into the output video
        out.write(frame)

    # Display the frame
        cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


# 'haarcascade_eye.xml' + 'haarcascade_eye_tree_eyeglasses.xml' + 'haarcascade_frontalcatface.xml' + 'haarcascade_frontalcatface_extended.xml' + 'haarcascade_frontalface_alt.xml' + 'haarcascade_frontalface_alt_tree.xml' + 'haarcascade_frontalface_alt2.xml' + 'haarcascade_frontalface_default.xml' + 'haarcascade_fullbody.xml' + 'haarcascade_lefteye_2splits.xml' + 'haarcascade_license_plate_rus_16stages.xml' + 'haarcascade_lowerbody.xml' + 'haarcascade_profileface.xml' + 'haarcascade_righteye_2splits.xml' + 'haarcascade_russian_plate_number.xml' + 'haarcascade_smile.xml' + 'haarcascade_upperbody.xml'