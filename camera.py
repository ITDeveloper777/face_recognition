import cv2

# Replace with your RTSP stream URL
rtsp_url = "rtsp://admin:toqjys-hywwa6-nitFem@192.168.0.12:554"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the stream is opened
if not cap.isOpened():
    print("Error: Could not open RTSP stream")
    exit()

# Set the frame counter
frame_count = 0

while True:
    # Read a frame from the stream
    ret, frame = cap.read()
    
    if not ret:
        print("Fail to read the camera")
        break
    frame_count += 1
    print(frame_count)
    

# Release the capture and close all windows
cap.release()