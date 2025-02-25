import subprocess
import numpy as np
import cv2

# RTSP URL
rtsp_url = "rtsp://admin:toqjys-hywwa6-nitFem@192.168.0.12:554"

# FFmpeg command to decode RTSP stream
ffmpeg_command = [
    "ffmpeg",
    "-i", rtsp_url,  # Input RTSP stream
    "-f", "image2pipe",  # Output format as raw video frames
    "-pix_fmt", "bgr24",  # Pixel format (compatible with OpenCV)
    "-vcodec", "rawvideo",  # Output raw video frames
    "-"  # Output to stdout
]

# Start FFmpeg process
process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Frame dimensions (adjust based on your camera's resolution)
frame_width = 640
frame_height = 480
frame_size = frame_width * frame_height * 3  # 3 channels (BGR)
frame_num = 0
while True:
    # Read raw video frame from FFmpeg stdout
    raw_frame = process.stdout.read(frame_size)
    if not raw_frame:
        print("Error: Failed to read frame.")
        break

    # Convert raw frame to numpy array
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3))
    frame_num += 1
    print(frame_num)

   

# Clean up
process.terminate()
