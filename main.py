import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from requests import post
from datetime import datetime
import pickle  # For saving and loading embeddings

# Configuration
KNOWN_FACES_DIR = "known_faces"  # Folder containing known faces
EMBEDDINGS_FILE = "known_faces_embeddings.pkl"  # File to store embeddings
THRESHOLD = 0.6  # Similarity threshold for face recognition
WEBHOOK_URL = "https://your-webhook-url.com"  # Replace with your webhook URL

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Load or initialize known faces
if os.path.exists(EMBEDDINGS_FILE):
    # Load embeddings from file
    with open(EMBEDDINGS_FILE, "rb") as f:
        known_faces, known_names = pickle.load(f)
else:
    known_faces = []
    known_names = []

def save_embeddings():
    """
    Saves the known faces and their embeddings to a file.
    """
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump((known_faces, known_names), f)

def add_new_face(image_path, name):
    """
    Adds a new face to the known faces list and saves the embeddings.
    """
    image = cv2.imread(image_path)
    faces = app.get(image)
    if len(faces) > 0:
        embedding = faces[0].embedding
        known_faces.append(embedding)
        known_names.append(name)
        save_embeddings()  # Save updated embeddings
        print(f"Added new face: {name}")
    else:
        print(f"No face detected in {image_path}")

# Load known faces from the directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        name = os.path.splitext(filename)[0]
        if name not in known_names:  # Avoid duplicates
            add_new_face(image_path, name)

# Initialize webcam
cap = cv2.VideoCapture(0)

def send_webhook(name):
    """
    Sends a webhook notification when a recognized face is detected.
    """
    payload = {
        "event": "face_recognized",
        "name": name,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        response = post(WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"Webhook sent for {name}")
    except Exception as e:
        print(f"Failed to send webhook: {e}")

def show_popup(name):
    """
    Displays a popup notification (macOS only).
    """
    os.system(f'osascript -e \'display notification "{name} detected!" with title "Face Recognized"\'')

def recognize_face(embedding):
    """
    Compares the detected face embedding with known faces.
    """
    for i, known_embedding in enumerate(known_faces):
        similarity = np.dot(embedding, known_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
        )
        if similarity > THRESHOLD:
            return known_names[i]
    return None

# Main loop for real-time face recognition
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = app.get(frame)

    for face in faces:
        # Recognize the face
        name = recognize_face(face.embedding)
        if name:
            print(f"Recognized: {name}")
            show_popup(name)  # Show popup notification
            # send_webhook(name)  # Send webhook

        # Draw bounding box and name on the frame
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name if name else "Unknown",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()