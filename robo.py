import cv2
from inference_sdk import InferenceHTTPClient

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="GSmqROFEVOSjhodKPQ8j"
)

# Function to perform inference on a single frame
def infer_frame(frame):
    # Send the frame for inference
    result = CLIENT.infer(frame, model_id="waste-object-detection-hltsh/1")
    print(result)

# Capture video from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Perform inference on the frame
    infer_frame(frame)

# Release the capture
cap.release()
cv2.destroyAllWindows()