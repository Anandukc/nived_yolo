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
    # Extract bounding boxes and labels from the result
    bounding_boxes = result["predictions"]
    for box in bounding_boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        label = box["label"]
        confidence = box["confidence"]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame and draw bounding boxes
    frame_with_boxes = infer_frame(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame_with_boxes)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
