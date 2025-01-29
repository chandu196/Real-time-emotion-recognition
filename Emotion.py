import cv2
import numpy as np
import time
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import argparse

# Feedback responses
feedback_responses = {
    "angry": ["Take a deep breath. It's okay to feel angry, but remember to stay calm.", "Count to ten slowly. This can help manage your anger."],
    "disgust": ["Sometimes things aren't pleasant. Try to focus on something positive.", "If something's bothering you, it's okay to remove yourself from the situation."],
    "scared": ["It's normal to feel scared sometimes. Remember that you're safe right now.", "Try some deep breathing exercises to help calm your nerves."],
    "happy": ["It's great to see you happy! Keep spreading that positivity!", "Happiness looks good on you. Enjoy this moment!"],
    "sad": ["It's okay to feel sad. Remember that all feelings are temporary.", "Would you like to try a quick mood-lifting exercise?"],
    "surprised": ["Surprises can be exciting! I hope it's a good surprise.", "Take a moment to process your surprise. It's okay to feel overwhelmed."],
    "neutral": ["How are you feeling right now? Sometimes it's good to check in with yourself.", "Remember to take breaks and relax throughout your day."]
}

def get_feedback(emotion):
    return np.random.choice(feedback_responses[emotion])

# Parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# Loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def main(camera_index=0, frame_width=300):
    last_feedback_time = 0
    feedback_cooldown = 5  # seconds
    preds = np.zeros(len(EMOTIONS))  # Initialize preds with zeros
    current_feedback = ""

    # Creating windows
    cv2.namedWindow('your_face')
    cv2.namedWindow('Probabilities')
    cv2.namedWindow('Feedback', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Feedback', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        print(f"Error: Unable to open camera at index {camera_index}")
        return

    while True:
        # Reading the frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        frame = imutils.resize(frame, width=frame_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()

        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # Extract the ROI of the face from the grayscale image, resize it to a fixed 64x64 pixels, and then prepare the ROI for classification 
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            # Add the feedback system here
            current_time = time.time()
            if current_time - last_feedback_time > feedback_cooldown:
                current_feedback = get_feedback(label)
                last_feedback_time = current_time

            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # Construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # Draw the label + probability bar on the canvas
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # Display the results
        cv2.imshow('your_face', frameClone)
        cv2.imshow("Probabilities", canvas)

        # Display feedback in full-screen
        feedback_display = np.zeros((1080, 1920, 3), dtype="uint8")  # Adjust size as needed
        y0, dy = 50, 30
        for i, line in enumerate(current_feedback.split('\n')):
            y = y0 + i*dy
            cv2.putText(feedback_display, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Feedback", feedback_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Emotion Recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=300, help='Frame width (default: 300)')
    args = parser.parse_args()

    main(camera_index=args.camera, frame_width=args.width)
