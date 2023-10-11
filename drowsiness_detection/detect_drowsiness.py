# USAGE
# python detect_facial_expression.py --shape-predictor shape_predictor_68_face_landmarks.dat
from flatbuffers.builder import np
from keras.src.saving.saving_api import load_model
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3

engine = pyttsx3.init()
newVoiceRate = 145
engine.setProperty('rate', newVoiceRate)
voices = engine.getProperty('voices')
engine.say("A-I based Facial Expression Detection System")
engine.runAndWait()

def facial_expression(model, face, emotion_labels=None):
    # Preprocess the face image (resize, normalize, etc.)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape((1, 48, 48, 1))

    # Predict the emotion label
    emotion_prediction = model.predict(face)
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]
    return emotion_label

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load a pre-trained model for facial expression recognition
model = load_model('facial_expression_model.h5')

# define constants for the number of consecutive frames an emotion must be detected
EMOTION_CONSEC_FRAMES = 10

# initialize the frame counter and emotion label
COUNTER = 0
emotion_label = ""

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(2)

# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y+h, x:x+w]

        # detect facial expression
        emotion_label = facial_expression(model, face)

        # increment the emotion counter
        COUNTER += 1

        # if the same emotion is detected for a sufficient number of consecutive frames, alert
        if COUNTER >= EMOTION_CONSEC_FRAMES:
            cv2.putText(frame, f"Emotion: {emotion_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            time.sleep(4)
            engine = pyttsx3.init()
            engine.say(f"Alert! Detected emotion: {emotion_label}")
            print(f"Alert! Detected emotion: {emotion_label}")
            engine.runAndWait()

        # draw the face rectangle and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # reset the emotion counter when no face is detected
    if not rects:
        COUNTER = 0

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
