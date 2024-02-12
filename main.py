
import cv2
import numpy as np
from imutils import face_utils
import dlib
import pygame
from datetime import datetime, timedelta
from twilio.rest import Client
import geocoder


account_sid = 'AC36ea6efe9fef56b8f1365558ee44fb02'
auth_token = 'd9f0c83cf319d8d4b503d11981da2204'
client = Client(account_sid, auth_token)

def sendAlertWithLocation():
    location = geocoder.ip('me')    
    lat = location.latlng[0]
    long = location.latlng[1]
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body='Your SafeDriver has detected that the driver named Ram is either sleeping or drowsing. Please contact him immediately to ensure his safety.\n\n Location: (' + str(long) + ',' + str(lat) + ')', 
        to='whatsapp:+919708667464'
        )


def sendAlert():
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body='Your SafeDriver has detected that the driver named Ram is either sleeping or drowsing. Please contact him immediately to ensure his safety.', 
        to='whatsapp:+919708667464'
        )
    return (message.sid)


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

blink_frames = 0
blink_ratio_threshold = 0.2
drowsy_threshold = 6
distracted_frames = 0
distracted_threshold = 15
alarm_active = False
drowsy_start_time = None
send_sms = False
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(ear, blink_frames, threshold):
    if ear < threshold:
        blink_frames += 1
    else:
        blink_frames = 0
    return blink_frames

def detect_drowsiness(blink_ratio, drowsy_frames, drowsy_threshold):
    global drowsy_start_time

    if blink_ratio < blink_ratio_threshold:
        drowsy_frames += 1
        if drowsy_frames > drowsy_threshold:
            if drowsy_start_time is None:
                drowsy_start_time = datetime.now()
            return True
    else:
        drowsy_frames = 0
        drowsy_start_time = None
    return False

def detect_distraction(face_points):
    if face_points[0][0] < 100:
        return True
    return False

def play_alarm():
    global alarm_active
    if not alarm_active:
        alarm_sound.play()
        alarm_active = True

def reset_alarm():
    global alarm_active
    alarm_sound.stop()
    alarm_active = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:48]
        right_eye = landmarks[42:60]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        blink_ratio = (left_ear + right_ear) / 2.0

        blink_frames = is_blinking(blink_ratio, blink_frames, blink_ratio_threshold)

        if detect_drowsiness(blink_ratio, blink_frames, drowsy_threshold):
            cv2.putText(frame, "Drowsiness Detected!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            play_alarm()
            if drowsy_start_time is not None:
                elapsed_time = datetime.now() - drowsy_start_time
                if elapsed_time > timedelta(seconds=3):
                    cv2.putText(frame, "Sleeping!", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    if (send_sms  == False):
                        sendAlertWithLocation()                     
                        cv2.putText(frame, "Alert Message Sent!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        send_sms = True

        elif detect_distraction(landmarks):
            cv2.putText(frame, "Distraction Detected!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            play_alarm()
        else:
            cv2.putText(frame, "Active", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            reset_alarm()

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Driver Monitoring System", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()