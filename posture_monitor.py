import cv2
import mediapipe as mp
import pygame
import math

# Initialize pygame for sound
pygame.mixer.init()

# Load beep sound
beep_sound = pygame.mixer.Sound('beep.wav')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open webcam
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """Calculates angle between three points"""
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    if angle < 0:
        angle += 360
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Process the frame to detect pose
    results = pose.process(frame)

    # Draw the pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check for posture (for example, shoulder and hip alignment)
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # Angle check for shoulder alignment (more advanced check)
        shoulder_angle = calculate_angle(
            [left_shoulder.x, left_shoulder.y],
            [right_shoulder.x, right_shoulder.y],
            [left_hip.x, left_hip.y]
        )

        # A very basic threshold for a bad posture (you can fine-tune this)
        if abs(shoulder_angle - 180) > 15:
            # Play beep sound when posture is bad
            beep_sound.play()

    # Display the frame with landmarks
    cv2.imshow('Posture Monitor', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
