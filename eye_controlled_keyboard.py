import cv2
import mediapipe as mp
import pyautogui

# Initialize Webcam
cam = cv2.VideoCapture(0)

# Initialize Face Mesh Detector
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get Screen Size
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip to match mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # **Scroll Based on Eye Position**
        eye = landmarks[1]  # Nose bridge as reference
        screen_y = int(screen_h * eye.y)

        if eye.y < 0.4:  # Looking Up → Scroll Up
            pyautogui.scroll(50)
        elif eye.y > 0.6:  # Looking Down → Scroll Down
            pyautogui.scroll(-50)

        # **Blink Detection for Clicking**
        left_eye = [landmarks[145], landmarks[159]]  # Upper & lower eyelid
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        if (left_eye[0].y - left_eye[1].y) < 0.004:  # Blink detected
            pyautogui.click()
            pyautogui.sleep(1)  # Prevent multiple clicks

    # Show Video Feed
    cv2.imshow('Eye-Controlled Scrolling & Clicking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
