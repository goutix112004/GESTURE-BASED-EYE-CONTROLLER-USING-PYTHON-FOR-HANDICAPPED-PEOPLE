import cv2
import mediapipe as mp
import pyautogui
import time
import os

# Initialize Camera
cam = cv2.VideoCapture(0)

# Initialize Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get Screen Size
screen_w, screen_h = pyautogui.size()

# Tracking Blinks
blink_times = []
scrolling_enabled = False
screenshot_count = 1  # Screenshot numbering

# Directory to Save Screenshots
screenshot_dir = "screenshots"
if not os.path.exists(screenshot_dir):
    os.makedirs(screenshot_dir)

def take_screenshot():
    """Capture and save screenshot with a unique name."""
    global screenshot_count
    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{screenshot_count}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(screenshot_path)
    print(f"Screenshot saved: {screenshot_path}")
    screenshot_count += 1

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

        # **Move Cursor with Eyes**
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Eye tracking dot

            if id == 1:
                screen_x = int(screen_w * landmark.x)
                screen_y = int(screen_h * landmark.y)
                pyautogui.moveTo(screen_x, screen_y)

        # **Detect Blinks**
        left_eye = [landmarks[145], landmarks[159]]  # Left eye landmarks
        right_eye = [landmarks[374], landmarks[386]]  # Right eye landmarks

        for landmark in left_eye + right_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # **Detect Single Eye Blink (Left Eye for Clicking)**
        if (left_eye[0].y - left_eye[1].y) < 0.004:
            pyautogui.click()
            time.sleep(0.2)  # Prevents rapid clicking

        # **Detect Both Eyes Blinking (Double Blink for Screenshot)**
        if (left_eye[0].y - left_eye[1].y) < 0.004 and (right_eye[0].y - right_eye[1].y) < 0.004:
            blink_times.append(time.time())
            
            # Keep only blinks in the last 1 second
            blink_times = [t for t in blink_times if time.time() - t < 1]

            if len(blink_times) == 2:  # Detect double blink
                take_screenshot()
                blink_times = []  # Reset after screenshot

        # **Detect Long Blink (2 sec) to Toggle Scrolling**
        if (left_eye[0].y - left_eye[1].y) < 0.004 and (right_eye[0].y - right_eye[1].y) < 0.004:
            start_time = time.time()
            while (left_eye[0].y - left_eye[1].y) < 0.004 and (right_eye[0].y - right_eye[1].y) < 0.004:
                if time.time() - start_time > 2:
                    scrolling_enabled = not scrolling_enabled  # Toggle scrolling
                    print(f"Scrolling {'Enabled' if scrolling_enabled else 'Disabled'}")
                    break

        # **Scroll Based on Head Movement (If Scrolling Enabled)**
        if scrolling_enabled:
            nose_tip = landmarks[1]  # Nose tip landmark
            if nose_tip.y < 0.4:  # If nose moves up
                pyautogui.scroll(5)
            elif nose_tip.y > 0.6:  # If nose moves down
                pyautogui.scroll(-5)

    # Show Video Feed
    cv2.imshow('Eye-Controlled System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
