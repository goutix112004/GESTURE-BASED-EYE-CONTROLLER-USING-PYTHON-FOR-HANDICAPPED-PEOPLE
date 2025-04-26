import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Webcam
cam = cv2.VideoCapture(0)

# Initialize Face Mesh Detector
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get Screen Size
screen_w, screen_h = pyautogui.size()

# Virtual Keyboard Layout
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "←"],
    ["Z", "X", "C", "V", "B", "N", "M", "Enter"]
]

# Blink tracking
blink_times = []

def draw_keyboard(frame):
    """Draws the virtual keyboard on the frame"""
    key_x, key_y = 50, 300  # Position of the keyboard
    key_size = 60  # Size of each key
    spacing = 5  # Spacing between keys

    for row_idx, row in enumerate(keys):
        for col_idx, key in enumerate(row):
            x = key_x + col_idx * (key_size + spacing)
            y = key_y + row_idx * (key_size + spacing)

            # Draw Key Background
            cv2.rectangle(frame, (x, y), (x + key_size, y + key_size), (255, 255, 255), -1)

            # Draw Key Text
            text_x = x + 15 if key != "Enter" else x + 5
            text_y = y + 40
            cv2.putText(frame, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame

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

        # Track Eye Movement
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Eye tracking dot

            if id == 1:
                screen_x = int(screen_w * landmark.x)
                screen_y = int(screen_h * landmark.y)
                pyautogui.moveTo(screen_x, screen_y)

        # Detect Blinks for Typing
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Blink Detection (Without Freezing)
        if (left[0].y - left[1].y) < 0.004:  # Blink detected
            blink_times.append(time.time())

            # Keep only blinks in the last 1 second
            blink_times = [t for t in blink_times if time.time() - t < 1]

            if len(blink_times) == 1:  # Single Blink - Type "A"
                pyautogui.write("A")
                time.sleep(0.2)  # Small delay to prevent continuous writing

    # Draw Virtual Keyboard
    frame = draw_keyboard(frame)

    # Show Video Feed
    cv2.imshow('Eye-Controlled Virtual Keyboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
