"""
Eye Blink Alarm
===============qq
Triggers an alarm if eyes remain closed for 5 seconds (configurable).

Setup:
    pip install opencv-python mediapipe pygame numpy

Run:
    python eye_blink_alarm.py
"""
print("START")
import cv2
import mediapipe as mp
import time
import math
import pygame
import numpy as np
import sys


# ─── Settings ────────────────────────────────────────────────────────────────
CLOSED_THRESHOLD_SECONDS = 5.0   # How long eyes can stay closed before alarm
EAR_THRESHOLD            = 0.22  # Eye Aspect Ratio below this = eyes closed
ALARM_BEEP_HZ            = 880   # Alarm frequency in Hz
ALARM_DURATION_MS        = 400   # Duration of each beep in milliseconds
# ─────────────────────────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices for left and right eyes
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """Calculate Eye Aspect Ratio (EAR) — lower value means eyes are more closed."""
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def generate_beep(frequency=880, duration_ms=400, volume=0.5):
    """Generate a beep sound using numpy — no WAV file needed."""
    sample_rate = 44100
    n_samples   = int(sample_rate * duration_ms / 1000)
    t           = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave        = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)
    stereo      = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


def draw_overlay(frame, eyes_closed, elapsed, threshold, alarm_on, face_found):
    """Draw status bar and progress indicator on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if not face_found:
        cv2.putText(frame, "No face detected — move closer to the camera",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        return

    # Progress bar (fills as eyes stay closed)
    if eyes_closed:
        progress  = min(elapsed / threshold, 1.0)
        bar_color = (0, 0, 255) if alarm_on else (0, 165, 255)
        cv2.rectangle(frame, (15, 55), (15 + int((w - 30) * progress), 65), bar_color, -1)
        cv2.rectangle(frame, (15, 55), (w - 15, 65), (100, 100, 100), 1)

    # Status text
    if alarm_on:
        status = f"ALARM! Eyes closed for {elapsed:.1f}s — Wake up!"
        color  = (0, 0, 255)
    elif eyes_closed:
        status = f"Eyes closed: {elapsed:.1f}s / {threshold:.0f}s"
        color  = (0, 165, 255)
    else:
        status = "Eyes open — All good"
        color  = (0, 200, 80)

    cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Flashing red border during alarm
    if alarm_on and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)


def main():
    # Initialize audio
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    beep_sound = generate_beep(ALARM_BEEP_HZ, ALARM_DURATION_MS)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    eyes_closed_since = None
    alarm_active      = False
    last_beep_time    = 0

    print("Eye Blink Alarm started!")
    print(f"Threshold: {CLOSED_THRESHOLD_SECONDS} seconds")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame      = cv2.flip(frame, 1)
        h, w       = frame.shape[:2]
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = face_mesh.process(rgb)

        face_found   = False
        eyes_closed  = False
        elapsed      = 0.0

        if results.multi_face_landmarks:
            face_found = True
            lms        = results.multi_face_landmarks[0].landmark

            left_ear   = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear  = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            avg_ear    = (left_ear + right_ear) / 2.0
            eyes_closed = avg_ear < EAR_THRESHOLD

            if eyes_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()
                elapsed = time.time() - eyes_closed_since

                if elapsed >= CLOSED_THRESHOLD_SECONDS:
                    alarm_active = True
                    now = time.time()
                    if now - last_beep_time >= (ALARM_DURATION_MS / 1000 + 0.1):
                        beep_sound.play()
                        last_beep_time = now
            else:
                eyes_closed_since = None
                alarm_active      = False
        else:
            eyes_closed_since = None
            alarm_active      = False

        draw_overlay(frame, eyes_closed, elapsed,
                     CLOSED_THRESHOLD_SECONDS, alarm_active, face_found)

        cv2.imshow("Eye Blink Alarm — press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pygame.mixer.quit()
    print("Alarm stopped.")


if __name__ == "__main__":
    main()
