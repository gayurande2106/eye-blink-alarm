"""
Eye Blink Alarm — Streamlit Edition
====================================
Triggers an alarm if eyes remain closed for N seconds (configurable).

Setup:
    pip install streamlit streamlit-webrtc opencv-python-headless mediapipe numpy av

Run:
    streamlit run app.py
"""

import time
import math
import base64
import io
import wave

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Eye Blink Alarm", page_icon="👁️", layout="centered")
st.title("👁️ Eye Blink Alarm")
st.caption("Sounds an alarm if your eyes stay closed too long — great for drowsiness detection.")

# ─── Sidebar settings ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    CLOSED_THRESHOLD_SECONDS = st.slider("Alarm after eyes closed for (seconds)", 1.0, 15.0, 5.0, 0.5)
    EAR_THRESHOLD = st.slider("Eye Aspect Ratio threshold (lower = stricter)", 0.10, 0.35, 0.22, 0.01)
    ALARM_BEEP_HZ = st.slider("Alarm tone frequency (Hz)", 300, 1500, 880, 10)
    ALARM_DURATION_MS = st.slider("Beep duration (ms)", 100, 1000, 400, 50)

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


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


def make_beep_b64(frequency=880, duration_ms=400, volume=0.5, sample_rate=44100):
    """Generate a beep as an in-memory WAV file, base64-encoded for HTML <audio>."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, False)
    wave_data = (np.sin(2 * np.pi * frequency * t) * volume * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def draw_overlay(frame, eyes_closed, elapsed, threshold, alarm_on, face_found):
    """Draw status bar and progress indicator on the frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if not face_found:
        cv2.putText(frame, "No face detected - move closer to the camera",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        return

    if eyes_closed:
        progress = min(elapsed / threshold, 1.0)
        bar_color = (0, 0, 255) if alarm_on else (0, 165, 255)
        cv2.rectangle(frame, (15, 55), (15 + int((w - 30) * progress), 65), bar_color, -1)
        cv2.rectangle(frame, (15, 55), (w - 15, 65), (100, 100, 100), 1)

    if alarm_on:
        status = f"ALARM! Eyes closed for {elapsed:.1f}s - Wake up!"
        color = (0, 0, 255)
    elif eyes_closed:
        status = f"Eyes closed: {elapsed:.1f}s / {threshold:.0f}s"
        color = (0, 165, 255)
    else:
        status = "Eyes open - All good"
        color = (0, 200, 80)

    cv2.putText(frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if alarm_on and int(time.time() * 2) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)


class EyeBlinkProcessor:
    """Video frame processor for streamlit-webrtc. Tracks eye-closed state across frames."""

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.eyes_closed_since = None
        self.alarm_active = False
        self.elapsed = 0.0
        self.eyes_closed = False
        self.face_found = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        self.face_found = False
        self.eyes_closed = False
        self.elapsed = 0.0

        if results.multi_face_landmarks:
            self.face_found = True
            lms = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(lms, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            self.eyes_closed = avg_ear < EAR_THRESHOLD

            if self.eyes_closed:
                if self.eyes_closed_since is None:
                    self.eyes_closed_since = time.time()
                self.elapsed = time.time() - self.eyes_closed_since
                self.alarm_active = self.elapsed >= CLOSED_THRESHOLD_SECONDS
            else:
                self.eyes_closed_since = None
                self.alarm_active = False
        else:
            self.eyes_closed_since = None
            self.alarm_active = False

        draw_overlay(img, self.eyes_closed, self.elapsed,
                     CLOSED_THRESHOLD_SECONDS, self.alarm_active, self.face_found)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Main app layout ───────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="eye-blink-alarm",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EyeBlinkProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### Status")
    status_placeholder = st.empty()
    audio_placeholder = st.empty()

# ─── Live status + alarm sound loop ────────────────────────────────────────
if ctx.state.playing:
    last_played_flag = st.session_state.get("last_alarm_flag", False)

    while ctx.state.playing:
        if ctx.video_processor:
            vp = ctx.video_processor
            if vp.alarm_active:
                status_placeholder.error(f"🚨 ALARM! Eyes closed {vp.elapsed:.1f}s")
                b64 = make_beep_b64(ALARM_BEEP_HZ, ALARM_DURATION_MS)
                audio_placeholder.markdown(
                    f"""<audio autoplay="true">
                            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                        </audio>""",
                    unsafe_allow_html=True,
                )
            elif vp.eyes_closed:
                status_placeholder.warning(f"Eyes closed: {vp.elapsed:.1f}s / {CLOSED_THRESHOLD_SECONDS:.0f}s")
                audio_placeholder.empty()
            elif not vp.face_found:
                status_placeholder.info("No face detected")
                audio_placeholder.empty()
            else:
                status_placeholder.success("Eyes open — all good")
                audio_placeholder.empty()
        time.sleep(0.5)
else:
    st.info("Click **START** above to begin monitoring. Allow camera access when prompted.")

st.markdown("---")
st.caption(
    "Note: browsers may block autoplaying audio until you interact with the page once "
    "(e.g. clicking START). If you don't hear the alarm, click anywhere on the page first."
)