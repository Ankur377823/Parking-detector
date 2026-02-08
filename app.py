import streamlit as st
import cv2
import numpy as np
import os
from utils import get_parking_spots_bboxes, empty_or_not

st.set_page_config(page_title="Parking Detector", layout="centered")
st.title("🚗 High-Quality Parking Detection")

# --- PATHS ---
# Make sure this matches the file you uploaded (e.g., parking_short.mp4)
input_video_path = "compressed_video.mp4" 
mask_path = "mask.png"
output_video_path = "final_output.mp4"

# --- CHECK FILES ---
if not os.path.exists(mask_path):
    st.error("❌ mask.png not found! Please upload it to GitHub.")
    st.stop()
if not os.path.exists(input_video_path):
    st.error(f"❌ Video file '{input_video_path}' not found! Check your filename on GitHub.")
    st.stop()

# --- LOAD MASK ---
mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# --- APP LOGIC ---
st.markdown("### Click below to process.")

if st.button("🎬 Process & Play Video"):
    # Open Video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- VIDEO WRITER SETUP ---
    # We try 'avc1' (H.264) first because it works in browsers natively.
    # If that fails, we fall back to 'mp4v'.
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # If avc1 fails to initialize, fall back to mp4v
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_nmr = 0
    step = 30
    previous_frame = None
    spots_status = [True for _ in spots]
    diffs = [0 for _ in spots]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- DETECTION LOGIC ---
        if frame_nmr % step == 0:
            if previous_frame is not None:
                for idx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    diff = np.abs(np.mean(frame[y1:y1+h, x1:x1+w]) - np.mean(previous_frame[y1:y1+h, x1:x1+w]))
                    diffs[idx] = diff
                max_d = np.max(diffs) if np.max(diffs) > 0 else 1
                arr_ = [j for j in range(len(spots)) if diffs[j] / max_d > 0.4]
                for spot_indx in arr_:
                    x1, y1, w, h = spots[spot_indx]
                    spots_status[spot_indx] = empty_or_not(frame[y1:y1+h, x1:x1+w])
            
            if previous_frame is None:
                for i in range(len(spots)):
                    x1, y1, w, h = spots[i]
                    spots_status[i] = empty_or_not(frame[y1:y1+h, x1:x1+w])
            previous_frame = frame.copy()

        # --- DRAWING ---
        for idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            color = (0, 255, 0) if spots_status[idx] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

        # Write to file
        out.write(frame)

        # Update Progress Bar (Every 10 frames)
        frame_nmr += 1
        if frame_nmr % 10 == 0:
            progress_bar.progress(min(frame_nmr / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_nmr}/{total_frames}...")

    # Cleanup
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    # --- FINAL CHECK AND PLAY ---
    if os.path.exists(output_video_path):
        st.success("✅ Processing Complete!")
        # We try to convert with ffmpeg ONLY if it's installed, otherwise play raw
        if os.system("ffmpeg -version") == 0:
            temp_name = "temp_ffmpeg.mp4"
            os.system(f"ffmpeg -y -i {output_video_path} -vcodec libx264 {temp_name}")
            if os.path.exists(temp_name):
                st.video(temp_name)
            else:
                st.video(output_video_path)
        else:
            # Fallback: Just play what we have
            st.video(output_video_path)
    else:
        st.error("❌ Failed to save video file. Please check permissions.")
