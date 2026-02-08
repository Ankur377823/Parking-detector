import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from utils import get_parking_spots_bboxes, empty_or_not

st.set_page_config(page_title="Parking Detector", layout="centered")
st.title("🚗 High-Quality Parking Detection")

# --- PATHS ---
input_video_path = "compressed_video.mp4" # Ensure this file exists
mask_path = "mask.png"
output_video_path = "output_processed.mp4"

# --- CHECK FILES ---
if not os.path.exists(mask_path):
    st.error("❌ mask.png not found!")
    st.stop()
if not os.path.exists(input_video_path):
    st.error(f"❌ Video {input_video_path} not found!")
    st.stop()

# --- LOAD MASK ---
mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# --- APP LOGIC ---
st.markdown("### Click below to process the video and watch the smooth result.")

if st.button("🎬 Process & Play Video"):
    # Open Video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup Video Writer (to save the result)
    # H.264 is needed for web, but OpenCV writes .mp4 easily. 
    # If this fails on Cloud, we will try 'avc1'
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

        # --- DETECTION LOGIC (Same as before) ---
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

        # --- SAVE FRAME TO FILE ---
        out.write(frame)

        # Update Progress
        frame_nmr += 1
        if frame_nmr % 10 == 0: # Update bar every 10 frames to save speed
            progress_bar.progress(min(frame_nmr / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_nmr}/{total_frames}...")

    # Cleanup
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()

    # --- RE-ENCODE FOR BROWSER (CRITICAL STEP) ---
    # OpenCV creates MP4s that browsers sometimes hate. 
    # We use ffmpeg (installed on Streamlit Cloud) to fix it.
    st.info("Optimizing video for web playback...")
    os.system(f"ffmpeg -y -i {output_video_path} -vcodec libx264 final_output.mp4")

    # --- PLAY VIDEO ---
    st.success("✅ Processing Complete! Watch below:")
    st.video("final_output.mp4")
