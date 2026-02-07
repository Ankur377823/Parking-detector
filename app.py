import streamlit as st
import cv2
import numpy as np
import os
import time
from utils import get_parking_spots_bboxes, empty_or_not 

st.set_page_config(page_title="Parking Detector", layout="wide")
st.title("🚗 Real-Time Parking Detection")

# Setup paths
video_path = "compressed_video.mp4"
mask_path = "mask.png"

# Basic error checking
if not os.path.exists(mask_path):
    st.error("❌ Error: mask.png not found.")
    st.stop()

# Initialize data only once
mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots_status = [True for _ in spots]
diffs = [0 for _ in spots]

# Create UI placeholders outside the loop
st_status = st.empty()
st_frame = st.empty()

if st.button("Start Detection"):
    cap = cv2.VideoCapture(video_path)
    previous_frame = None
    frame_nmr = 0
    step = 30  
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop the video
            continue

        # --- Detection Logic (Every 30 frames) ---
        if frame_nmr % step == 0:
            if previous_frame is not None:
                for idx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    # Calculate pixel difference to see if spot needs re-checking
                    diff = np.abs(np.mean(frame[y1:y1+h, x1:x1+w]) - np.mean(previous_frame[y1:y1+h, x1:x1+w]))
                    diffs[idx] = diff

                max_d = np.amax(diffs) if np.amax(diffs) > 0 else 1
                arr_ = [j for j in range(len(spots)) if diffs[j] / max_d > 0.4]
                
                for spot_indx in arr_:
                    x1, y1, w, h = spots[spot_indx]
                    spots_status[spot_indx] = empty_or_not(frame[y1:y1+h, x1:x1+w])
            
            # Initial detection on first frame
            if previous_frame is None:
                for i in range(len(spots)):
                    x1, y1, w, h = spots[i]
                    spots_status[i] = empty_or_not(frame[y1:y1+h, x1:x1+w])
            
            previous_frame = frame.copy()

        # --- Drawing Overlays ---
        for idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            color = (0, 255, 0) if spots_status[idx] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

        # --- UI Updates ---
        available_count = sum(spots_status)
        st_status.markdown(f"### Available Spots: **{available_count} / {len(spots)}**")
        
        # Display the frame in the placeholder
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB")
        
        # Small delay to keep the browser responsive
        time.sleep(0.01)
        frame_nmr += 1

    cap.release()
