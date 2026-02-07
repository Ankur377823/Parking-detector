import streamlit as st
import cv2
import numpy as np
import os
import time
from utils import get_parking_spots_bboxes, empty_or_not 

st.set_page_config(page_title="Parking Detector", layout="wide")
st.title("🚗 Real-Time Parking Detection")

video_path = "compressed_video.mp4"
mask_path = "mask.png"

if not os.path.exists(mask_path):
    st.error(f"❌ Error: Mask file not found.")
    st.stop()

mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots_status = [True for _ in spots]
diffs = [0 for _ in spots]

previous_frame = None
frame_nmr = 0
step = 30  
display_freq = 3 # Only update the browser every 3rd frame to boost FPS

st_frame = st.empty()
st_status = st.empty()

if st.button("Start Detection"):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # DETECTION LOGIC (Every 30 frames)
        if frame_nmr % step == 0:
            if previous_frame is not None:
                for idx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1+h, x1:x1+w]
                    prev_crop = previous_frame[y1:y1+h, x1:x1+w]
                    diffs[idx] = np.abs(np.mean(spot_crop) - np.mean(prev_crop))

            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                max_d = np.amax(diffs) if np.amax(diffs) > 0 else 1
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / max_d > 0.4]
            
            for spot_indx in arr_:
                x1, y1, w, h = spots[spot_indx]
                spots_status[spot_indx] = empty_or_not(frame[y1:y1+h, x1:x1+w])
            
            previous_frame = frame.copy()

        # UI UPDATE (Every 3rd frame)
        if frame_nmr % display_freq == 0:
            for idx, spot in enumerate(spots):
                x1, y1, w, h = spot
                color = (0, 255, 0) if spots_status[idx] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

            available_count = sum(spots_status)
            st_status.markdown(f"### Available Spots: **{available_count} / {len(spots)}**")
            
            # Convert and push to browser
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")
        
        frame_nmr += 1

    cap.release()
