import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from utils import get_parking_spots_bboxes, empty_or_not

# --- PAGE CONFIG ---
st.set_page_config(page_title="Parking Detector", layout="centered")
st.title("🚗 High-Quality Parking Detection")

# --- PATHS ---
input_video_path = "compressed_video.mp4" # Ensure this matches your file on GitHub
mask_path = "mask.png"
output_video_path = "final_output.mp4"

# --- CHECK FILES ---
if not os.path.exists(mask_path):
    st.error("❌ mask.png not found!")
    st.stop()
if not os.path.exists(input_video_path):
    st.error(f"❌ Video {input_video_path} not found!")
    st.stop()

# --- 1. LOAD MASK & GET SPOTS (YOUR LOGIC) ---
mask = cv2.imread(mask_path, 0)
# This is your specific logic to find spots from the mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# --- APP INTERFACE ---
st.markdown(f"### Found {len(spots)} parking spots from mask.")
st.markdown("Click below to process the video using your model logic.")

mode = st.radio("Choose Mode:", ["Process & Play Smoothly (Recommended)", "View Live (Choppy)"])

if st.button("🚀 Start Detection"):
    
    # Open Video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup Video Writer (For Smooth Mode)
    if mode == "Process & Play Smoothly (Recommended)":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        progress_bar = st.progress(0)
        status_text = st.empty()
    else:
        # Live Mode setup
        st_frame = st.empty()
        st_status = st.empty()

    # --- INITIALIZE VARIABLES ---
    frame_nmr = 0
    step = 30  # Run detection every 30 frames
    previous_frame = None
    spots_status = [True for _ in spots]
    diffs = [0 for _ in spots]

    # --- MAIN LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- YOUR CORE LOGIC ---
        if frame_nmr % step == 0:
            if previous_frame is not None:
                for idx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    # YOUR DIFF LOGIC
                    diff = np.abs(np.mean(frame[y1:y1+h, x1:x1+w]) - np.mean(previous_frame[y1:y1+h, x1:x1+w]))
                    diffs[idx] = diff
                
                max_d = np.max(diffs) if np.max(diffs) > 0 else 1
                arr_ = [j for j in range(len(spots)) if diffs[j] / max_d > 0.4]
                
                for spot_indx in arr_:
                    x1, y1, w, h = spots[spot_indx]
                    spot_crop = frame[y1:y1+h, x1:x1+w]
                    spots_status[spot_indx] = empty_or_not(spot_crop)
            
            if previous_frame is None:
                for i in range(len(spots)):
                    x1, y1, w, h = spots[i]
                    spot_crop = frame[y1:y1+h, x1:x1+w]
                    spots_status[i] = empty_or_not(spot_crop)
            
            previous_frame = frame.copy()

        # --- DRAWING ---
        for idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            color = (0, 255, 0) if spots_status[idx] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

        # --- OUTPUT HANDLING ---
        if mode == "Process & Play Smoothly (Recommended)":
            # Save frame to file
            out.write(frame)
            # Update Progress Bar
            if frame_nmr % 10 == 0:
                progress_bar.progress(min(frame_nmr / total_frames, 1.0))
                status_text.text(f"Processing frame {frame_nmr}/{total_frames}...")
        else:
            # Live Mode: Update Streamlit Image directly
            # Resize for speed (Critical for Live Mode)
            frame_small = cv2.resize(frame, (700, int(700 * (height / width))))
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB")
            
            # Update counters
            available = sum(spots_status)
            st_status.markdown(f"### Available: {available}/{len(spots)}")

        frame_nmr += 1

    # Cleanup
    cap.release()
    if mode == "Process & Play Smoothly (Recommended)":
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        # --- PLAY THE RESULT ---
        st.success("✅ Detection Complete! Playing Video:")
        
        # Attempt to convert for web compatibility if ffmpeg is present
        if os.system("ffmpeg -version") == 0:
            os.system(f"ffmpeg -y -i {output_video_path} -vcodec libx264 final_web.mp4")
            st.video("final_web.mp4")
        else:
            # Fallback (Might not play in all browsers)
            st.video(output_video_path)
