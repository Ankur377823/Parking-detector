import streamlit as st
import cv2
import numpy as np
import pickle
import os


from utils import get_parking_spots_bboxes, empty_or_not 

st.set_page_config(page_title="Parking Detector", layout="wide")
st.title("🚗 Real-Time Parking Detection")


st.sidebar.header("Configuration")


video_path = "parking_loop.mp4"
mask_path = "mask.png"
model_path = "model.p"


if not os.path.exists(mask_path):
    st.error(f"❌ Error: Mask file '{mask_path}' not found.")
    st.stop()
if not os.path.exists(model_path):
    st.error(f"❌ Error: Model file '{model_path}' not found.")
    st.stop()



mask = cv2.imread(mask_path, 0)

cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)
spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None
frame_nmr = 0
step = 30  # Process every 30 frames
ret = True


st_frame = st.empty()
st_status = st.empty()


if st.button("Start Detection"):
    while ret:
        ret, frame = cap.read()
        
        # Loop video if it ends
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret = True
            continue

        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                prev_crop = previous_frame[y1:y1 + h, x1:x1 + w, :]
                diffs[spot_indx] = np.abs(np.mean(spot_crop) - np.mean(prev_crop))

        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                max_diff = np.amax(diffs) if np.amax(diffs) > 0 else 1
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / max_diff > 0.4]
            
            for spot_indx in arr_:
                spot = spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = empty_or_not(spot_crop)
                spots_status[spot_indx] = spot_status

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

      
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            
            if spot_status: # Available
                color = (0, 255, 0) # Green
            else: # Occupied
                color = ( 0, 0,255) # Red
            
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        # Update Text Status
        available_count = sum([1 for s in spots_status if s is True])
        st_status.markdown(f"### Available Spots: **{available_count} / {len(spots)}**")

        # Display Frame (Convert BGR to RGB for Web)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_column_width=True)
        
        frame_nmr += 1

    cap.release()
