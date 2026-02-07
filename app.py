import streamlit as st
import cv2
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from utils import get_parking_spots_bboxes, empty_or_not

st.set_page_config(page_title="Parking Detector", layout="wide")
st.title("🚗 Real-Time Parking Detection")

# Load your static files
mask_path = "mask.png"
if not os.path.exists(mask_path):
    st.error("Missing mask.png")
    st.stop()

mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

class ParkingProcessor(VideoTransformerBase):
    def __init__(self):
        self.spots_status = [True for _ in spots]
        self.frame_nmr = 0
        self.previous_frame = None
        self.diffs = [0 for _ in spots]

    def transform(self, frame):
        # Convert the WebRTC frame to a standard OpenCV image
        img = frame.to_ndarray(format="bgr24")
        
        # Detection Logic (Every 30 frames)
        if self.frame_nmr % 30 == 0:
            if self.previous_frame is not None:
                for idx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    # Calculate difference to see if spot needs re-checking
                    diff = np.abs(np.mean(img[y1:y1+h, x1:x1+w]) - np.mean(self.previous_frame[y1:y1+h, x1:x1+w]))
                    self.diffs[idx] = diff

                max_d = np.amax(self.diffs) if np.amax(self.diffs) > 0 else 1
                arr_ = [j for j in range(len(spots)) if self.diffs[j] / max_d > 0.4] if self.previous_frame is not None else range(len(spots))
                
                for spot_indx in arr_:
                    x1, y1, w, h = spots[spot_indx]
                    # Call your model logic from utils.py
                    self.spots_status[spot_indx] = empty_or_not(img[y1:y1+h, x1:x1+w])
            
            self.previous_frame = img.copy()

        # Draw the overlays directly on the frame
        for idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            color = (0, 255, 0) if self.spots_status[idx] else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)
            
        self.frame_nmr += 1
        return img

# This replaces the while loop and the 'Start' button
webrtc_streamer(key="parking-detection", video_transformer_factory=ParkingProcessor)
