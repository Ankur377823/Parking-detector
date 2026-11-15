# Parking Detector

Detects which parking spots are empty or occupied in a video of a car park and shows a live count of available spots. Built with OpenCV and a scikit-learn classifier, with a Streamlit interface.

## How it works

1. **Find the spots.** A binary mask image marks every parking spot. Connected-component analysis on the mask gives a bounding box for each one.
2. **Classify each spot.** Each spot is cropped from the frame, resized to 15 x 15 pixels and passed to a pre-trained classifier (`model.p`) that labels it empty or occupied.
3. **Skip unchanged spots.** Frames are sampled every 30 frames, and a spot is re-classified only when its pixels have changed noticeably since the last sample, which keeps processing fast.
4. **Draw the result.** Empty spots are outlined in green and occupied spots in red, with an "Available Spots" counter on the frame.

## Tech stack

Python, OpenCV, scikit-learn, scikit-image, NumPy, Streamlit

## Run locally

```bash
git clone https://github.com/Ankur377823/Parking-detector.git
cd Parking-detector
pip install -r requirements.txt
streamlit run app.py
```

The app processes the bundled sample video and plays back the annotated result.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface that processes and plays the video |
| `main.py` | Standalone OpenCV script for running the detector in a desktop window |
| `utils.py` | Spot extraction from the mask and the empty/occupied classifier |
| `model.p` | Trained classifier |
| `mask.png` | Mask marking the parking spots |
| `compressed_video.mp4` | Sample car park video |

## License

See [LICENSE](LICENSE).
