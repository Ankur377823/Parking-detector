import cv2
from utils import get_parking_spots_bboxes,empty_or_not
import numpy as np
import matplotlib.pyplot as plt

def calc_diffs(im1,im2):
    return np.abs(np.mean(im1)- np.mean(im2))# mean of all pixels diff (how diff two img are)

mask=r"C:\Users\ankur\Downloads\mask_1920_1080.png"

video_path=r"C:\Users\ankur\Downloads\parking_1920_1080_loop.mp4"

mask=cv2.imread(mask,0)


cap=cv2.VideoCapture(video_path)

connected_components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S) # mask is as similar as graph

spots=get_parking_spots_bboxes(connected_components)

spots_status=[None for j in spots]

diffs=[None for j in spots]

previous_frame=None

frame_nmr=0
ret=True
step=30  #at 30 sec intervala
while ret: #as parking of car takes  some time usually 4-5 sec so we don,t need to use each frame
    ret,frame=cap.read()

    if frame_nmr %step==0 and previous_frame is not None:
        for spot_indx,spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx]=calc_diffs(spot_crop,previous_frame[y1:y1 + h, x1:x1 + w, :])
    #print(diffs) tells how diff previous frame was
        print([diffs[j] for j in np.argsort(diffs)[::-1]])


    if frame_nmr %step==0:
        if(previous_frame is None):
            arr_=range(len(spots))
        else:
            arr_=[j for j in np.argsort(diffs) if diffs[j]/np.amax(diffs)>0.4]
        for spot_indx in arr_: #handling outliers by polling histo in pervious code
            spot=spots[spot_indx]
            x1,y1,w,h=spot

            spot_crop=frame[y1:y1+h,x1:x1+w,:]

            spot_status=empty_or_not(spot_crop)
            spots_status[spot_indx]=spot_status

    if frame_nmr % step == 0:
        previous_frame=frame.copy()
    for spot_indx,spot in enumerate(spots):
        spot_status=spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0),
                                  2)  # drawing bounding boxed near all parking spots

        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255),
                                  2)  # drawing bounding boxed near all parking spots


    cv2.rectangle(frame,(80,20),(550,80),(0,0,0),-1)
    cv2.putText(frame,'Available Spots: {} / {}'.format(str(sum(spots_status)),str(len(spots))),(100,65),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
    frame_nmr+=1

cap.release()

