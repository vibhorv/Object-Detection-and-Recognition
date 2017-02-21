import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



sift = cv2.xfeatures2d.SIFT_create(150)
cap = cv2.VideoCapture('1.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=19, detectShadows= False)
k=0
label=0

import pickle
with open('classifier.pkl', 'rb') as f:
    clas = pickle.load(f)


print "clas loaded"
while(cap.isOpened()):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    frame = cv2.resize(frame,(900, 700), interpolation = cv2.INTER_CUBIC)
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    fgamsk= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
    
    i,cnts,u = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    mg2_fg = cv2.bitwise_and(frame,frame,mask = fgmask)
    for c in cnts:
        if cv2.contourArea(c) < 8000:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        

        img = cv2.cvtColor(mg2_fg, cv2.COLOR_BGR2GRAY)
    
        im=img[y:y+h,x:x+w]
        kp, des = sift.detectAndCompute(im,None)
        start_time = timeit.default_timer()
        label=clas.predict_proba(des)
        elapsed = timeit.default_timer() - start_time
        one=np.zeros(8,dtype=np.uint8)
        for a in range(0,label.shape[0]):
          one=one+label[a]
        one[4]=0.75*one[4]
        l=np.argmax(one)
        l=l+1
        print 

        if l==1:
          cv2.putText(frame,"Autorikshaw",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Autorikshaw"
        elif l==2:
          cv2.putText(frame,"Car",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Car"
        elif l==3:
          cv2.putText(frame,"Cycle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Cycle"
        elif l==4:
          cv2.putText(frame,"Motorcycle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Motorcycle"
        elif l==5:
          cv2.putText(frame,"Person",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Person"
        elif l==6:
          cv2.putText(frame,"Rikshaw",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Rikshaw"
        elif l==8:
          cv2.putText(frame,"Scooty",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          txt="Scooty"
    

    print txt


   

    cv2.imshow('Object frame',frame)
    cv2.imshow('BGS frame',mg2_fg)
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


