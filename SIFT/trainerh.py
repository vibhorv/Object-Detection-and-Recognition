import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

sift = cv2.xfeatures2d.SIFT_create(150)
Y=[]
descriptors = np.zeros((1,128))
k=0
path1="./Dataset/Autorikshaw"
path2="./Dataset/Car"
path3="./Dataset/Cycle"
path4="./Dataset/Motorcycle"
path5="./Dataset/Person"
path6="./Dataset/Rikshaw"



for root, dirs, files in os.walk(path2, topdown=False):
  for name in files:
    if ".jpg" in (os.path.join(root, name)):
      im1=cv2.imread((os.path.join(root, name)))
      im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      kp, des = sift.detectAndCompute(im1,None)
      img3 = np.zeros((1,1))
      row,col= des.shape
      descriptors = np.concatenate((descriptors,des),axis=0)
      img=cv2.drawKeypoints(im1,kp,img3,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      

s,h=descriptors.shape
for a in range(s-k):
  Y.append(1)
  k=k+1

for root, dirs, files in os.walk(path3, topdown=False):
  for name in files:
    if ".jpg" in (os.path.join(root, name)):
      im1=cv2.imread((os.path.join(root, name)))
      im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      kp, des = sift.detectAndCompute(im1,None)
      img3 = np.zeros((1,1))
      row,col= des.shape
      descriptors = np.concatenate((descriptors,des),axis=0)
      img=cv2.drawKeypoints(im1,kp,img3,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      

s,h=descriptors.shape

for a in range(s-k):
  Y.append(2)
  k=k+1

for root, dirs, files in os.walk(path4, topdown=False):
  for name in files:
    if ".jpg" in (os.path.join(root, name)):
      im1=cv2.imread((os.path.join(root, name)))
      im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      kp, des = sift.detectAndCompute(im1,None)
      img3 = np.zeros((1,1))
      row,col= des.shape
      descriptors = np.concatenate((descriptors,des),axis=0)
      img=cv2.drawKeypoints(im1,kp,img3,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      

s,h=descriptors.shape

for a in range(s-k):
  Y.append(4)
  k=k+1

for root, dirs, files in os.walk(path5, topdown=False):
  for name in files:
    if ".jpg" in (os.path.join(root, name)):
      im1=cv2.imread((os.path.join(root, name)))
      im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      kp, des = sift.detectAndCompute(im1,None)
      img3 = np.zeros((1,1))
      row,col= des.shape
      descriptors = np.concatenate((descriptors,des),axis=0)
      img=cv2.drawKeypoints(im1,kp,img3,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      

s,h=descriptors.shape

for a in range(s-k):
  Y.append(5)
  k=k+1


print k
print descriptors.shape

print "hui gava bc"



#clf = KNeighborsClassifier(n_neighbors=2)
clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

clas=clf.fit(descriptors, Y) 




cap = cv2.VideoCapture('1.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=19, detectShadows= False)
k=0
label=0
print 'Classifier Trained'
count=0
while(cap.isOpened()):
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    frame = cv2.resize(frame,(900, 700), interpolation = cv2.INTER_CUBIC)
    fgmask = fgbg.apply(frame)
    #fgmask = fgbg.apply(frame,learningRate=0.5)
    #fg1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    fgamsk= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
    #fgmask = cv2.dilate(fgmask,kernel,iterations = 1)
    
    i,cnts,u = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    mg2_fg = cv2.bitwise_and(frame,frame,mask = fgmask)
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 8000:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        

        img = cv2.cvtColor(mg2_fg, cv2.COLOR_BGR2GRAY)
    
        im=img[y:y+h,x:x+w]
        kp, des = sift.detectAndCompute(im,None)
        start_time = timeit.default_timer()
        label=clas.predict_proba(des)
        elapsed = timeit.default_timer() - start_time
        print elapsed,
        one=np.zeros(4,dtype=np.uint8)
        for a in range(0,label.shape[0]):
          one=one+label[a]
        one[3]=0.75*one[3]
        l=np.argmax(one)
        l=l+2
        print 
        
        if l==1:
          cv2.putText(frame,"Autorikshaw",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          img.save('./Dataset2/Autorikshaw/img{:>05}.jpg'.format(count))
          count=count+1
        elif l==2:
          cv2.putText(frame,"Car",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          img.save('./Dataset2/Car/img{:>05}.jpg'.format(count))
          count=count+1
        elif l==3:
          cv2.putText(frame,"Cycle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          img.save('./Dataset2/Cycle/img{:>05}.jpg'.format(count))
          count=count+1
        elif l==4:
          cv2.putText(frame,"Motorcycle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          img.save('./Dataset2/Motorcycle/img{:>05}.jpg'.format(count))
          count=count+1
        elif l==5:
          cv2.putText(frame,"Person",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
          img.save('./Dataset2/Person/img{:>05}.jpg'.format(count))
          count=count+1

        elif l==6:
          cv2.putText(frame,"Rikshaw",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)      
          img.save('./Dataset2/Rikshaw/img{:>05}.jpg'.format(count))
          count=count+1
    


   

    cv2.imshow('Object frame',frame)
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


