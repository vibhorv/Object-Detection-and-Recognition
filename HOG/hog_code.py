import os
from PIL import Image
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import cv2
from scipy import ndimage
import random
import struct
from sklearn import svm
from skimage import data, color, exposure
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import timeit

winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

path1="./Dataset/Autorikshaw"
path2="./Dataset/Car"
path3="./Dataset/Cycle"
path4="./Dataset/Motorcycle"
path5="./Dataset/Person"
path6="./Dataset/Rikshaw"
ext=".jpg"
X=[]
Y=[]
name2=[]
temp=np.zeros((265,50000),dtype=np.float64)
pi=0
for root, dirs, files in os.walk(path1, topdown=False):
	for name in files:
		if ".jpg" in (os.path.join(root, name)):
			im1=cv2.imread((os.path.join(root, name)))
			im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			Y.append(1)
			name2.append(name)
			col1,row1 = im1.shape
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=im1[a,b]
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
			pi=pi+1
	        
			
	        

for root, dirs, files in os.walk(path2, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=cv2.imread((os.path.join(root, name)))
			im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			Y.append(2)
			name2.append(name)
			col1,row1 = im1.shape
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=im1[a,b]
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
				#print temp[pi][i],X1[i][0]
			pi=pi+1
	        

for root, dirs, files in os.walk(path3, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=cv2.imread((os.path.join(root, name)))
			im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			Y.append(3)
			name2.append(name)
			col1,row1 = im1.shape
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=im1[a,b]
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
				#print temp[pi][i],X1[i][0]
			pi=pi+1
	        
	        

for root, dirs, files in os.walk(path4, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=cv2.imread((os.path.join(root, name)))
			im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			Y.append(4)
			name2.append(name)
			col1,row1 = im1.shape
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=im1[a,b]
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
				#print temp[pi][i],X1[i][0]
			pi=pi+1
	        
	        

for root, dirs, files in os.walk(path5, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=cv2.imread((os.path.join(root, name)))
			im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			Y.append(5)
			name2.append(name)
			col1,row1 = im1.shape
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=im1[a,b]
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
				#print temp[pi][i],X1[i][0]
			pi=pi+1
	        
	        

for root, dirs, files in os.walk(path6, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=cv2.imread((os.path.join(root, name)))
			im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			Y.append(6)
			name2.append(name)
			col1,row1 = im1.shape
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=im1[a,b]
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
				#print temp[pi][i],X1[i][0]
			pi=pi+1
	        
Y=np.array(Y)
l=Y.size
u=X1.size

X=np.array(temp)
print X.shape
print l
print "Training Sample Ready"

clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
clf = clf.fit(X,Y)
Ls= clf.predict(X)


eff=0.0
for a in range(0,l):
	if Ls[a]==Y[a]:
		eff=eff+1.0

print eff*100.0/float(l)



cap = cv2.VideoCapture('1.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorMOG2(history=500,varThreshold=19, detectShadows= False)
k=0
label=0
print 'Classifier Trained'

while(1):
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

        img = cv2.cvtColor(mg2_fg, cv2.COLOR_BGR2GRAY)
    
        im=img[y:y+h,x:x+w]
      
        cv2.imwrite('./data/1img{:>07}.jpg'.format(k),im)
        
        col,row=im.shape
        Y=np.zeros((col,row),dtype=np.uint8)
        for a in range(0,col):
        	for b in range(0,row):
        		Y[a][b]=im[a,b]
        Z1=hog.compute(Y,winStride,padding,locations)#
        k=k+1 # Dumps your pc :P
        Z12=np.zeros(50000,dtype=np.float64)
        for u in range(0,Z1.size):
        	Z12[u]=Z1[u][0]
      
        Z12=Z12.reshape(1,-1)
        start_time = timeit.default_timer()
        label=clf.predict(Z12)
        elapsed = timeit.default_timer() - start_time
        print elapsed
        k=k+1
        if label[0]==1:
        	cv2.putText(frame,"Autorikshaw",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif label[0]==2:
        	cv2.putText(frame,"Car",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif label[0]==3:
        	cv2.putText(frame,"Cycle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif label[0]==4:
        	cv2.putText(frame,"Motorcycle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif label[0]==5:
        	cv2.putText(frame,"Person",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
        elif label[0]==6:
        	cv2.putText(frame,"Rikshaw",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),thickness=2,lineType=8,bottomLeftOrigin=False)
    


   

    cv2.imshow('frame',frame)
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()