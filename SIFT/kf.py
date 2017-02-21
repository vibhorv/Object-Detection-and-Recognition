import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import timeit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



sift = cv2.xfeatures2d.SIFT_create(150)
k=0
label=0

import pickle
with open('classifier.pkl', 'rb') as f:
    clas = pickle.load(f)


print "Performing K-fold Verification"

path2="./Dataset_1/Car"
path3="./Dataset_1/Cycle"
path4="./Dataset_1/Motorcycle"
path5="./Dataset_1/Person"

case=0.0
t=0.0

for root, dirs, files in os.walk(path2, topdown=False):
  for name in files:
    if ".jpg" in (os.path.join(root, name)):
      im1=cv2.imread((os.path.join(root, name)))
      im1=cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
      kp, des = sift.detectAndCompute(im1,None)
      img3 = np.zeros((1,1))
      row,col= des.shape
      label=clas.predict_proba(des)
      case=case+1.0
      one=np.zeros(8,dtype=np.uint8)
      for a in range(0,label.shape[0]):
          one=one+label[a]
      one[3]=0.75*one[3]
      l=np.argmax(one)
      if l==2:
        t=t+1.0


print case
print t













