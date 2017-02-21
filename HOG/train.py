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
from sklearn.cross_validation import KFold

image = cv2.imread("test.jpg",0)
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
temp=np.zeros((3099,50000),dtype=np.float64)
pi=0
for root, dirs, files in os.walk(path1, topdown=False):
	for name in files:
		if ".jpg" in (os.path.join(root, name)):
			im1=Image.open((os.path.join(root, name))).convert('1')
			Y.append(1)
			name2.append(name)
			pix1 = im1.load()
			col1,row1 = im1.size
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			print Y1.shape
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=pix1[a,b]
			Y1=np.array(Y1)
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]

			pi=pi+1
			
	        

for root, dirs, files in os.walk(path2, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=Image.open((os.path.join(root, name))).convert('1')
			Y.append(2)
			name2.append(name)
			pix1 = im1.load()
			col1,row1 = im1.size
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=pix1[a,b]
			Y1=np.array(Y1)
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
				#print temp[pi][i],X1[i][0]
			pi=pi+1
	        

for root, dirs, files in os.walk(path3, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=Image.open((os.path.join(root, name))).convert('1')
			Y.append(3)
			name2.append(name)
			pix1 = im1.load()
			col1,row1 = im1.size
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=pix1[a,b]
			Y1=np.array(Y1)
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
			pi=pi+1
	        

for root, dirs, files in os.walk(path4, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=Image.open((os.path.join(root, name))).convert('1')
			Y.append(4)
			name2.append(name)
			pix1 = im1.load()
			col1,row1 = im1.size
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=pix1[a,b]
			Y1=np.array(Y1)
			Y1=np.array(Y1)
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
			pi=pi+1
	        

for root, dirs, files in os.walk(path5, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=Image.open((os.path.join(root, name))).convert('1')
			Y.append(5)
			name2.append(name)
			pix1 = im1.load()
			col1,row1 = im1.size
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=pix1[a,b]
			Y1=np.array(Y1)
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
			pi=pi+1
	        

for root, dirs, files in os.walk(path6, topdown=False):
	for name in files:
		if ext in (os.path.join(root, name)):
			im1=Image.open((os.path.join(root, name))).convert('1')
			Y.append(6)
			name2.append(name)
			pix1 = im1.load()
			col1,row1 = im1.size
			Y1=np.zeros((col1,row1),dtype=np.uint8)
			for a in range(0,col1):
				for b in range(0,row1):
					Y1[a][b]=pix1[a,b]
			Y1=np.array(Y1)
			X1=hog.compute(Y1,winStride,padding,locations)
			al=X1.size
			for i in range(0,al):
				temp[pi][i]=X1[i][0]
			pi=pi+1

Y=np.array(Y)
l=Y.size
u=X1.size

X=np.array(temp)
print X.shape
print l
clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
kf = KFold(l, n_folds=2,shuffle=True)
for train_index, test_index in kf:
	X_train, X_test = 	X[train_index], X[test_index]
	y_train, y_test =   Y[train_index],Y[test_index]
	clf = clf.fit(X_train,y_train)
	Ls= clf.predict(X_test)
	eff=0.0
	l=Ls.size
	for a in range(0,l):
		if Ls[a]==Y[a]:
			eff=eff+1.0
	print eff*100.0/float(l)

#for a in range(0,l):
	#print np.sum(X[a]),


