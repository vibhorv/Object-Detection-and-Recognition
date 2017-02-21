import json
k=0

frame_dict  = {}
with open('datasample1.json') as json_data:
        d = json.load(json_data)
        for item in d:
            label = d[item]["label"]
            for frame in d[item]["boxes"]:
                if not frame in frame_dict:
                    frame_dict[frame] = []
                print frame, label
                temp_list = [d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
                frame_dict[frame].append(temp_list)

print frame_dict
import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture('1.mp4')
fr=0
while(cap.isOpened()):
    ret, frame = cap.read()
    fp=str(fr)
    fr=fr+1
    for name in frame_dict[fp]:
        x1=name[0]
        y1=name[1]
        x2=name[2]
        y2=name[3]
        label=name[4]
        img = Image.fromarray(frame)
        img=img.crop((x1,y1,x2,y2))
        if label=='Car':
            img.save('./Dataset/Car/img{:>05}.jpg'.format(k))
            k=k+1
        if label=='Person':
            img.save('./Dataset/Person/img{:>05}.jpg'.format(k))
            k=k+1
        if label=='Motorcycle':
            img.save('./Dataset/Motorcycle/img{:>05}.jpg'.format(k))
            k=k+1
        if label=='Bicycle':
            img.save('./Dataset/Cycle/img{:>05}.jpg'.format(k))
            k=k+1
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




frame_dict  = {}
with open('input_video_sample2.json') as json_data:
        d = json.load(json_data)
        for item in d:
            label = d[item]["label"]
            for frame in d[item]["boxes"]:
                if not frame in frame_dict:
                    frame_dict[frame] = []
                print frame, label
                temp_list = [d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
                frame_dict[frame].append(temp_list)


cap = cv2.VideoCapture('2.mp4')
fr=0
while(cap.isOpened()):
    ret, frame = cap.read()
    fp=str(fr)
    fr=fr+1
    for name in frame_dict[fp]:
        x1=name[0]
        y1=name[1]
        x2=name[2]
        y2=name[3]
        label=name[4]
        img = Image.fromarray(frame)
        img=img.crop((x1,y1,x2,y2))
        if label=='Car':
            img.save('./Dataset/Car/img{:>05}.jpg'.format(k))
            k=k+1
        if label=='Person':
            img.save('./Dataset/Person/img{:>05}.jpg'.format(k))
            k=k+1
        if label=='Motorcycle':
            img.save('./Dataset/Motorcycle/img{:>05}.jpg'.format(k))
            k=k+1
        if label=='Bicycle':
            img.save('./Dataset/Cycle/img{:>05}.jpg'.format(k))
            k=k+1
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()