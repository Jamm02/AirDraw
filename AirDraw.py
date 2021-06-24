#complete code
import numpy as np 
import cv2
from collections import deque
index = 0
points = [deque(maxlen=1024)]
video = cv2.VideoCapture(1)
cX = 0
cY = 0

while(video.isOpened()):
    _, frame = video.read()
    cv2.imshow("Image",frame)
    if cv2.waitKey(10) == 13:
        bbox = cv2.selectROI(frame)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        obj_img = hsv[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        h, s, v = np.median(obj_img[:,:,0]), np.median(obj_img[:,:,1]), np.median(obj_img[:,:,2])
        lower = np.array([h-5, max(0,s-50),max(0,v-50)])
        upper = np.array([h+5, min(s+50,255),min(v+50,255)])
        print(lower,upper)
        break


while(video.isOpened()):
    _, frame = video.read()
    frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masked = cv2.inRange(hsv, lower, upper)
    blur = cv2.medianBlur(masked, 5)
    blob_mask = cv2.bitwise_and(frame,frame,mask=blur)
    cv2.imshow("blob_mask",blob_mask)
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        M = cv2.moments(c)
        if M["m00"]!=0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
   
    idx, current_max, counter = 0, 0, 0

    for n in contours:
        area = cv2.contourArea(n)
        if area > current_max:
            current_max = area
            idx = counter
        counter += 1
    
    cv2.drawContours(frame, contours, idx, (0,255,255),2)

    if len(contours)==0:
        cX=0
        cY=0
    if len(contours) > 0:
        if(40 <= center[0] <= 140 and 40 <= center[1] <= 80):
            points = [deque(maxlen=512)] 
            index = 0

        points[index].appendleft(center)
    else:
        points.append(deque(maxlen=512))
        index += 1
    
    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
               
 
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(frame, points[j][k - 1], points[j][k], (255,0,0), 2)
    cv2.imshow("Output",frame)

    if cv2.waitKey(10) == ord('x'):
        cv2.destroyAllWindows()
        video.release()
        break
