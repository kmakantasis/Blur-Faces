# -*- coding: utf-8 -*-

import cv2
import numpy as np
import kmDetection   


cap = cv2.VideoCapture('/home/kostas/PythonProjects/BlurFaces/DSC_0239.MOV')
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

map_x = np.asarray(range(gray.shape[1]), dtype='float32')
map_x = gray.shape[1] - map_x - 1
map_x_32 = np.tile(map_x, (gray.shape[0],1))
map_y = np.asarray(range(gray.shape[0]), dtype='float32')
map_y_32 = np.tile(map_y, (gray.shape[1],1))
map_y_32 = np.transpose(map_y_32)

count = 0
while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_det = kmDetection.Detect(gray, frame, map_x_32, map_y_32)


    cv2.imshow('img_det',img_det)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    filename = "results/img_det%d.jpg" % count
    cv2.imwrite(filename, img_det)
    
    count = count + 1

cap.release()
cv2.destroyAllWindows()




