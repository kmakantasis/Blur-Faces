# -*- coding: utf-8 -*-

import cv2

def Detect(img, img_color, map_x_32, map_y_32):  
    cascade_alt = cv2.CascadeClassifier()
    cascade_alt.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_alt.xml")
    rects = cascade_alt.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    
    #cascade_alt2 = cv2.CascadeClassifier()
    #cascade_alt2.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_alt2.xml")
    #rects_add = cascade_alt2.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    #if len(rects_add) != 0:
    #    rects = np.concatenate((rects, rects_add), axis=0)     
    
    #cascade_alt_tree = cv2.CascadeClassifier()
    #cascade_alt_tree.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_alt_tree.xml")
    #rects_add = cascade_alt_tree.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    #if len(rects_add) != 0:
    #    rects = np.concatenate((rects, rects_add), axis=0)     
    
    #cascade_default = cv2.CascadeClassifier()
    #cascade_default.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_frontalface_default.xml")
    #rects_add = cascade_default.detectMultiScale(img, 1.1, 3, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    #if len(rects_add) != 0:
    #    rects = np.concatenate((rects, rects_add), axis=0) 
    
    cascade_prof = cv2.CascadeClassifier()
    cascade_prof.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_profileface.xml")
    rects_prof = cascade_prof.detectMultiScale(img, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    
    img_r = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_AREA) 
    cascade_prof2 = cv2.CascadeClassifier()
    cascade_prof2.load("/home/kostas/PythonProjects/BlurFaces/haarcascades/haarcascade_profileface.xml")
    rects_prof2 = cascade_prof.detectMultiScale(img_r, 1.1, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (3,3))
    
    if len(rects_prof2) != 0:
        rects_prof2[:, 2:] += rects_prof2[:, :2]
    #box(rects_prof2, img_r)  
    img = cv2.remap(img_r, map_x_32, map_y_32, cv2.INTER_AREA) 


    
    if len(rects) != 0:
        rects[:, 2:] += rects[:, :2]
    #box(rects, img)    
    
    if len(rects_prof) != 0:
        rects_prof[:, 2:] += rects_prof[:, :2]
    #box(rects_prof, img)   
    
    
    for x1, y1, x2, y2 in rects_prof:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
        
    for x1, y1, x2, y2 in rects_prof2:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
    
    for x1, y1, x2, y2 in rects:
        roi = img_color[y1:y2, x1:x2, :]
        roi = cv2.blur(roi,(20,20))
        img_color[y1:y2, x1:x2, :] = roi
        
 
    return img_color

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
        
    
    
    
    
    
