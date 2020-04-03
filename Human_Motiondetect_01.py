import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
key = 'a'
cX, cY = 300, 300
centre_list = [0,0]
X,Y,W,H = 200,200,10,10
isflag = False

def add_contours(x,y,w,h,cX,cY):
    W = max(abs(cX - x),abs(cX-x-w))
    H = max(abs(cY- y),abs(cY-x-h))
    return W,H

def centroid_calc(thd):
    M = cv.moments(thd)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 200, 200
    return cX, cY

while key != ord('q'):
    ret1, frame1 = cap.read()
    cv.waitKey(10)
    ret2, frame2 = cap.read()
    diff_img = cv.subtract(frame1, frame2)
    gray_img = cv.cvtColor(diff_img, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), 2)
    ret3, thd = cv.threshold(blur_img, 50, 255, cv.THRESH_BINARY)
    
    contour, not_used = cv.findContours(thd, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    wsum, hsum = 0,0
    i = 1
    for contour_var in contour:
        x,y,w,h = cv.boundingRect(contour_var)
        if cv.contourArea(contour_var) > 100:
            cX, cY = centroid_calc(thd)
            i += 1
            isflag = True
            W, H = add_contours(x,y,w,h,cX,cY)
            wsum += W
            hsum += H

    if isflag == True:
        cv.circle(frame1,(int(cX),int(cY)), 7, (0,0,255), 2)
        W = wsum/i + 100
        H = hsum/i + 100
        cv.putText(frame1,'Movement Found',(20,20),cv.FONT_HERSHEY_SIMPLEX, 1 ,(0,0,0),1)
        isflag = False
    
    cv.rectangle(frame1,(int(cX-W/2-10),int(cY-H/2-10)),(int(cX+W/2+10),int(cY+H/2+10)), (0,200,0), 2)
    cv.imshow('Window_roi', thd)
    cv.imshow('Window_0', frame1)
    key = cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
