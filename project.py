import enum
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
path = "CVtask.jpg"
img = cv.imread(path)
r_img = cv.resize(img, (1000,1000))

def findAruco(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_5X5_250')
    aruco_dict = aruco.Dictionary_get(key)
    arucoPara = aruco.DetectorParameters_create()

    (corners, ids, rejected) = cv.aruco.detectMarkers(img, aruco_dict, parameters = arucoPara)

    print(ids)

    if len(corners)>0:
        ids = ids.flatten()

        for (markerCorner, markerId) in zip(corners,ids):
            corners = markerCorner.reshape((4,2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topLeft = (int(topLeft[0]) , int(topLeft[1]))
            topRight = (int(topRight[0]) , int(topRight[1]))
            bottomLeft = (int(bottomLeft[0]) , int(bottomLeft[1]))
            botomRight = (int(bottomRight[0]) , int(bottomRight[1]))

            cx = int((topLeft[0] + bottomRight[0])/2.0)
            cy = int((topLeft[1] + bottomRight[1])/2.0)

            # print(cx)
            # print(cy)

            bx = int((topLeft[0] + topRight[0])/2.0)
            by = int((topLeft[1] + topRight[1])/2.0)
            if (bx-cx)!=0:

              diff = (int(by - cy)/(bx - cx))
              slope = math.atan(diff)
              degrees = slope*(180/(math.pi))
              print(degrees)

def rotate(image,angle, rotPoint =None):
  (height,width) = img.shape[:2]

  if rotPoint is None:
    rotPoint = (300,300)
  
  rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
  dimensions = (width, height)
  return cv.warpAffine(image, rotMat, dimensions)





path1 = "Ha.jpg"
img1 = cv.imread(path1)
path2 = "HaHa.jpg"
img2 = cv.imread(path2)
path3 = "LMAO.jpg"
img3 = cv.imread(path3)
path4 = "XD.jpg"
img4 = cv.imread(path4)

cv.imshow('marker1', img1)

rotated2 = rotate(img2, 75.13620837855433)
cv.imshow('rotated2', rotated2)

rotated3 = rotate(img3,74.69665253769905 )
cv.imshow('rotated3', rotated3)

rotated4 = rotate(img4,-77.33763593714257 )
cv.imshow('rotated4', rotated4)


findAruco(img1)
findAruco(img2)
findAruco(img3)
findAruco(img4)

r_ar2 = cv.resize(img2, (1000,1000))
gray = cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray, 150 ,255)
dilated = cv.dilate(canny, (3,3), iterations = 2)
imgcontour = r_img.copy()
contours , hierarchy = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(imgcontour, contours, -1, (0,100,255),5)
cv.imshow('contour',imgcontour)
for cnt in contours:
  peri = cv.arcLength(cnt, True)
  approx = cv.approxPolyDP(cnt, 0.02*peri, True)
  x = approx.ravel()[0]
  y = approx.ravel()[1]
  print(x)
  print(y)
  if len(approx) == 4:
    cv.drawContours(r_img, [approx], 0, (0,100,255), 5)
    x, y, w, h = cv.boundingRect(approx)
    aspectRatio = float(w)/h
    print(aspectRatio)
    if aspectRatio >= 0.95 and aspectRatio<= 1.05:
        cv.putText(r_img, "Square", (x//2 , y//2), cv.FONT_HERSHEY_COMPLEX, 0.5,(0,0,0))
        hsv = cv.cvtColor(r_img, cv.COLOR_BGR2HSV)
        green_lower = np.array([25,52,72])   
        green_upper = np.array([102, 255, 255])
        mask = cv.inRange(hsv, green_lower, green_upper)
        kernel = np.ones((5,5), "uint8")
        green_mask = cv.dilate(mask, kernel)
        res_green = cv.bitwise_and(r_img, r_img, mask= green_mask)
        contours , hierarchy = cv.findContours(green_mask , cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for pic,cnt in enumerate(contours):
          area = cv.contourArea(cnt)
          if area>300:
            x,y,w,h = cv.boundingRect(cnt)
            imgFrame = cv.rectangle(r_img, (x,y), (x+w , y+h), (0,255,0), 5)
            dst = cv.addWeighted(r_img,0.3, r_ar2, 0.8,0)
        cv.imshow("blended", dst)
        


cv.imshow('r_img', r_img)





cv.waitKey(0)

