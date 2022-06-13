import cv2 as cv
path1 = "cube.webp"
path2 = "image.webp"
img1 = cv.imread(path1)
img2 = cv.imread(path2)
# cv.imshow("a", img1)
# cv.imshow('b', img2)
r_img1 = cv.resize(img1, (500,400))
r_img2 = cv.resize(img2, (500,400))
# cv.imshow("a", r_img1)
# cv.imshow('b', r_img2)
blended = cv.addWeighted(r_img1, 0.4, r_img2, 0.1,0)
cv.imshow('blended', blended)
cv.waitKey(0)