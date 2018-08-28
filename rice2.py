import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread('rice.jpg',cv.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
plt.title("rice")
plt.show()

#kernel = np.ones((5,5),np.float32)/25

ret,thresh =cv.threshold(img,90,255,cv.THRESH_BINARY)

plt.imshow(thresh,cmap="gray")
plt.title("Thresholded")
plt.show()



ret, labels = cv.connectedComponents(thresh)
#map componant lables to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch=255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue,blank_ch,blank_ch])

labeled_img = cv.cvtColor(labeled_img,cv.COLOR_HSV2BGR)

labeled_img[label_hue==0]=0

cv.imshow('labled.jpg',labeled_img)
cv.waitKey()

nLabels = cv.connectedComponents(thresh,labels=thresh,connectivity=8,ltype= cv.CV_32S)
print(nLabels)