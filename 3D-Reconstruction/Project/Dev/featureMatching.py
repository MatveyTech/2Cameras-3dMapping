
# Example of Brute Force matching base on ORB Algorithm
  #Modify Author : Waheed Rafiq R&D student Birmingham City University UK
  #Original author : OpenCV.org
  #Date Updated : 21/04/2016 : 13:45 
import numpy as np
import cv2
from matplotlib import pyplot as plt 

mainPath = ""
if os.path.isdir("C:/Users/matvey/"):
    mainPath = "C:/Users/matvey/Documents/CS2/CV Lab Project (2Cameras-3dMapping)/"
  
img1 = cv2.imread('0cam1.jpeg',1)          # queryImage
img2 = cv2.imread('0cam2.jpeg',1)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance, reverse=True)
l = []
r = []
#l.append(kp1[matches[0].trainIdx].pt)

for i in range(len(matches)):
    l.append( kp1[matches[i].trainIdx].pt)
    r.append( kp2[matches[i].queryIdx].pt)
    

print(np.transpose(r).shape)
print(np.transpose(l).shape)
#print(l)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5] ,None, flags=2)
cv2.imwrite('sift_featureMatched.jpg',img3)
plt.imshow(img3),plt.show()
