ra# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:24:04 2018

@author: matvey
"""

#%% Intrinsic calibration
import numpy as np
import cv2
import glob
import os
from FM import FindCommonFeatures

mainPath = ""
if os.path.isdir("C:/Users/matvey/"):
    mainPath = "C:/Users/matvey/Documents/CS2/CV Lab Project (2Cameras-3dMapping)/"

chess_w = 9
chess_h = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def GetObjectPoints():
    res = np.zeros((chess_h*chess_w,3), np.float32)
    res[:,:2] = np.mgrid[0:chess_w,0:chess_h].T.reshape(-1,2) 
    res = res * 27
    return res


def GetIntrinsicMatrix(pathToImages):
    print ("Calibrating from:\n"+pathToImages)
    
# termination criteria    
    objp = GetObjectPoints()
    
    objpoints = [] 
    imgpoints = [] 
    
    images = glob.glob(pathToImages+"*.jpeg")
   
    #print("\n\n\n\n\n\n")
    #print (pathToImages)
   # print (images)
    for fname in images:
        
        img = cv2.imread(fname)
#        if img is None:
#            print ('No such file {0}'.format(fname))
#            continue
#        else:
#            print (fname)
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        ret, corners = cv2.findChessboardCorners(gray, (chess_w,chess_h),None)    
        
        if ret == True:
            objpoints.append(objp)    
            corners_improved = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners_improved)
                       
#            img = cv2.drawChessboardCorners(img, (7,6), corners_improved,ret)
#            cv2.imshow('img',img)
#            cv2.waitKey(1000)
            #print ("Good")
        else:
            print ("Bad "+fname)
    reprojection_error, camera_matrix, distortion_coefficient, rotation_v,\
            translation_v = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    #print ("\n\n")
    #print (camera_matrix)
    cv2.destroyAllWindows()
    return camera_matrix,distortion_coefficient

def GetCameraPosition_chess(img, camera_int_mat,dist_coeff,showResult=False):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w,chess_h),None)
    ##ret = False
    if not ret:
        print("No chess corners found for this image")
        return False,None,None
    corners_improved = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    if showResult:
    
        xx= corners_improved[0][0][0]
        yy =corners_improved[0][0][1]
        img = cv2.drawChessboardCorners(img, (chess_w,chess_h), corners_improved,ret)
        cv2.circle(img, (xx,yy), 10, (255,0,255), -1)
        cv2.imshow('img',img)
        cv2.waitKey(30000)
        cv2.destroyAllWindows()

    rres = cv2.solvePnP(GetObjectPoints(),corners_improved,camera_int_mat,dist_coeff)
    return rres,corners_improved

def GetCamera3x4ProjMat(rvec, tvec,K):
    res = cv2.Rodrigues(rvec)[0]
    temp = np.hstack((res,tvec))
    return np.dot(K,temp)


def GetFirstChessImageMatches(img):
    #img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
    ret, corners = cv2.findChessboardCorners(gray, (chess_w,chess_h),None)    
    corners_improved = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)    
    return GetObjectPoints(), corners_improved      

    return kp1,kp2

def GetMatchedFeatures(img1, img2):
        
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
    matches = sorted(matches, key = lambda x:x.distance)
    l = []
    r = []
    #l.append(kp1[matches[0].trainIdx].pt)
    
    # Draw first 3 matches.
#    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:3], flags=2,outImg=None)
#    cv2.imshow('img3',img3)
#    cv2.waitKey()
    
    for i in range(len(matches)):
        l.append( kp1[matches[i]. trainIdx].pt)
        r.append( kp2[matches[i].queryIdx].pt)
    return np.transpose(r),np.transpose(l)

def Extract3DPoints(proj_pp,d3p):
    #print (proj_pp.tolist())
    #print (d3p)
    res = []
    for x in np.nditer(proj_pp.tolist()):
        key = "{0},{1}".format(x[0],x[1])
        #print ("Current key is {0}".format(key))
        res.append(d3p[key])
    return res

#def FindCommonFeatures(a,b):
#    r1 = [(1,2),(3,4),(5,6),(7,8)]
#    r2 = [(10,20),(30,40),(50,60),(70,80)]
#    return r1,r2

def TriangulatePoints():
    res = [(1,2,10),(3,4,20),(5,6,30),(7,8,40)]
    return res

def SanityCheck(wp,ip,K):
    
    
#%%
print("\n\n\n\n\n\n\n")

print("\n\n\n\n\n\n\n")
#print("start")
mainPath = ""
if os.path.isdir("C:/Users/matvey/"):
    mainPath = "C:/Users/matvey/Documents/CS2/CV Lab Project (2Cameras-3dMapping)/"


need_calib=False
if need_calib:
    int_calib_path1 = mainPath + "rep/Debug media/LeftCalib/"
    int_calib_path2 = mainPath + "rep/Debug media/RightCalib/"


    cam1_int_matrix, cam1_dist_coeff = (GetIntrinsicMatrix(int_calib_path1))
    cam2_int_matrix, cam2_dist_coeff = (GetIntrinsicMatrix(int_calib_path2))
    


firstFrameSuccessed = False

i01_c = cv2.imread(mainPath+"rep/Debug media/Pair from left/1.jpeg")
i01_g = cv2.cvtColor(i01_c,cv2.COLOR_BGR2GRAY)

i02_c = cv2.imread(mainPath+"rep/Debug media/0cam2.jpeg")
i02_g = cv2.cvtColor(i02_c,cv2.COLOR_BGR2GRAY)

#i01 = cv2.imread(int_calib_path1+"left04.jpg")
#i02 = cv2.imread(int_calib_path1+"left05.jpg")

i11 = cv2.imread(mainPath+"rep/Debug media/1cam1.jpeg")
i12 = cv2.imread(mainPath+"rep/Debug media/1cam2.jpeg")

projected_1,projected_2 = GetMatchedFeatures(i01_g,i02_g)

# Just for now :
d3p =	{
  "1,2": [3,4,5],
  "10,20": [30,40,50],
  "100,200": [300,400,500]
}
proj_pp = np.array([[1,2],[10,20],[100,200]]).T
#proj_pp=[1,2]

D3Points = Extract3DPoints(proj_pp,d3p)
print (D3Points)
#%%
if not firstFrameSuccessed:
    retval1, rvec, tvec = GetCameraPosition_chess(i01_c,cam1_int_matrix,cam1_dist_coeff)
    cam1_pm = GetCamera3x4ProjMat(rvec,tvec)

    retval2, rvec, tvec = GetCameraPosition_chess(i02_c,cam2_int_matrix,cam2_dist_coeff)
    cam2_pm = GetCamera3x4ProjMat(rvec,tvec)

#cv2.imshow('img1',np.hstack((i01_g,i01_g)))
#cv2.waitKey()
#cv2.destroyAllWindows()

X = cv2.triangulatePoints(cam1_pm,cam2_pm,projected_1,projected_2)
#cv2.convertPointsFromHomogeneous(
#np.transpose(np.array([[1,2],[3,4],[5,6]]))

#x,y = GetFirstChessImageMatches(cv2.imread(path+"left04.jpg"))

#FindChessMatches(path+"left04.jpg")
#print res

#%%
ppath = mainPath + "Intrinsic calibration files/Left/"
print (ppath)
GetIntrinsicMatrix(ppath)

#%% intrinsic calibration
int_calib_path1 = mainPath + "rep/Debug media/LeftCalibGood/"
int_calib_path2 = mainPath + "rep/Debug media/RightCalibGood/"


cam1_int_matrix, cam1_dist_coeff = (GetIntrinsicMatrix(int_calib_path1))
cam2_int_matrix, cam2_dist_coeff = (GetIntrinsicMatrix(int_calib_path2))

#%% main loop


import numpy as np
import cv2
i=0
cap1 = cv2.VideoCapture(mainPath + "rep/Debug media/debug_video8.avi")
cap2 = cv2.VideoCapture(mainPath + "rep/Debug media/debug_video7.avi")
firstFrameDone=False
while(cap1.isOpened()):
    i=i+1
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break
    if i<20:
        continue
    
#     # I N J E C T I O N #############################
#    ##################################################
#    ##################################################
#    
#    int_calib_path = mainPath + "rep/Debug media/LeftCalibGood/"
#    cam1_int_matrix, cam1_dist_coeff = (GetIntrinsicMatrix(int_calib_path))
#    cam2_int_matrix, cam2_dist_coeff = cam1_int_matrix, cam1_dist_coeff
#    print ("INT calibration done")
#    path1 = mainPath + "rep/Debug media/cameraLocationDebug-left/1.jpeg"
#    frame1 = cv2.imread(path1)
#    q
#    path4 = mainPath + "rep/Debug media/cameraLocationDebug-left/2.jpeg"
#    frame2 = cv2.imread(path4)    
#    
#    ###################################################
#    ###################################################
#    ###################################################
    
    
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    if not firstFrameDone:
        print ("firstFrameDone")
        
        m1,corners1 = GetCameraPosition_chess(frame1,cam1_int_matrix,cam1_dist_coeff,True)
        print(frame1.shape)
        retval1, rvec1, tvec1 = m1
        cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
        print (cam1_pm)
        
        m2,corners2 = GetCameraPosition_chess(frame2,cam2_int_matrix,cam2_dist_coeff,True)
        retval2, rvec2, tvec2 = m2
        cam2_pm = GetCamera3x4ProjMat(rvec2,tvec2,cam2_int_matrix)
        print (cam2_pm)
        im1_f,im2_f = FindCommonFeatures(frame1,frame2)
        
        #p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,im1_f,im2_f)
        p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,corners1.reshape(54,2).T,corners2.reshape(54,2).T)
        p3d_orig = p3d
        p3d/= p3d[3]
        p3d=p3d[0:3]
        p3d = p3d.T
        #print (p3d)
        np.save("testout77", p3d)
        #sss = np.load("testout.npy")
        #p3d = TriangulatePoints()
        
        break
        firstFrameDone=True
    im1_f,im2_f = FindCommonFeatures(frame1,frame2,i) 
    print(str(i)+" Num of features "+str(len(im1_f)))
        
#    cv2.imshow('frame',gray1)
#    if cv2.waitKey(1000) & 0xFF == ord('q'):
#        break
#    if i==20:
#        cv2.imwrite(mainPath + "rep/Debug media/video1_frame1.jpeg", gray1)
#        cv2.imwrite(mainPath + "rep/Debug media/video1_frame2.jpeg", gray2)
    
cap1.release()
cap2.release()
cv2.destroyAllWindows()
print(i)


















