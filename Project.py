# -*- coding: utf-8 -*-
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
import math

W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red

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

def CalculateDistance(p1,p2):  
     dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  
     return dist  

def SanityCheck(wp,ip,K,dist_coeff):     
     retval, rvec, tvec = cv2.solvePnP(wp,ip,K,dist_coeff)
     cam_pm = GetCamera3x4ProjMat(rvec,tvec,K)
     hg_wp = np.vstack((wp.T,np.ones(54))).T
     res=[]
     for x in  hg_wp:
         x2 = np.dot(cam_pm,x)
         x2/=x2[2]
         res.append(x2[0:2])
     res = np.asarray(res)
     ip = ip.reshape(ip.shape[0],2)
     summ = 0
     for x in range(0, ip.shape[0]):
         d = CalculateDistance(res[x],ip[x])
         #print(d)
         summ = summ + math.sqrt(d)
     if(summ>66):
         print(R+"\n\n\n\n\n    W A R N I N G. The sanity check value is:%f\n\n\n\n\n" % summ)
         print(W)
     return res,ip

#%% intrinsic calibration
int_calib_path1 = mainPath + "rep/Debug media/LeftCalibGood/"
int_calib_path2 = mainPath + "rep/Debug media/RightCalibGood/"


cam1_int_matrix, cam1_dist_coeff = (GetIntrinsicMatrix(int_calib_path1))
cam2_int_matrix, cam2_dist_coeff = (GetIntrinsicMatrix(int_calib_path2))

#%% main loop
def Match2Dand3D(frame,prev_frame,prev_im_f,pr_p3d):
#    cv2.imshow('frame',frame)
#    cv2.imshow('prev_frame',prev_frame)
#    cv2.waitKey(10000)
    f1,f2 = FindCommonFeatures(frame,prev_frame)
    f1=np.unique(f1.T,axis=0)
    f2=np.unique(f2.T,axis=0)
    prev_im_f = prev_im_f.T
    li_curr=[]
    li_3d=[]
    for ii in range(0, f2.shape[0]):
        mindist=99999
        j_idx=None
        for j in range(0, prev_im_f.shape[0]):
            dist = CalculateDistance(f2[ii],prev_im_f[j])
            if (dist<mindist):
                mindist = dist
                j_idx=j
        #print (mindist)
        if(mindist<1):
            #print(ii,j_idx,f1[ii],pr_p3d[j_idx])
            li_curr.append(f1[ii])
            li_3d.append(pr_p3d[j_idx]) 
    return np.asarray(li_curr),np.asarray(li_3d)

import numpy as np
import cv2
i=0
cap1 = cv2.VideoCapture(mainPath + "rep/Debug media/debug_video8.avi")
cap2 = cv2.VideoCapture(mainPath + "rep/Debug media/debug_video7.avi")
firstFrameDone=False
p3d=None
frame1=None
frame2=None
im1_f=None
im2_f=None
while(cap1.isOpened()):
    i=i+1
    
    prev_p3d=p3d
    prev_frame1 = frame1
    prev_frame2 = frame2   
    prev_im1_f = im1_f
    prev_im2_f = im2_f
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print ("not ret1 or not ret2")
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
    
    if not firstFrameDone:
        
        m1,corners1 = GetCameraPosition_chess(frame1,cam1_int_matrix,cam1_dist_coeff,False)
        retval1, rvec1, tvec1 = m1
        cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
        SanityCheck(GetObjectPoints(),corners1,cam1_int_matrix,cam1_dist_coeff)
        
        m2,corners2 = GetCameraPosition_chess(frame2,cam2_int_matrix,cam2_dist_coeff,False)
        retval2, rvec2, tvec2 = m2
        cam2_pm = GetCamera3x4ProjMat(rvec2,tvec2,cam2_int_matrix)
        SanityCheck(GetObjectPoints(),corners2,cam2_int_matrix,cam2_dist_coeff)
        
        im1_f,im2_f = FindCommonFeatures(frame1,frame2)
        #print(im1_f.T[0:5])
        #print(im2_f.T[0:5])
        
        #l_curr,l_3d = Match2Dand3D(frame1,frame1,im1_f,prev_p3d)
       # print(im1_f.T[0:5])
        
        p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,im1_f,im2_f)
        #p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,corners1.reshape(54,2).T,corners2.reshape(54,2).T)
        p3d_orig = p3d
        p3d/= p3d[3]
        p3d=p3d[0:3]
        p3d = p3d.T
        
        np.save("testout77", p3d)       
        firstFrameDone=True
    else:
        #print(prev_im1_f.T[0:5])
        l_curr,l_3d = Match2Dand3D(prev_frame1,prev_frame1,prev_im1_f,prev_p3d)
        #l_curr,l_3d = Match2Dand3D(prev_frame1,prev_frame1,prev_im1_f,prev_p3d)
        break
        #im1_f,im2_f = FindCommonFeatures(frame1,frame2,i)
        
#    cv2.imshow('frame',gray1)
#    if cv2.waitKey(1000) & 0xFF == ord('q'):
#        break
#    if i==20:
#        cv2.imwrite(mainPath + "rep/Debug media/video1_frame1.jpeg", gray1)
#        cv2.imwrite(mainPath + "rep/Debug media/video1_frame2.jpeg", gray2)
    
cap1.release()
cap2.release()
cv2.destroyAllWindows()
#%%

    


















