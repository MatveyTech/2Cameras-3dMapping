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
import math
from numpy.linalg import inv
from numpy import linalg as LA

W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red

mainPath = ""
if os.path.isdir("C:/Users/matvey/"):
    mainPath = "C:/Users/matvey/Documents/CS2/CV Lab Project (2Cameras-3dMapping)/"
else:
    mainPath = "C:/matvery/2Cameras-3dMapping/"

leftVideoPath = "rep/Debug media/v6_left.avi"
rightVideoPath = "rep/Debug media/v6_right.avi"

leftIntrinsicCalibFolder =  "rep/Debug media/LeftCalibGood/"
rightIntrinsicCalibFolder = "rep/Debug media/RightCalibGood/"

chess_w = 9
chess_h = 6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#  input:  path to a folder with calibration pictures from a camera
#  return:  camera_matrix, distortion_coefficient

def FindCommonFeatures(img1, img2, img_num=0):
    visualFeedback = False
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # Create flann matcher
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #matcher = cv2.FlannBasedMatcher_create()
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kpts1, descs1 = sift.detectAndCompute(gray1, None)
    #print(descs1.shape)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kpts2, descs2 = sift.detectAndCompute(gray2, None)

    # Ratio test
    res1 = []
    res2 = []
    descriptors2return1 =[]
    descriptors2return2 =[]
    # Dictionarys to store points that already been seen
    dict1 = {}
    dict2 = {}
    matches = matcher.knnMatch(descs1, descs2, 2)
    #print (len(matches))
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.3 * m2.distance:
            matchesMask[i] = [1, 0]
            # Notice: How to get the index
            pt1 = kpts1[m1.queryIdx].pt
            pt2 = kpts2[m1.trainIdx].pt
            # visualFeedback:
            # print(i, pt1,pt2 )
            if(dict1.get(pt1, None) != 1 and dict2.get(pt2, None) != 1):
                dict1[pt1] = 1
                dict2[pt2] = 1

                res1.append(pt1)
                res2.append(pt2)
                
                descriptors2return1.append(descs1[m1.queryIdx])
#                print("ADDED")
#                print(descs1[m1.queryIdx].shape)
#                print(len(descriptors2return1))
                descriptors2return2.append(descs2[m1.trainIdx])
                # Draw pairs in purple, to make sure the result is ok
                if visualFeedback:
                    print(i, pt1, pt2)
                    cv2.circle(img1, (int(pt1[0]), int(
                        pt1[1])), 5, (255, 0, 255), -1)
                    cv2.circle(img2, (int(pt2[0]), int(
                        pt2[1])), 5, (255, 0, 255), -1)
            else:
                if visualFeedback:
                    if(dict1.get(pt1, None) == 1):
                        print(
                            "duplicate:(" + str(pt1[0]) + "," + str(pt1[1]) + ")")
                    if(dict2.get(pt2, None) == 1):
                        print(
                            "duplicate:(" + str(pt2[0]) + "," + str(pt2[1]) + ")")

    # Draw match in blue, error in red

    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=0)

    res = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2,
                             matches, None, **draw_params)

    if visualFeedback:
        fname = "sift/"+str(img_num)+".jpeg"
        #print("writing to "+fname)
        cv2.imwrite(fname, res)
        cv2.imshow("Result", res)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return np.asarray(res1), np.asarray(res2), np.asarray(descriptors2return1), np.asarray(descriptors2return2)

def GetObjectPoints():
    """    
    Returns 3d points in chess coordinate system 
    """
    res = np.zeros((chess_h*chess_w,3), np.float32)
    res[:,:2] = np.mgrid[0:chess_w,0:chess_h].T.reshape(-1,2) 
    res = res * 27
    return res


def GetIntrinsicMatrix(pathToImages):
    """    
    Gets path to calibration images(chess)
    Returns cameras intrinsic matrix and the distortion coefficient
    """
    print ("Calibrating from:\n"+pathToImages)
    
    # termination criteria    
    objp = GetObjectPoints()
    
    objpoints = [] 
    imgpoints = [] 
    
    # setting path to images
    images = glob.glob(pathToImages+"*.jpeg")   
    
    for fname in images:
        
        img = cv2.imread(fname)       
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
        ret, corners = cv2.findChessboardCorners(gray, (chess_w,chess_h),None)    
        
        # True means chessboard corenrs were found
        if ret == True:
            objpoints.append(objp)    
            corners_improved = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners_improved)
        # False means chessboard corenrs were not found              
        else:
            print ("Bad "+fname)
    reprojection_error, camera_matrix, distortion_coefficient, rotation_v,\
            translation_v = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
  
    cv2.destroyAllWindows()
    return camera_matrix,distortion_coefficient

def FindCorners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w,chess_h),None)
    if not ret:
        return False,None
    corn_imp = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    return True, corn_imp

def GetCameraPosition_chess(img, camera_int_mat,dist_coeff,showResult=False):
    """    
    returns solvePNN output on chess corners and the corners
    """
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w,chess_h),None)
    ##ret = False
    if not ret:
        print("No chess corners found for this image")
        return ((False,None,None),None)
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
    """    
    Retrieves 3x4 projection from data SolvePNP returns
    """
    res = cv2.Rodrigues(rvec)[0]
    temp = np.hstack((res,tvec))
    return np.dot(K,temp)

def GetCamera3x4ProjMatNoK(rvec, tvec):
    """    
    Retrieves 3x4 projection from data SolvePNP returns
    """
    res = cv2.Rodrigues(rvec)[0]
    temp = np.hstack((res,tvec))
    return temp


def CalculateDistance(p1,p2): 
    """    
    Calculates distance between 2 2d points
    """
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)  
    return dist  


def SanityCheck(wp,ip,K,dist_coeff,threshold=70):
    """    
    Checks if reconstructed 2d points are good.
    in case of distance (2d) more that threshold prints Warning to output
    """
    retval, rvec, tvec = cv2.solvePnP(wp,ip,K,dist_coeff)
    cam_pm = GetCamera3x4ProjMat(rvec,tvec,K)
    hg_wp = np.vstack((wp.T,np.ones(wp.shape[0]))).T
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
    if(summ>threshold):
        print(R+"W A R N I N G. The sanity check value is:%d. number of points is:%d" % (summ,ip.shape[0]))
        print(W)
    #return res,ip
    return summ
 

def Match2Dand3D(frame,p_descriptors_till_now,p_3d_till_now):
    """    
    This function returns all common descriptors between current frame and @p_descriptors_till_now param
    and its 3d value
    """
    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kpts, descs = sift.detectAndCompute(gray, None)
    
    _common_descriptors=[]
    _common_2d=[]
    _common_descriptors_3d=[]
    
    matches = matcher.knnMatch(descs,p_descriptors_till_now,2)
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.3 * m2.distance: 
            _common_descriptors.append(descs[m1.queryIdx])
            _common_descriptors_3d.append(p_3d_till_now[m1.trainIdx])
            _common_2d.append(kpts[m1.queryIdx].pt)   
   
    return np.asarray(_common_2d),np.asarray(_common_descriptors),np.asarray(_common_descriptors_3d)

def Get3DFrom4D(p4d):
    """    
    Returns 3d coordinate from the homogenius coord
    """
    p4d/= p4d[3]
    p4d=p4d[0:3]
    p4d = p4d.T  
    return p4d 

def FilterPoints1(points,desc):
    """    
    Filter points by mean and std - all axes
    """
    times_st = 2
    dict1 = {}
    for i in range(0, 3):
        vec = points[:,i]  
        #print(z_vec)
        hz = np.percentile(vec, 50)     
        st_d = np.std(vec)        
        dict1[i]= np.where((vec < hz-times_st*st_d) | (vec > hz+times_st*st_d))
    all_ind = np.union1d(np.union1d(dict1[0][0], dict1[1][0]),dict1[2][0])
    #print (all_ind)
    new_p = np.delete(points, all_ind,axis=0)
    new_desc = np.delete(desc, all_ind,axis=0)
    return new_p,new_desc

def FilterPoints2(points,desc):
    """    
    Filter points by threshold - z only
    """
    threshold = 500
    dict1 = {}
    for i in range(0, 3):
        vec = points[:,i] 
        if i==2:#z axis
            dict1[i]= np.where((vec < -threshold) | (vec > 0))
        else:
            dict1[i]= np.where((vec < -threshold) | (vec > threshold))
    all_ind = np.union1d(np.union1d(dict1[0][0], dict1[1][0]),dict1[2][0])
    #print (all_ind)
    new_p = np.delete(points, all_ind,axis=0)
    new_desc = np.delete(desc, all_ind,axis=0)
    return new_p,new_desc

def FilterPoints(points,desc):
    """    
    Filter points by axes or z only - depends on filterMode
    """
    filterMode=2
    if filterMode==1:
        return FilterPoints1(points,desc)
    if filterMode==2:
        return FilterPoints2(points,desc)
    

def GetCamera4x4ProjMat(rvec, tvec):
    """    
    Retrieves 4x4 projection matrix in homogenius coordinates
    """
    res = cv2.Rodrigues(rvec)[0]
    temp = np.hstack((res,tvec))
    return np.vstack((temp,np.asarray([0,0,0,1])))
#%%

"""    
Calibrate cameras (intrinsic calibration)
"""

int_calib_path1 = mainPath + leftIntrinsicCalibFolder
int_calib_path2 = mainPath + rightIntrinsicCalibFolder

cam1_int_matrix, cam1_dist_coeff = (GetIntrinsicMatrix(int_calib_path1))
cam2_int_matrix, cam2_dist_coeff = (GetIntrinsicMatrix(int_calib_path2))


#%% main loop

import numpy as np
import cv2
i=0

cap1 = cv2.VideoCapture(mainPath + leftVideoPath)
cap2 = cv2.VideoCapture(mainPath + rightVideoPath)


firstFrameDone=False
p3d=None
all_p3d=None
all_desc=None
frame1=None
frame2=None
im1_f=None
im2_f=None
camera1_to_camera2=None

#0-Chess only; 1-Sift only; 2-Prefer chess
CameraPositioningMode=0

while(cap1.isOpened()):
    i=i+1
    #print (i)
    
    prev_p3d=p3d
    prev_frame1 = frame1
    prev_frame2 = frame2   
    prev_im1_f = im1_f
    prev_im2_f = im2_f
    
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print ("not ret1 or not ret2")
        print ("Done")
        break
    
#    
    # first frame handling
    if not firstFrameDone:        
        print("Working on frame %d"%(i))
        m1,corners1 = GetCameraPosition_chess(frame1,cam1_int_matrix,cam1_dist_coeff,False)
        retval1, rvec1, tvec1 = m1
        cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
        rot1 = cv2.Rodrigues(rvec1)[0]
#        temp1 = np.hstack((rot1,tvec1))
#        chess2cam1 = np.vstack((temp1,[0,0,0,1]))
#        cam1_2chess = np.linalg.inv(chess2cam1)
#        print(np.dot(rot1,tvec1))
#        print(np.dot(rot1.T,tvec1))
        
        SanityCheck(GetObjectPoints(),corners1,cam1_int_matrix,cam1_dist_coeff)
        
        m2,corners2 = GetCameraPosition_chess(frame2,cam2_int_matrix,cam2_dist_coeff,False)
        retval2, rvec2, tvec2 = m2
        cam2_pm = GetCamera3x4ProjMat(rvec2,tvec2,cam2_int_matrix)
        rot2 = cv2.Rodrigues(rvec2)[0]
        #print(np.dot(rot2,tvec2)) 
        #print("____________________________________") 
        SanityCheck(GetObjectPoints(),corners2,cam2_int_matrix,cam2_dist_coeff)
        
        c1 = GetCamera4x4ProjMat(rvec1,tvec1)
        c2 = GetCamera4x4ProjMat(rvec2,tvec2)
        camera1_to_camera2 = np.dot(c2,inv(c1))
        im1_f,im2_f,im1_desc,im2_desc = FindCommonFeatures(frame1,frame2)
        # input:  cam1_pm : cam 1 projectionMatrix, cam2_pm : cam 2 projecetionMatrix ,im1_f: frame 1 features ,im2_f: frame 2 features
        # triangulatePoints	Output array with computed 3d points. Is 3 x N.
        p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,im1_f.T,im2_f.T)
        #p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,corners1.reshape(54,2).T,corners2.reshape(54,2).T)
        p3d_orig = p3d
        p3d=Get3DFrom4D(p3d)
        all_p3d = p3d
        all_desc = im1_desc
        #print (p3d.shape)
        #np.save("testout77", p3d)       
        firstFrameDone=True
    else: 
#uncomment this if you want to run the mainloop on first i frames
#        if i>10:
#            break

        
        print("Working on frame %d"%(i))
        t_im1_f,t_im2_f,t_im1_desc,t_im2_desc = FindCommonFeatures(frame1,frame2)
        #print("Features found: Left:%d, Right:%d."%(t_im1_f.shape[0],t_im2_f.shape[0]))
        if t_im1_f.shape[0] < 5 or t_im2_f.shape[0] < 5:
            print ("Not enough data. Image is skipped")
            continue
#        if t_im1_f.shape[0] < 5 or t_im2_f.shape[0] < 5:
#            print ("Not enough data. Image is skipped")
#            continue
        
        im1_f,im2_f,im1_desc,im2_desc= t_im1_f,t_im2_f,t_im1_desc,t_im2_desc
        
        common_2d_l,common_desc_l,common_3d_l = Match2Dand3D(frame1,all_desc,all_p3d)                  
        
        
        if CameraPositioningMode==0:
            #print ("Chess only")
            m1,corners1 = GetCameraPosition_chess(frame1,cam1_int_matrix,cam1_dist_coeff,False)
            retval1, rvec1, tvec1 = m1
            if retval1 == True:
                cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
            else:
                print ("You asked to calc the camera position from the chess but there is no chess detected.Continuing to the next frame.")
                continue
        elif CameraPositioningMode==1:
            #print ("Sift only")
            retval1, rvec1, tvec1 = cv2.solvePnP(common_3d_l,common_2d_l,cam1_int_matrix, cam1_dist_coeff)
            cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
        elif CameraPositioningMode==2:
            #print ("Chess prefered")
            m1,corners1 = GetCameraPosition_chess(frame1,cam1_int_matrix,cam1_dist_coeff,False)
            retval1, rvec1, tvec1 = m1
            if retval1 == True:
                cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
            else:
                #print ("You asked to calc the camera position from the chess but there is no chess detected.Calc from sift.")
                retval1, rvec1, tvec1 = cv2.solvePnP(common_3d_l,common_2d_l,cam1_int_matrix, cam1_dist_coeff)
                cam1_pm = GetCamera3x4ProjMat(rvec1,tvec1,cam1_int_matrix)
                
               
        #calculate the 2nd camera position from the relation to the 1st
        w_to_c1 = GetCamera4x4ProjMat(rvec1,tvec1)
        w_to_c2 = np.dot(camera1_to_camera2,w_to_c1)
        cam2_pm = np.dot(cam2_int_matrix,w_to_c2[0:3])
                    
        
        r1,corners1 = FindCorners(frame1)
        r2,corners2 = FindCorners(frame2)
        
        p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,im1_f.T,im2_f.T)       
        #p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,corners1.reshape(54,2).T,corners2.reshape(54,2).T)
        p3d_orig = p3d
        p3d=Get3DFrom4D(p3d)
        
        sc_left = SanityCheck(p3d,im1_f,cam1_int_matrix,cam1_dist_coeff)        
        sc_right = SanityCheck(p3d,im2_f,cam2_int_matrix,cam2_dist_coeff)
        
        #checking if sanity was good enough
        if sc_left>100 or sc_right > 100:
            print ("Bad frame!")
            continue
        
        current3dPoints = p3d
        currentDescriptors = im1_desc
        
        current3dPoints , currentDescriptors = FilterPoints(p3d,im1_desc)
        current3dPoints , currentDescriptors = FilterPoints(current3dPoints,currentDescriptors)
        current3dPoints , currentDescriptors = FilterPoints(current3dPoints,currentDescriptors)
        current3dPoints , currentDescriptors = FilterPoints(current3dPoints,currentDescriptors)
        
        print("%d points filtered"%(p3d.shape[0]-current3dPoints.shape[0]))
        #output to file
        np.save("testout"+str(i), current3dPoints)
        
        all_p3d = np.vstack((all_p3d,current3dPoints))
        all_desc = np.vstack((all_desc,currentDescriptors))
                
np.save("testout77", all_p3d)    
cap1.release()
cap2.release()
cv2.destroyAllWindows()

    


















