
# %% Intrinsic calibration
import numpy as np
import cv2
import glob
import os
from FM import FindCommonFeatures
import math

W = '\033[0m'  # white (normal)
R = '\033[31m'  # red

mainPath = ""
if os.path.isdir("C:/Users/matvey/"):
    mainPath = "C:/Users/matvey/Documents/CS2/CV Lab Project (2Cameras-3dMapping)/"
else:
    mainPath = "C:/matvery/2Cameras-3dMapping/"
chess_w = 9
chess_h = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def GetObjectPoints():
    res = np.zeros((chess_h*chess_w, 3), np.float32)
    res[:, :2] = np.mgrid[0:chess_w, 0:chess_h].T.reshape(-1, 2)
    res = res * 27
    return res

#  input:  path to a folder with calibration pictures from a camera
#  return:  camera_matrix, distortion_coefficient


def GetIntrinsicMatrix(pathToImages):
    print("Calibrating from folder: \n"+pathToImages)

    # termination criteria
    objp = GetObjectPoints()

    objpoints = []
    imgpoints = []
    # setting path to images
    images = glob.glob(pathToImages+"*.jpeg")
    # print("\n\n\n\n\n\n")
    #print (pathToImages)
    #print (images)
    for fname in images:
        img = cv2.imread(fname)
        # if img is None:
        #print ('No such file {0}'.format(fname))
        # continue
        # else:
        #print (fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Function that returns the corners of the ChessBoard on this image frame
        ret, corners = cv2.findChessboardCorners(
            gray, (chess_w, chess_h), None)

        # True means chessboard corenrs were found
        if ret == True:
            objpoints.append(objp)
            corners_improved = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_improved)
            #img = cv2.drawChessboardCorners(img, (7,6), corners_improved,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(1000)
            #print ("Good")

        # False means chessboard corenrs were not found
        else:
            print("Bad "+fname)
    reprojection_error, camera_matrix, distortion_coefficient, rotation_v,\
        translation_v = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
    #print ("\n\n")
    #print (camera_matrix)
    cv2.destroyAllWindows()
    return camera_matrix, distortion_coefficient


def GetCameraPosition_chess(img, camera_int_mat, dist_coeff, showResult=False):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
    ##ret = False
    if not ret:
        print("No chess corners found for this image")
        return False, None, None
    corners_improved = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria)
    if showResult:

        xx = corners_improved[0][0][0]
        yy = corners_improved[0][0][1]
        img = cv2.drawChessboardCorners(
            img, (chess_w, chess_h), corners_improved, ret)
        cv2.circle(img, (xx, yy), 10, (255, 0, 255), -1)
        cv2.imshow('img', img)
        cv2.waitKey(30000)
        cv2.destroyAllWindows()

    rres = cv2.solvePnP(GetObjectPoints(), corners_improved,
                        camera_int_mat, dist_coeff)
    return rres, corners_improved


def GetCamera3x4ProjMat(rvec, tvec, K):
    res = cv2.Rodrigues(rvec)[0]
    temp = np.hstack((res, tvec))
    return np.dot(K, temp)


def GetFirstChessImageMatches(img):
    #img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (chess_w, chess_h), None)
    corners_improved = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), criteria)
    return GetObjectPoints(), corners_improved

    return kp1, kp2


def GetMatchedFeatures(img1, img2):

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    l = []
    r = []
    # l.append(kp1[matches[0].trainIdx].pt)

    # Draw first 3 matches.
#    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:3], flags=2,outImg=None)
#    cv2.imshow('img3',img3)
#    cv2.waitKey()

    for i in range(len(matches)):
        l.append(kp1[matches[i]. trainIdx].pt)
        r.append(kp2[matches[i].queryIdx].pt)
    return np.transpose(r), np.transpose(l)


def Extract3DPoints(proj_pp, d3p):
    #print (proj_pp.tolist())
    #print (d3p)
    res = []
    for x in np.nditer(proj_pp.tolist()):
        key = "{0},{1}".format(x[0], x[1])
        #print ("Current key is {0}".format(key))
        res.append(d3p[key])
    return res

# def FindCommonFeatures(a,b):
#    r1 = [(1,2),(3,4),(5,6),(7,8)]
#    r2 = [(10,20),(30,40),(50,60),(70,80)]
#    return r1,r2


def TriangulatePoints():
    res = [(1, 2, 10), (3, 4, 20), (5, 6, 30), (7, 8, 40)]
    return res


def CalculateDistance(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist


def SanityCheck(wp, ip, K, dist_coeff):
    retval, rvec, tvec = cv2.solvePnP(wp, ip, K, dist_coeff)
    cam_pm = GetCamera3x4ProjMat(rvec, tvec, K)
    hg_wp = np.vstack((wp.T, np.ones(54))).T
    res = []
    for x in hg_wp:
        x2 = np.dot(cam_pm, x)
        x2 /= x2[2]
        res.append(x2[0:2])
    res = np.asarray(res)
    ip = ip.reshape(ip.shape[0], 2)
    summ = 0
    for x in range(0, ip.shape[0]):
        d = CalculateDistance(res[x], ip[x])
        # print(d)
        summ = summ + math.sqrt(d)
    if(summ > 66):
        print(
            R+"\n\n\n\n\n    W A R N I N G. The sanity check value is:%f\n\n\n\n\n" % summ)
        print(W)
    return res, ip


# %% intrinsic calibration
int_calib_path1 = mainPath + "rep/Debug media/LeftCalibGood/"
int_calib_path2 = mainPath + "rep/Debug media/RightCalibGood/"

# camera_matrix, distortion_coefficient
cam1_int_matrix, cam1_dist_coeff = (GetIntrinsicMatrix(int_calib_path1))
cam2_int_matrix, cam2_dist_coeff = (GetIntrinsicMatrix(int_calib_path2))

# %% main loop

# input: frame: first frame, prev_frame: previous frame, prev_im_f: previous frame features, pr_p3d:


def Match2Dand3D(frame, prev_frame, prev_im_f, pr_p3d):
    #    cv2.imshow('frame',frame)
    #    cv2.imshow('prev_frame',prev_frame)
    #    cv2.waitKey(10000)

    # find commonFeatures on both frames using Sift/Surf alforithem
    orig_f1, orig_f2 = FindCommonFeatures(frame, prev_frame)
#    np.save("orig_f1", orig_f1)
#    np.save("orig_f2", orig_f2)

    #orig_f1 = np.load("orig_f1.npy")
    #orig_f2 = np.load("orig_f2.npy")

#    f1=np.unique(orig_f1.T,axis=0)
#    f2=np.unique(orig_f2.T,axis=0)
    f1 = orig_f1.T
    f2 = orig_f2.T
    prev_im_f = prev_im_f.T
    li_curr = []
    li_prev = []
    li_3d = []
    for ii in range(0, f2.shape[0]):
        mindist = 99999
        j_idx = None
        for j in range(0, prev_im_f.shape[0]):
            dist = CalculateDistance(f2[ii], prev_im_f[j])
            if (dist < mindist):
                mindist = dist
                j_idx = j
        #print (mindist)
        if(mindist == 0):  # <1
            # print(ii,j_idx,f1[ii],pr_p3d[j_idx])
            #            if len(li_curr)==0:
            #                print(dist)
            #                print(mindist)
            #                print(j)
            #                print(j_idx, pr_p3d[j_idx])
            #                print(ii, f1[ii])
            li_curr.append(f1[ii])
            li_prev.append(f2[ii])
            li_3d.append(pr_p3d[j_idx])
    return np.asarray(li_curr), np.asarray(li_prev), np.asarray(li_3d), orig_f1, orig_f2


def MatchingSanityCheck(one, one_3d, two, two_3d):
    #print (one)
    #print (two)
    for i2 in range(0, one.shape[0]):
        for j2 in range(0, two.shape[0]):
            if one[i2][0] == two[j2][0] and one[i2][1] == two[j2][1]:
                # print(i2,j2)
                if one_3d[i2][0] != two_3d[j2][0] or one_3d[i2][1] != two_3d[j2][1] or one_3d[i2][2] != two_3d[j2][2]:
                    print(
                        "MatchingSanityCheck failed:one index%d, two index %d" % (i2, j2))


def Get3DFrom4D(p4d):
    p4d /= p4d[3]
    p4d = p4d[0:3]
    p4d = p4d.T
    return p4d


import numpy as np
import cv2
i = 0
cap1 = cv2.VideoCapture(mainPath + "rep/Debug media/debug_video8.avi")
cap2 = cv2.VideoCapture(mainPath + "rep/Debug media/debug_video7.avi")
firstFrameDone = False
p3d = None
frame1 = None
frame2 = None
im1_f = None
im2_f = None

while(cap1.isOpened()):

    i = i+1
    # updates for next iteration
    prev_p3d = p3d
    prev_frame1 = frame1
    prev_frame2 = frame2
    prev_im1_f = im1_f
    prev_im2_f = im2_f

    # read next frame from both videos
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("not ret1 or not ret2")
        break
    if i < 20:
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

    # first frame handaling
    if not firstFrameDone:

        #get corners of chessboard on first frame
        m1, corners1 = GetCameraPosition_chess(
            frame1, cam1_int_matrix, cam1_dist_coeff, False)
        retval1, rvec1, tvec1 = m1

        #get Camera projection Matrix
        cam1_pm = GetCamera3x4ProjMat(rvec1, tvec1, cam1_int_matrix)

        rot1 = cv2.Rodrigues(rvec1)[0]
#        temp1 = np.hstack((rot1,tvec1))
#        chess2cam1 = np.vstack((temp1,[0,0,0,1]))
#        cam1_2chess = np.linalg.inv(chess2cam1)
        # print(np.dot(rot1,tvec1))

        SanityCheck(GetObjectPoints(), corners1,
                    cam1_int_matrix, cam1_dist_coeff)

        m2, corners2 = GetCameraPosition_chess(
            frame2, cam2_int_matrix, cam2_dist_coeff, False)
        retval2, rvec2, tvec2 = m2
        cam2_pm = GetCamera3x4ProjMat(rvec2, tvec2, cam2_int_matrix)
        rot2 = cv2.Rodrigues(rvec2)[0]
        # print(np.dot(rot2,tvec2))
        # print("____________________________________")

        SanityCheck(GetObjectPoints(), corners2,
                    cam2_int_matrix, cam2_dist_coeff)

        im1_f, im2_f = FindCommonFeatures(frame1, frame2)

        # input:  cam1_pm : cam 1 projectionMatrix, cam2_pm : cam 2 projecetionMatrix ,im1_f: frame 1 features ,im2_f: frame 2 features
        # triangulatePoints	Output array with computed 3d points. Is 3 x N.
        p3d = cv2.triangulatePoints(cam1_pm, cam2_pm, im1_f, im2_f)
        #p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,corners1.reshape(54,2).T,corners2.reshape(54,2).T)
        p3d_orig = p3d
        p3d = Get3DFrom4D(p3d)

        #np.save("testout77", p3d)
        firstFrameDone = True
    else:

        l_curr, l_prev, l_3d, l_f1, l_f2 = Match2Dand3D(
            frame1, prev_frame1, prev_im1_f, prev_p3d)
        MatchingSanityCheck(l_prev, l_3d, prev_im1_f.T, prev_p3d)

        r_curr, r_prev, r_3d, r_f1, r_f2 = Match2Dand3D(
            frame2, prev_frame2, prev_im2_f, prev_p3d)
        MatchingSanityCheck(r_prev, r_3d, prev_im2_f.T, prev_p3d)

        retval1, rvec1, tvec1 = cv2.solvePnP(
            l_3d, l_curr, cam1_int_matrix, cam1_dist_coeff)
        cam1_pm = GetCamera3x4ProjMat(rvec1, tvec1, cam1_int_matrix)

        rot1 = cv2.Rodrigues(rvec1)[0]
        # print(np.dot(rot1,tvec1))

        retval2, rvec2, tvec2 = cv2.solvePnP(
            r_3d, r_curr, cam2_int_matrix, cam2_dist_coeff)
        cam2_pm = GetCamera3x4ProjMat(rvec2, tvec2, cam2_int_matrix)

        rot2 = cv2.Rodrigues(rvec2)[0]
       # print(np.dot(rot2,tvec2))

        im1_f, im2_f = FindCommonFeatures(frame1, frame2)

        p3d = cv2.triangulatePoints(cam1_pm, cam2_pm, im1_f, im2_f)
        #p3d = cv2.triangulatePoints(cam1_pm,cam2_pm,corners1.reshape(54,2).T,corners2.reshape(54,2).T)
        p3d_orig = p3d
        p3d = Get3DFrom4D(p3d)
        #print (p3d.shape)

        #output to file
        np.save("testout"+str(i), p3d)
        #break
        #l_common_f,l_common3d = Get3D(im1_f,l_curr,l_3d)
        #l_curr,l_3d = Match2Dand3D(prev_frame1,prev_frame1,prev_im1_f,prev_p3d)

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
# %%
