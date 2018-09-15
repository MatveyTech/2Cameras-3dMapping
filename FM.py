def FindCommonFeatures(img1, img2,img_num=0): 
    visualFeedback=True
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    
    
    ## Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    
    ## Create flann matcher
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #matcher = cv2.FlannBasedMatcher_create()
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kpts1, descs1 = sift.detectAndCompute(gray1,None)
    
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kpts2, descs2 = sift.detectAndCompute(gray2,None)
    
    ## Ratio test
    res1=[]
    res2=[]
    counter = 0 
    matches = matcher.knnMatch(descs1, descs2, 2)
    matchesMask = [[0,0] for i in range(len(matches))]
    for i, (m1,m2) in enumerate(matches):
        if m1.distance < 0.35 * m2.distance:
            if counter < 40:
                counter = counter +1 
                matchesMask[i] = [1,0]
                ## Notice: How to get the index
                pt1 = kpts1[m1.queryIdx].pt
                pt2 = kpts2[m1.trainIdx].pt
                if visualFeedback:
                    print(i, pt1,pt2 )
                res1.append(pt1)
                res2.append(pt2)
                #if i % 5 ==0:
                ## Draw pairs in purple, to make sure the result is ok
                if visualFeedback:
                    cv2.circle(img1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
                    cv2.circle(img2, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)
    
    
    ## Draw match in blue, error in red
    
    draw_params = dict(matchColor = (255, 0,0),
                       singlePointColor = (0,0,255),
                       matchesMask = matchesMask,
                       flags = 0)
    
    res = cv2.drawMatchesKnn(img1,kpts1,img2,kpts2,matches,None,**draw_params)
    
    
    if visualFeedback:
        fname = "sift/"+str(img_num)+".jpeg"
        #print("writing to "+fname)
        cv2.imwrite(fname,res)
        cv2.imshow("Result", res)    
        cv2.waitKey()
        cv2.destroyAllWindows()
    return np.asarray(res1).T,np.asarray(res2).T