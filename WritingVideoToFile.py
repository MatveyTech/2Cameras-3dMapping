#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:49:28 2018

@author: matvey
"""

import cv2
import numpy as np
 
# Create a VideoCapture object
cap = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(3)
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
if (cap2.isOpened() == False): 
  print("Unable to read camera feed")
  
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_width2 = int(cap2.get(3))
frame_height2 = int(cap2.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('debug_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
out2 = cv2.VideoWriter('debug_video2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width2,frame_height2))
  
while(True):
  ret, frame = cap.read()
  ret2, frame2 = cap2.read()
 
  if ret == True and ret2 == True: 
     
    # Write the frame into the file 'output.avi'
    out.write(frame)
    out2.write(frame2)
 
    # Display the resulting frame    
    cv2.imshow('frame',frame)
    cv2.imshow('frame2',frame2)
 
    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
cap2.release()

out.release()
out2.release()
  
# Closes all the frames
cv2.destroyAllWindows() 