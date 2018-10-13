import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    if cv2.waitKey(1) & 0xFF == ord('t'):
        print ("Image saved")
        im = Image.fromarray(gray)
        im.save(str(counter)+"cam1.jpeg")
        im2 = Image.fromarray(gray2)
        im2.save(str(counter)+"cam2.jpeg")
        
        counter = counter +1
    # Display the resulting frame
    cv2.imshow('frame',gray)
    cv2.imshow('frame2',gray2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cap2.release()
cv2.destroyAllWindows()