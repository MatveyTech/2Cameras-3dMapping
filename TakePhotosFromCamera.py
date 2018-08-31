#%%s
import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)

counter = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if cv2.waitKey(1) & 0xFF == ord('t'):
        print ("Image saved")
        im = Image.fromarray(gray)
        im.save(str(counter)+".jpeg")
        counter = counter +1
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()