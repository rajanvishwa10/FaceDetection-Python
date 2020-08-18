import cv2 
import numpy as np
path = "multiple.jpg"
face_classifier = cv2.CascadeClassifier("haarcascade frontalface default.xml")
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face = face_classifier.detectMultiScale(gray, 1.3, 3)

for(x,y,w,h) in face:
    cv2.rectangle(image, (x,y), (x+w, y+h), (50,205,50), 2)

cv2.imshow("Face detection", image)
cv2.waitKey()
cv2.destroyAllWindows()
