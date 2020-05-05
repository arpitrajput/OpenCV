import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData/trainner.yml')
cascadePath = "Classifiers/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_TRIPLEX
fontscale = 3
fontcolor = (0, 255, 0)

cam = cv2.VideoCapture(0)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.5,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<80):
            if(Id==1):
                Id="Arpit"
            if(Id==2):
                Id="Anirudh"
        else:
            Id="Unknown"
        cv2.putText(im,str(Id), (x+w+20,y+h-180),font, fontscale,fontcolor)
    cv2.imshow('im',im) 
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()