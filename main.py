import cv2
import numpy as np
import math
import face_recognition
#Converting BGR To RGB
imgabhi = face_recognition.load_image_file('images/Train.jpg')
imgabhi = cv2.cvtColor(imgabhi,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('images/Abhitestn.jpg')
imgTest= cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
#Finding tthe faces in out imagr and there encodings
faceLOC=face_recognition.face_locations(imgabhi)[0] # gives four value right bottom and left
encodeAbhi=face_recognition.face_encodings(imgabhi)[0]
cv2.rectangle(imgabhi,(faceLOC[3],faceLOC[0]),(faceLOC[1],faceLOC[2]),(255,0,255),2)

faceLOC1=face_recognition.face_locations(imgTest)[0] # gives four value right bottom and left
encodeAbhitest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLOC1[3],faceLOC1[0]),(faceLOC1[1],faceLOC1[2]),(255,0,255),2)

#Comparing the faces basically the encodings which at the backend uses linear svm to match if they are similar or not
results =  face_recognition.compare_faces([encodeAbhi],encodeAbhitest)
#Sometimes there are a lot of images we need to find how much simialr they are
#TO do that we will find the distance
facedis=face_recognition.face_distance([encodeAbhi],encodeAbhitest)#Lower the diatnce better the match
print(results,facedis)
cv2.putText(imgTest,f'{results}{round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)#facedis[0] coz an array
cv2.imshow('Abhimanyu',imgabhi)
cv2.imshow('AbhimanyuTest',imgTest)
cv2.waitKey(0)

#13:09

