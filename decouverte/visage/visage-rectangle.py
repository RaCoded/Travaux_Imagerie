import cv2
import numpy as np
import dlib

#Début de la capture vidéo en temps réel
capture=cv2.VideoCapture(0)
#Création d'un detecteur de face
detecteur=dlib.get_frontal_face_detector()
#Tant qu'on filme:
while True:
    #Lecture de la frame
    ret, frame=capture.read()
    #Passage en gris nécessaire
    gris=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Récupération des faces
    faces=detecteur(gris)
    #Pour chaque face on récupére un rectangle
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    #affichage à l'écran
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()