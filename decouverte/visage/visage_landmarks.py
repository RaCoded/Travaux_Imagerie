import cv2
import numpy as np
import dlib

#Début de la capture vidéo en temps réel
capture=cv2.VideoCapture(0)
#Création d'un detecteur de face
detecteur=dlib.get_frontal_face_detector()
predicteur=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
        landmarks=predicteur(gris, face)
        for n in range(0, 68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                if n==30 or n==36 or n==45:
                    cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
                else:
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    #affichage à l'écran
    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()