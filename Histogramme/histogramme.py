import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_histogramme(image):
    """Generation d'un histogramme pour une image donnee
       sous la forme [1,8,10,..] pour une image en niveaux de gris
       sous la forme [[1,8,10,..],[9,..],..] sinon    
    """
    if len(image.shape) == 2:
        # Image en niveaux de gris
        histogramme = cv2.calcHist([image], [0], None, [256], [0, 256])
    else:
        # Image en couleur bgr
        histogramme =[cv2.calcHist([image], [i], None, [256], [0, 256]) for i in range(3)]
    return histogramme



def afficher_histogram(histogramme,image):
    plt.figure(figsize=(12, 6))
    if(len(histogramme)==3):
        #format cv2:bgr
        colors = ('b', 'g', 'r')
        
        # Affichage de l'image en couleur
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image en couleur')

        # Affichage des histogrammes pour chaque canal
        plt.subplot(1, 2, 2)
        for i, color in enumerate(colors):
            plt.plot(histogramme[i], color=color)
            plt.xlim([0, 256])

        plt.title('Histogramme des canaux R, G, B')
        plt.xlabel('Intensité des pixels')
        plt.ylabel('Nombre de pixels')

        plt.show()
    else:

        # Affichage de l'image en niveaux de gris
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Image en niveaux de gris')

        # Affichage de l'histogramme
        plt.subplot(1, 2, 2)
        plt.plot(histogramme, color='black')
        plt.title('Histogramme des niveaux de gris')
        plt.xlabel('Intensité des pixels')
        plt.ylabel('Nombre de pixels')

        plt.show()



def histogramme_process(nom_image):
    """
        Capture l'image correspondant au nom, calcule son histogramme et retourne
        une visualisation de ce dernier
    """
    image = cv2.imread(nom_image)
    histogramme=generate_histogramme(image)
    afficher_histogram(histogramme,image)

histogramme_process("images_test/DSCN0125.JPG")