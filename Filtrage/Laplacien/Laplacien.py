import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


def Appliquer_Laplacien(image):
    """
    Applique Un filtre laplacien afin d'obtenir le laplacien
    
    :param image: Image d'entrée (en niveaux de gris ou couleur)
    :return: Gradients (Gx, Gy)
    """
                
    #Filtres de sobel en x et y
    filtre_laplacien = np.array([[ 0, -1,  0],
                                 [-1,  4, -1],
                                 [ 0, -1,  0]], dtype=np.float32)


    # Calcul des gradients
    image_laplacien = convolve(image_float, filtre_laplacien)


    return image_laplacien



def afficherLaplacien(image_name):
    #Acquisition de l'image
    image = cv2.imread(image_name)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Calcul du laplacien
    image_laplacien=Appliquer_Laplacien(image_gray)

    
    # Prendre la valeur absolue
    image_laplacien_abs = np.abs(image_laplacien)
    #Clipping et passage à l'uint8
    image_laplacien=(np.clip(image_laplacien_abs, 0, 255)).astype(np.uint8)

    # Affichage des résultats
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image du laplacien")
    plt.imshow(cv2.cvtColor(image_laplacien, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
    

afficherLaplacien("images_test/cindy.JPG")