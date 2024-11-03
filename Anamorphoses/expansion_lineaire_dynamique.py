import cv2
import matplotlib.pyplot as plt
import numpy as np
def generate_expansion_lineaire_dynamique(image,borne_inf,borne_sup ):
    """
         Applique l'expansion linéaire dynamique à une image pour améliorer le contraste.
    """
    #Passage aux float pour les caluls
    image_float = image.astype(np.float32)
    v_min, v_max = np.min(image), np.max(image)
    if(len(image.shape)==2): #Image en niveaux de gris
        expansion = np.where(image_float < borne_inf, 0, np.where(image_float > borne_sup, 255, (image_float - v_min) * 255 / (v_max - v_min)))
    else: #Image en couleur
        expansion=np.zeros_like(image_float)
        for i in range(3):  
            expansion[:, :, i] = np.where(image_float[:, :, i] < borne_inf, 0, np.where(image_float[:, :, i] > borne_sup, 255, (image_float[:, :, i] - v_min) * 255 / (v_max - v_min)))
    expansion = np.clip(expansion, 0, 255).astype(np.uint8) #clipping et retour au unisgned
    return expansion

def afficherExpansion(nom_image,borne_inf,borne_sup):
    image = cv2.imread(nom_image)
    expansion=generate_expansion_lineaire_dynamique(image,borne_inf,borne_sup)
    if(len(image.shape)==2):
        cmap="gray"
    else:
        cmap=None
    # Afficher les images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Image Originale')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(expansion, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Image avec expansion linéaire dynamique')

    plt.show()

afficherExpansion("images_test/DSCN2604.JPG",0,100)