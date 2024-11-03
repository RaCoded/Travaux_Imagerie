import cv2
import matplotlib.pyplot as plt
import numpy as np

def generate_gamma_correction(image,gamma):
    """
    Applique une correction gamma à une image.
    """
    image_float = image.astype(np.float32)
    if(len(image.shape)==2):#Image en niveaux de gris
        image_gamma=255*(image_float/255)**gamma
    else: #Image en couleur
        image_gamma=np.zeros_like(image_float)
        for i in range(3):  
            image_gamma[:,:,i]=255*((image_float[:,:,i]/255)**gamma)
    image_gamma = np.clip(image_gamma, 0, 255).astype(np.uint8) #clipping et retour au unisgned
    return image_gamma


def afficherCorrectionGamma(nom_image,gamma):
    image = cv2.imread(nom_image)
    image_gamma=generate_gamma_correction(image,gamma)
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
    plt.imshow(cv2.cvtColor(image_gamma, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Image avec correction gamma')

    plt.show()

afficherCorrectionGamma("images_test/DSCN4447.JPG",0.8)