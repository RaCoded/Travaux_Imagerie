import cv2
import matplotlib.pyplot as plt
def generate_negatif(image):
    """
        Generation de l'inverse video (negatif) d'une image
    """
    negatif=255-image
    return negatif
    
    
def afficher_negatif(nom_image):
    image = cv2.imread(nom_image)
    negatif=generate_negatif(image)

    #Vérifier si couleur ou gris
    if len(image.shape) == 2:  # Image en niveaux de gris
        cmap="gray"
    else:
        cmap=None

    # Afficher les images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Image Originale')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(negatif, cv2.COLOR_BGR2RGB), cmap)
    plt.title('Négatif')

    plt.show()

afficher_negatif("images_test/cindy.JPG")