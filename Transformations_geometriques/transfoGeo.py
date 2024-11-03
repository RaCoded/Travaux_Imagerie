import cv2
import numpy as np
import matplotlib.pyplot as plt

def translation_image(image,tx,ty):
    """
    Applique une translation (tx,ty) à une image
    tx: translation sur l'axe horizontal
    ty: translation sur l'axe vertical
    retourne l'image
    """
    #Dimensions de l'image
    hauteur,largeur=image.shape[:2]
    #Création de l'image finale
    image_translated=np.zeros_like(image)
    #Parcours de l'image originale
    for i in range(hauteur):
        for j in range(largeur):
            #Nouvelles coordonées
            new_i=i+ty
            new_j=j+tx
            if 0<=new_i<hauteur and 0<=new_j<largeur:
                image_translated[new_i,new_j]=image[i,j]
    return image_translated

def changement_echelle_image(image,alphax,alphay):
    """
        Applique un changement d'echelle à une image
        alphax: facteur de changement sur l'axe horizontal
        alphay: facteur de changement sur l'axe vertical
    """
    # Vérification des dimensions de l'image
    if len(image.shape) == 2:  # Image en niveaux de gris
        hauteur, largeur = image.shape
        nb_canaux=1
    else:  # Image en couleur 
        hauteur, largeur, nb_canaux = image.shape

    nouvelle_hauteur=int(alphay*hauteur)
    nouvelle_largeur=int(alphax*largeur)

    # Création de l'image redimensionnée avec le bon nombre de canaux
    if nb_canaux==1:
        image_redimensionnee = np.zeros((nouvelle_hauteur, nouvelle_largeur), dtype=np.uint8)
    else:
        image_redimensionnee = np.zeros((nouvelle_hauteur, nouvelle_largeur, nb_canaux), dtype=np.uint8)


    # Parcours de chaque canal de l'image
    for c in range(nb_canaux):
        print("etape:",c)
        for i in range(nouvelle_hauteur):  # Parcours vertical
            # Coordonnées dans l'image d'origine
            old_i = i / alphay
            # Coordonnées voisines
            x0 = int(old_i)
            x1 = min(x0 + 1, hauteur - 1)
            # Distances
            alpha = old_i - x0
            a=1 - alpha
            for j in range(nouvelle_largeur):  # Parcours horizontal
                # Coordonnées dans l'image d'origine
                old_j = j / alphax

                # Coordonnées voisines
                y0 = int(old_j)
                y1 = min(y0 + 1, largeur - 1)

                # Valeurs des pixels voisins pour chaque canal
                if nb_canaux == 1:
                    I11 = image[x0, y0]
                    I21 = image[x0, y1]
                    I12 = image[x1, y0]
                    I22 = image[x1, y1]
                else:
                    I11 = image[x0, y0, c]
                    I21 = image[x0, y1, c]
                    I12 = image[x1, y0, c]
                    I22 = image[x1, y1, c]

                # Distances
                beta = old_j - y0
                b=1-beta
                # Interpolation bilinéaire
                valeur = (I11 * a * b+
                          I12 * a * beta +
                          I21 * alpha * b+
                          I22 * alpha * beta)
                if nb_canaux == 1:
                    image_redimensionnee[i, j] = valeur.astype(np.uint8) 
                else:
                    image_redimensionnee[i, j,c] = valeur.astype(np.uint8) 

    return image_redimensionnee

def rotation_image(image,theta):
    """
        Applique une rotation à l'image
        theta: angle d'inclinaison
    """
    # Vérification des dimensions de l'image
    if len(image.shape) == 2:  # Image en niveaux de gris
        hauteur, largeur = image.shape
        nb_canaux=1
    else:  # Image en couleur 
        hauteur, largeur, nb_canaux = image.shape

    # Calcul du centre de l'image
    centre_y, centre_x = largeur // 2, hauteur // 2
    # Création de l'image redimensionnée avec les mêmes dimensions que l'originale
    if nb_canaux == 1:
        image_rotate = np.zeros((hauteur, largeur), dtype=np.uint8)
    else:
        image_rotate = np.zeros((hauteur, largeur, nb_canaux), dtype=np.uint8)

    # Matrice de rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for c in range (nb_canaux):
        for i in range(hauteur):  # Parcours vertical
            if(i%100==0):
                print("avancement total:",i/(3*hauteur)*100)
            for j in range(largeur):  # Parcours horizontal
                # Coordonnées dans l'image d'origine
                old_j = (j-centre_y)*cos_theta - (i-centre_x)*sin_theta + centre_y
                old_i = (j-centre_y)*sin_theta +(i-centre_x)*cos_theta + centre_x
                # Coordonnées voisines
                x0 = int(old_i)
                y0 = int(old_j)
                if 0<=x0<hauteur and 0<=y0<largeur:

                    x1 = min(x0 + 1, hauteur - 1)
                    y1 = min(y0 + 1, largeur - 1)
                    # Distances
                    alpha = old_i - x0
                    beta = old_j - y0
                    a=1 - alpha
                    b=1-beta
                    # Valeurs des pixels voisins pour chaque canal
                    if nb_canaux == 1:
                        I11 = image[x0, y0]
                        I21 = image[x0, y1]
                        I12 = image[x1, y0]
                        I22 = image[x1, y1]
                    else:
                        I11 = image[x0, y0, c]
                        I21 = image[x0, y1, c]
                        I12 = image[x1, y0, c]
                        I22 = image[x1, y1, c]
                    # Interpolation bilinéaire
                    valeur = (I11 * a * b+
                                I12 * a * beta +
                                I21 * alpha * b+
                                I22 * alpha * beta)
                    if nb_canaux == 1:
                        image_rotate[i, j] = valeur.astype(np.uint8) 
                    else:
                        image_rotate[i, j,c] = valeur.astype(np.uint8) 

    return image_rotate




def afficher_images(originale, traduite):
    """
    Affiche l'image originale et l'image traduite côte à côte.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(originale, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image après modif ")
    plt.imshow(cv2.cvtColor(traduite, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

# Chargement de l'image
image = cv2.imread('images_test/DSCN2642.JPG')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_traduite = rotation_image(image_gray, 180)
print(image.shape,image_traduite.shape)
# Affichage des résultats
afficher_images(image, image_traduite)