import cv2
import matplotlib.pyplot as plt
import numpy as np

def reduction_bilineaire(image,pas):
    image_float = image.astype(np.float32)
    # Dimensions de l'image réduite
    new_height = image_float.shape[0] // pas
    new_width = image_float.shape[1] // pas
    output = np.zeros((new_height, new_width, image_float.shape[2]), dtype=np.float32) if len(image.shape) == 3 else np.zeros((new_height, new_width), dtype=np.float32)
    
    for i in range (image.shape[0]//pas):
        if(i%10==0):
            print("avancement,",i/new_height*100)
        for j in range (image.shape[1]//pas):
            # Coordonnées dans l'image originale
            y = i * pas
            x = j * pas
            # Calcul des valeurs bilinéaires
            y1 = min(y + pas, image.shape[0] - 1)
            x1 = min(x + pas, image.shape[1] - 1)
            # Moyenne des valeurs dans la zone de réduction
            region = image_float[y:y1, x:x1] if len(image.shape) == 2 else image_float[y:y1, x:x1, :]
            output[i, j] = np.mean(region, axis=(0, 1)) if len(image.shape) == 3 else np.mean(region)
    return output.astype(np.uint8)

def afficherReduction_bilineaire(image_name,pas):
    #Acquisition de l'image
    image = cv2.imread(image_name)
    reducted_image=reduction_bilineaire(image,pas)
    print(image.shape,reducted_image.shape)
     # Affichage des résultats
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image réduite")
    plt.imshow(cv2.cvtColor(reducted_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

#afficherReduction_bilineaire("images_test/cindy.JPG",2)

def interpolation_bilineaire(image,facteur):
    image_float = image.astype(np.float32)
     # Dimensions de l'image d'origine
    hauteur_orig, largeur_orig = image.shape
    # Dimensions de l'image agrandie
    hauteur_nouvelle = int(hauteur_orig * facteur)
    largeur_nouvelle = int(largeur_orig * facteur)

    # Création d'une image vide pour stocker l'image agrandie
    image_agrandie = np.zeros((hauteur_nouvelle, largeur_nouvelle), dtype=np.float32)

    for i in range (hauteur_nouvelle):
        if (i%10==0):
            print(i/hauteur_nouvelle*100,"%")
        for j in range (largeur_nouvelle):
            #coordonées dans l'image d'origine
            x_interpol=i/facteur# i pour la hauteur (x)
            y_interpol=j/facteur# j pour la largeur (y)

            #coordonées voisine
            x0=int(x_interpol)
            x1=min(x0+1,hauteur_orig-1)
            y0=int(y_interpol)
            y1=min(y0+1,largeur_orig-1)

            # Valeurs des pixels voisins
            I11 = image_float[x0, y0]
            I21 = image_float[x0, y1]
            I12 = image_float[x1, y0]
            I22 = image_float[x1, y1]

            #distances
            alpha=x_interpol-x0
            beta=y_interpol-y0

            valeur= I11*(1-alpha)*(1-beta)+ I12*(1-beta)*(alpha) +I21*(beta)*(1-alpha) +I22*alpha*beta

            image_agrandie[i,j]=valeur

    return np.uint8(image_agrandie)

def afficher_interpolation_bilineaire(image_name,facteur):
    #Acquisition de l'image
    image = cv2.imread(image_name)
    print(image.shape)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Calcul de l'interpollation
    image_interpolle=interpolation_bilineaire(image_gray,facteur)
    print(image_interpolle.shape)
    
    #Clipping et passage à l'uint8
    image_interpolle=(np.clip(image_interpolle, 0, 255)).astype(np.uint8)

    # Affichage des résultats
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Image de l'interpolation")
    plt.imshow(cv2.cvtColor(image_interpolle, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

afficher_interpolation_bilineaire("images_test/X.JPG",2)