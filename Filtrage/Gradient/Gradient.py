import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

def convolution2d(image_canal,noyau):
    """
        Applique une convolution 2D entre une image de type float32 et un noyau (kernel).
        
        :image: Image en entrée
        :noyau: Filtre 2D
        retourne le canal de l'image convoluée
    """
    #Definition des différents paramètres 
    filtre_height,filtre_width=noyau.shape
    image_height,image_width=image_canal.shape
    padding_height=filtre_height//2
    padding_width=filtre_width//2
    #On crée un padding constant
    image_padded = np.pad(image_canal, ((padding_height, padding_width), (padding_height, padding_width)), mode='constant')
    # Image de sortie
    output = np.zeros_like(image_canal)

    # Convolution
    print(image_canal.shape)
    for i in range(image_height):
        for j in range(image_width):
                if(i%100==0 and j==image_width-1):
                    print("avancement du traitement: ",(i/image_height)*100,"%")
                # Multiplie le noyau par la région correspondante de l'image
                region = image_padded[i:i+filtre_height, j:j+filtre_width]
                output[i, j] = np.sum(region * noyau)
        
    return output



def Appliquer_Gradient(image):
    """
    Applique les gradients de Sobel sur une image (niveaux de gris ou couleur).
    
    :param image: Image d'entrée (en niveaux de gris ou couleur)
    :return: Gradients (Gx, Gy)
    """
    image_float = image.astype(np.float32)
    #Filtres de sobel en x et y
    sobelX = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    sobelY = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)
    #Gradients horizontal et vertical
    #Gx = np.zeros_like(image_float)
    #Gy = np.zeros_like(image_float)

    #Calcul des differentes convolution, en separant le cas gris du cas couleur
    #Gx = convolution2d(image_float, sobelX)
    #Gy = convolution2d(image_float, sobelY)

    # Calcul des gradients
    Gx = convolve(image_float, sobelX)
    Gy = convolve(image_float, sobelY)

    return Gx,Gy



def afficherGradient(image_name):
    #Acquisition de l'image
    image = cv2.imread(image_name)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Calcul des gradients et du module
    Gx,Gy=Appliquer_Gradient(image_gray)
    magnitude=np.sqrt(Gx**2 + Gy**2) 

    #Clipping 0-255 + passage à l'uint8
    magnitude=np.clip(magnitude, 0, 255).astype(np.uint8)
    Gx=np.clip(Gx, 0, 255).astype(np.uint8)
    Gy=np.clip(Gy, 0, 255).astype(np.uint8)

    # Affichage des résultats
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.title("Image originale")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Gradient Gx (horizontal)")
    plt.imshow(cv2.cvtColor(np.clip(Gx, 0, 255), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Gradient Gy (vertical)")
    plt.imshow(cv2.cvtColor(np.clip(Gy, 0, 255), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Magnitude du gradient")
    plt.imshow(magnitude, cmap='gray')
    plt.axis('off')
    
    plt.show()
    

afficherGradient("images_test/DSCN0836.JPG")