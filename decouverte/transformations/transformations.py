import cv2
import numpy as np
import random
def erosion(image, kernel):
    
    # Dimensions de l'image et du kernel
    if len(image.shape) == 2:
        image_height, image_width = image.shape
        num_channels = 1
    else:
        image_height, image_width,_ = image.shape
        num_channels = 3
    kernel_height, kernel_width = kernel.shape

    # Taille des bords à ajouter autour de l'image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Créer une image de sortie
    if num_channels == 1:
        eroded_image = np.zeros((image_height, image_width), dtype=image.dtype)
    else:
        eroded_image = np.zeros((image_height, image_width, num_channels), dtype=image.dtype)


    for canal in range (num_channels):
        #canal actuel: 
        if num_channels == 1:
            # Pour une image en niveaux de gris
            current_canal = image
        else:
            # Pour une image en couleur
            current_canal = image[:, :, canal]
        # Ajouter des bords au canal pour faciliter l'érosion
        padded_image = cv2.copyMakeBorder(current_canal, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
        
        # Parcourir chaque pixel du canal
        for i in range(pad_height, image_height + pad_height):
            for j in range(pad_width, image_width + pad_width):
                # Extraire la région de l'image correspondant au kernel
                region = padded_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
                # Appliquer l'érosion
                if num_channels == 1:
                    eroded_image[i-pad_height, j-pad_width] = np.min(region)
                else:
                    eroded_image[i-pad_height, j-pad_width, canal] = np.min(region)
    return eroded_image

def dilatation(image, kernel):
    
    # Dimensions de l'image et du kernel
    if len(image.shape) == 2:
        image_height, image_width = image.shape
        num_channels = 1
    else:
        image_height, image_width,_ = image.shape
        num_channels = 3
    kernel_height, kernel_width = kernel.shape

    # Taille des bords à ajouter autour de l'image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Créer une image de sortie
    if num_channels == 1:
        eroded_image = np.zeros((image_height, image_width), dtype=image.dtype)
    else:
        eroded_image = np.zeros((image_height, image_width, num_channels), dtype=image.dtype)


    for canal in range (num_channels):
        #canal actuel: 
        if num_channels == 1:
            # Pour une image en niveaux de gris
            current_canal = image
        else:
            # Pour une image en couleur
            current_canal = image[:, :, canal]
        # Ajouter des bords au canal pour faciliter l'érosion
        padded_image = cv2.copyMakeBorder(current_canal, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
        
        # Parcourir chaque pixel du canal
        for i in range(pad_height, image_height + pad_height):
            for j in range(pad_width, image_width + pad_width):
                # Extraire la région de l'image correspondant au kernel
                region = padded_image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]
                # Appliquer l'érosion
                if num_channels == 1:
                    eroded_image[i-pad_height, j-pad_width] = np.max(region)
                else:
                    eroded_image[i-pad_height, j-pad_width, canal] = np.max(region)
    return eroded_image



def kmeans(image,k,max_iters):
    #Passage au rgb
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #On crée un tableau de pixel: chaque ligne = 1 pixel
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    #Initialisation aléatoire des indices des clusters, récupération des valeurs 
    random_indices = random.sample(range(pixels.shape[0]), k)
    centers=pixels[random_indices]

    for _ in range(max_iters):
    # (n,k,3)                (n,1,3)       (k,3)
        distances=pixels[:, np.newaxis] - centers
        distances = np.linalg.norm(distances, axis=2) #(n,k)
        labels=np.argmin(distances,axis=1) #(n)
        new_centers=np.array([pixels[labels==i].mean(axis=0) for i in range(k)])
        
        # Vérifier la convergence
        if np.all(np.linalg.norm(new_centers - centers, axis=1) < 1e-5):
            break
        centers = new_centers
    segmentation=centers[labels]
    segmentation=segmentation.reshape(image.shape)
    return segmentation

# Charger l'image
image = cv2.imread('cindy.jpg')

# Définir la taille du kernel pour l'érosion (par exemple, un carré de 5x5)
#kernel_size = 3
#kernel = np.ones((kernel_size, kernel_size), np.uint8)

segmentation=kmeans(image,3,100)
segmentation_colorbgr=cv2.cvtColor(segmentation.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imshow('Image Originale', image)
cv2.imshow('Image Segmentée', segmentation_colorbgr)
# Attendre une touche de l'utilisateur et fermer les fenêtres
cv2.waitKey(0)
cv2.destroyAllWindows()
