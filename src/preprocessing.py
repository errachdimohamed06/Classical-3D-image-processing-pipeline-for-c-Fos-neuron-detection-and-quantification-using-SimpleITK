import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
import matplotlib.pyplot as plt
from .utils import *
import pandas as pd 



def aplatir_image(image):
    if isinstance(image,sitk.Image):

        array=convert_image_to_array(image)
        return array.flatten()

def aplatir_array(aray):
    return array.flatten()


def normaliser_coupes_z(image):

    if isinstance(image,sitk.Image):

        array=convert_image_to_array(image)

    else:

        array=image
    # Calculer la moyenne de chaque coupe Z (axe 0)
    moyennes = array.mean(axis=(1, 2))
    mediane = np.median(moyennes)

    # Normaliser chaque coupe
    for i in range(array.shape[0]):
        if moyennes[i] > 0:
            facteur = mediane / moyennes[i]
            array[i, :, :] = array[i, :, :] * facteur

    resultat = sitk.GetImageFromArray(array)
    resultat.CopyInformation(image)

    print(f" Normalisation Z : médiane={mediane:.2f}, {array.shape[0]} coupes traitées")
    return resultat


def filtre_median_3d(image, rayon=[2, 2, 2]):
    """
    Applique un filtre médian 3D pour enlever le bruit
    Retourne:
    --------
    sitk.Image : Image filtrée
    """
    if isinstance(rayon, int):
        rayon = [rayon, rayon, rayon]

    filtre = sitk.MedianImageFilter()
    filtre.SetRadius(rayon)
    resultat = filtre.Execute(image)

    print(f" Filtre médian 3D : rayon={rayon}")
    return resultat



def top_hat_3d(image, rayon=[4, 4, 2]):
    """
    Applique un top-hat 3D (White Top-Hat) pour extraire les petits objets brillants
    
    """
    if isinstance(rayon, int):
        rayon = [rayon, rayon, rayon]
    
    tophat = sitk.WhiteTopHatImageFilter()
    tophat.SetKernelRadius(rayon)
    tophat.SetKernelType(sitk.sitkBall)  # forme sphérique
    resultat = tophat.Execute(image)
    
    print(f"Top-hat 3D : rayon={rayon}")
    return resultat





def dilate_binary_image(image_binaire,rayon_xy=3,rayon_z=1):
    if isinstance(image_binaire, sitk.Image):
        dilate=sitk.BinaryDilateImageFilter()
        dilate.SetKernelRadius([rayon_xy,rayon_xy,rayon_z])
        dilate.SetKernelType(sitk.sitkBall)
        dilate.SetForegroundValue(1)
        dilate.SetBackgroundValue(1)
        image_binaire_dilater=dilate.Execute(image_binaire)
        image_binaire_dilater.CopyInformation(image_binaire)
        print(f" dilatation : rayon XY={rayon_xy}, Z={rayon_z}")
        return image_binaire_dilater
    else:
        print("errzeur image n'est pas un objet de simpleITK")
        return None
    
    
