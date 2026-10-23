import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
import matplotlib.pyplot as plt
from .utils import *
import pandas as pd 





def seuillage_double(image, seuil_min=70, seuil_max=330, valeur_objet=1, valeur_fond=0):

    filtre = sitk.BinaryThresholdImageFilter()
    filtre.SetLowerThreshold(seuil_min)
    filtre.SetUpperThreshold(seuil_max)
    filtre.SetInsideValue(valeur_objet)
    filtre.SetOutsideValue(valeur_fond)
    resultat = filtre.Execute(image)
    print(f"Double seuillage : [{seuil_min}, {seuil_max}]")
    return resultat


def create_image_label(image_binaire):
    if isinstance(image_binaire, sitk.Image):
        connect=sitk.ConnectedComponentImageFilter()
        connect.SetFullyConnected(True)
        image_label=connect.Execute(image_binaire)
        return image_label
    else:
        print("errzeur image n'est pas un objet de simpleITK")
        return None


