
import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
import matplotlib.pyplot as plt
from .utils import *
import pandas as pd 

def combine_neurons(img_neurons, img_separated):
    
    array_neurons = sitk.GetArrayFromImage(img_neurons)
    array_separated = sitk.GetArrayFromImage(img_separated)
    
    
    max_neurons = np.max(array_neurons)
    
    
    array_separated_shifted = array_separated.copy()
    mask = (array_separated_shifted > 0)
    array_separated_shifted[mask] += max_neurons
    
    array_combined = np.maximum(array_neurons, array_separated_shifted)
    

    img_combined = sitk.GetImageFromArray(array_combined)
    img_combined.CopyInformation(img_neurons)
    
    total_neurons = int(np.max(array_combined))
    print(f"Fusion : {max_neurons} (isolés) + {int(np.max(array_separated))} (séparés) = {total_neurons} neurones")
    
    return img_combined
