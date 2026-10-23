import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
import matplotlib.pyplot as plt
from .utils import *
import pandas as pd 



def get_objects_stat(image_label,image_originale=None):
    if isinstance(image_label, sitk.Image):
        shape_stats=sitk.LabeqqlShapeStatisticsImageFilter()
        shape_stats.Execute(image_label)
        donnees = []   
        for label in shape_stats.GetLabels():
            ligne = {
            'label': int(label),
            'taille_voxels': int(shape_stats.GetNumberOfPixels(label)),
            'elongation': float(shape_stats.GetElongation(label)),
            'flatness': float(shape_stats.GetFlatness(label)),
            'roundness': float(shape_stats.GetRoundness(label)),
            'centre_x': float(shape_stats.GetCentroid(label)[0]),
            'centre_y': float(shape_stats.GetCentroid(label)[1]),
            'centre_z': float(shape_stats.GetCentroid(label)[2]),
            'bbox_x': int(shape_stats.GetBoundingBox(label)[0]),
            'bbox_y': int(shape_stats.GetBoundingBox(label)[1]),
            'bbox_z': int(shape_stats.GetBoundingBox(label)[2]),
            'bbox_largeur': int(shape_stats.GetBoundingBox(label)[3]),
            'bbox_hauteur': int(shape_stats.GetBoundingBox(label)[4]),
            'bbox_profondeur': int(shape_stats.GetBoundingBox(label)[5]),
            }
            flatness = ligne['flatness']
            ligne['sphericite'] = 1.0 / max(flatness, 0.001)
            donnees.append(ligne)
        df=pd.DataFrame(donnees)
        df = pd.DataFrame(donnees)

        # Ajouter les statistiques d'intensité si une image originale est fournie
        if image_originale is not None and isinstance(image_originale, sitk.Image):
            intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
            intensity_stats.Execute(image_label, image_originale)
    
            df['intensite_moyenne'] = [intensity_stats.GetMean(l) for l in df['label']]
            df['intensite_std'] = [intensity_stats.GetStandardDeviation(l) for l in df['label']]
            df['intensite_min'] = [intensity_stats.GetMinimum(l) for l in df['label']]
            df['intensite_max'] = [intensity_stats.GetMaximum(l) for l in df['label']]
            df['intensite_median'] = [intensity_stats.GetMedian(l) for l in df['label']]

        print(f" Statistiques calculées : {len(df)} objets, {len(df.columns)} mesures") 
        return df
            
    else :
        print("errzeur image n'est pas un objet de simpleITK")
        return None
    
def save_stat_csv(dataframe, chemin_csv):
    
    dataframe.to_csv(chemin_csv, index=False,sep=';')
    print(f" CSV sauvegardé : {chemin_csv}")
    return 
