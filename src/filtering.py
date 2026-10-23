import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
import matplotlib.pyplot as plt
from .utils import *
import pandas as pd 


def filter_neurons(image_label):
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(image_label)
    
    array = sitk.GetArrayFromImage(image_label)
    
    array_neurons = np.zeros_like(array)
    array_deleted = np.zeros_like(array)
    array_multi = np.zeros_like(array)
    
    n_neurons = 0
    n_deleted = 0
    n_multi = 0
    
    data = []
    
    for label in shape_stats.GetLabels():
        taille = shape_stats.GetNumberOfPixels(label)
        flatness = shape_stats.GetFlatness(label)
        elongation = shape_stats.GetElongation(label)
        sphericite = 1.0 / max(flatness, 0.0001)
        
        if taille <=100:
            status = "neuron_isolated"
            n_neurons += 1
            array_neurons[array == label] = n_neurons
        
        if taille >100 and taille <= 1000:
            if sphericite >= 0.68:
                status = "neuron_isolated"
                n_neurons += 1
                array_neurons[array == label] = n_neurons
            else:
               status = "neuron_multi"
               n_multi += 1
               array_multi[array == label] = n_multi
        
        if taille>1000 and taille<= 2000:
            if sphericite >= 0.85:
                status = "neuron_isolated"
                n_neurons += 1
                array_neurons[array == label] = n_neurons
            else:
                if sphericite >= 0.40:
                    status = "neuron_multi"
                    n_multi += 1
                    array_multi[array == label] = n_multi
                else:
                    bbox = shape_stats.GetBoundingBox(label)
                    bbox_volume = bbox[3] * bbox[4] * bbox[5]
                    density = taille / max(bbox_volume, 1)
                    if density >=0.27 and elongation <=2 :
                        status = "neuron_multi"
                        n_multi += 1
                        array_multi[array == label] = n_multi
                    else:
                        status = "deleted"
                        n_deleted += 1
                        array_deleted[array == label] = n_deleted
                        
        if taille>2000 and taille<= 2300:
            if sphericite >0.50:
                status = "neuron_multi"
                n_multi += 1
                array_multi[array == label] = n_multi
            else :
                bbox = shape_stats.GetBoundingBox(label)
                bbox_volume = bbox[3] * bbox[4] * bbox[5]
                density = taille / max(bbox_volume, 1)
                if density >=0.27 and elongation <=2 :
                    status = "neuron_multi"
                    n_multi += 1
                    array_multi[array == label] = n_multi
                else:
                    status = "deleted"
                    n_deleted += 1
                    array_deleted[array == label] = n_deleted
                    
        if taille>2300 and taille<= 4000:
            if sphericite >0.52:
                status = "neuron_multi"
                n_multi += 1
                array_multi[array == label] = n_multi
            else :
                bbox = shape_stats.GetBoundingBox(label)
                bbox_volume = bbox[3] * bbox[4] * bbox[5]
                density = taille / max(bbox_volume, 1)
                if density >=0.27 and elongation <=2 and sphericite >0.40 :
                    status = "neuron_multi"
                    n_multi += 1
                    array_multi[array == label] = n_multi
                else:
                    status = "deleted"
                    n_deleted += 1
                    array_deleted[array == label] = n_deleted
        if taille>4000:
            status = "deleted"
            n_deleted += 1
            array_deleted[array == label] = n_deleted
            
        
        data.append({
            'label': label,
            'taille_voxels': taille,
            'sphericite': round(sphericite, 4),
            'elongation': round(elongation, 4),
            'status': status
        })
    
    img_neurons = sitk.GetImageFromArray(array_neurons)
    img_neurons.CopyInformation(image_label)
    
    img_multi = sitk.GetImageFromArray(array_multi)
    img_multi.CopyInformation(image_label)
    
    img_deleted = sitk.GetImageFromArray(array_deleted)
    img_deleted.CopyInformation(image_label)
    

    df = pd.DataFrame(data)
    
    print(f"Single neurons: {n_neurons}")
    print(f"Multi-neurons (to separate): {n_multi}")
    print(f"Deleted (vessels/artifacts): {n_deleted}")
    
    return img_neurons, img_multi, img_deleted, df




def separate_collapsed_neurons_by_distance_threshold_Danielson(image_multi, lower=2,upper=3, min_object_size=5, dilate_radius=1, use_danielsson=True):
    
    mask_binary = image_multi > 0
    
    
    if use_danielsson:
        distance_filter = sitk.SignedDanielssonDistanceMapImageFilter()
        distance_filter.SetInsideIsPositive(True)
        distance_filter.SetUseImageSpacing(False)
        distance = distance_filter.Execute(mask_binary)
    else:
        distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
        distance_filter.SetInsideIsPositive(True)
        distance_filter.SetSquaredDistance(False)
        distance = distance_filter.Execute(mask_binary)
    
    
    seuil = sitk.BinaryThresholdImageFilter()
    seuil.SetLowerThreshold(lower)
    seuil.SetUpperThreshold(upper)
    seuil.SetInsideValue(1)
    seuil.SetOutsideValue(0)
    centres = seuil.Execute(distance)
    

    connected = sitk.ConnectedComponentImageFilter()
    connected.SetFullyConnected(True)
    centres_label = connected.Execute(centres)

    if dilate_radius > 0:
        dilate = sitk.BinaryDilateImageFilter()
        dilate.SetKernelRadius([dilate_radius, dilate_radius, 1])
        dilate.SetKernelType(sitk.sitkBall)
        dilated = dilate.Execute(centres_label > 0)
        result = dilated * (image_multi > 0)
        result_label = connected.Execute(result)
    else:
        result_label = centres_label
    
    relabel = sitk.RelabelComponentImageFilter()
    result_final = relabel.Execute(result_label)
    
    
    if min_object_size > 0:
        relabel_min = sitk.RelabelComponentImageFilter()
        relabel_min.SetMinimumObjectSize(min_object_size)
        result_final = relabel_min.Execute(result_final)
    
    filter_name = "Danielsson" if use_danielsson else "Maurer"
    print(f"Distance map {filter_name},  {relabel.GetNumberOfObjects()} objets finaux")
    
    return result_final



