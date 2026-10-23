import numpy as np
import SimpleITK as sitk
import os
import sys
import gc
import tqdm
import matplotlib.pyplot as plt
"""
ce fichier utiles.py est creer pour implementer les fonctions qui vas nous servir et etre utiles pour toutes les scripts pour notre projet 
HappySmel  

"""

def read_nrrd_image(chemin):
    '''
    SUMMARY.

    Parameters
    ----------
    chemin : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    Raises
    ------
    FileNotFoundError
        DESCRIPTION.
    '''
    if not os.path.exists(chemin):
        raise FileNotFoundError(f"Le fichier {chemin} n'existe pas")
    reader=sitk.ImageFileReader()
    reader.SetImageIO("NrrdImageIO")
    reader.SetFileName(chemin)
    return reader.Execute()


def save_image(image,chemin):
    writter=sitk.ImageFileWriter()
    writter.SetFileName(chemin)
    writter.Execute(image)
    return


def show_image_info(image):
    if isinstance(image, sitk.Image):
        print(" Informations de l'image :")
        print(f"   - Dimensions : {image.GetSize()}")
        print(f"   - Espacement : {image.GetSpacing()}")
        print(f"   - Type : {image.GetPixelIDTypeAsString()}")
        print(f"   - Direction : {image.GetDirection()}")
        print(f"   - Origine : {image.GetOrigin()}")
    else:
        print("Erreur :ce n'est pas une image nrrd")
    return


def show_array_info(array):
    if isinstance(array,np.ndarray):
        print(" Informations de tableau representant l'image : ")
        print(f"FORME : {array.shape}")
        print(f"NOMBRE DE DIMENSIONS : {array.ndim}")
        print(f" TYPE DES DONNÉES : {array.dtype}")
    else:
        print("Erreur :ce n'est pas une image nrrd")
        return  


def get_imag_info_dictionnary(image):
    if isinstance(image, sitk.Image):
        dict={
            "dimensions":image.GetSize(),
            "espacement":image.GetSpacing(),
            "direction":image.GetDirection(),
            "origine":image.GetOrigin()
            }
    elif isinstance(image, np.ndarray):
        dict={
            "dimensions":image.ndim,
            "forme":image.shape,
            }
    else:
        dict=None
    return dict


def convert_image_to_array_xyz(image):
    return sitk.GetArrayFromImage(image).transpose((2,1,0))

def convert_image_to_array(image):
    return sitk.GetArrayFromImage(image)

def convert_array_to_image(array):
    return sitk.GetImageFromArray(array)
def convert_array_to_image_with_information_ref(array,image_ref):
    if isinstance(array,np.ndarray) and isinstance(image_ref, sitk.Image):
        return sitk.GetImageFromArray(array).CopyInformation(image_ref)
    elif isinstance(array,np.ndarray):
        print("l'image reference est invalide je retourne le tableau sous forme d'image sans information de reference")
        return sitk.GetImageFromArray(array)
    elif isinstance(image_ref, sitk.Image):
       print("le tableau a convertir est invalide je retourne l'image de reference")
       return image_ref
    else :
        print("le tableau a convertir est invalide de meme l'image de reference est invalide")
        return None


def save_array(array,chemin):
    writter=sitk.ImageFileWriter()
    writter.SetFileName(chemin)
    writter.Execute(convert_array_to_image(array))
    return


def get_volume_from_image(
    image,
    z_debut,
    z_fin,
    y_debut=None,
    y_fin=None,
    x_debut=None,
    x_fin=None
):

    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
    else:
        array = image

    if y_debut is None:
        y_debut = 0
    if y_fin is None:
        y_fin = array.shape[1]

    if x_debut is None:
        x_debut = 0
    if x_fin is None:
        x_fin = array.shape[2]

    volume_array = array[
        z_debut:z_fin,
        y_debut:y_fin,
        x_debut:x_fin
    ]

    volume = sitk.GetImageFromArray(volume_array)

    if isinstance(image, sitk.Image):

        volume.SetSpacing(image.GetSpacing())
        volume.SetDirection(image.GetDirection())

        new_origin = (
            image.GetOrigin()[0] + x_debut * image.GetSpacing()[0],
            image.GetOrigin()[1] + y_debut * image.GetSpacing()[1],
            image.GetOrigin()[2] + z_debut * image.GetSpacing()[2]
        )

        volume.SetOrigin(new_origin)

    return volume



def get_volume_from_array(array,z_debut,z_fin,y_debut=None,y_fin=None,x_debut=None,x_fin=None):
    
    if y_debut is None:
        y_debut=0
    if y_fin is None:
        y_fin=array.shape[1]
    if x_debut is None:
        x_debut=0
    if x_fin is None:
        x_fin=array.shape[2]
    
    return array[z_debut:z_fin,y_debut:y_fin,x_debut:x_fin]

def get_slices_from_image(image,z_debut,z_fin):
    return get_volume_from_image(image, z_debut, z_fin)

def get_slices_from_array(array,z_debut,z_fin):
    return get_volume_from_array(array, z_debut, z_fin)

def get_one_slice_from_image(image,position):
    if isinstance(image,sitk.Image):
        array=convert_image_to_array(image)
    else:
        array=image
    return sitk.GetImageFromArray(array[position,:,:])


def get_one_slice_from_array(array,position):
    if isinstance(array,sitk.Image):
        array=convert_image_to_array(array)
    return sitk.GetImageFromArray(array[position,:,:])


def show_one_slice2D(image,position,titre="Tranche", cmap='gray', figsize=(10, 10)):
    slice=get_one_slice_from_image(image, position)
    if isinstance(slice,sitk.Image):
        slice=convert_image_to_array(slice)
    plt.figure(figsize=figsize)
    plt.imshow(slice, cmap=cmap)
    plt.colorbar(label='intensité')
    plt.title(f"{titre} numero {position}")
    plt.xlabel("X (colonne)")
    plt.ylabel("Y (ligne)")
    plt.show()
    return
 
def progress_bar(iterable,description='Progression'):
    '''
    SUMMARY.

    Parameters
    ----------
    iterable : Elements a parcourir.
    description : Message a afficher.
        

    Returns un tqdm.tqdm : une barre de progression
    -------
    '''
    return tqdm.tqdm(iterable,desc=description)


def apply_transformation_for_all_slices(image,function_to_apply,barre_progression=False):
    
    if isinstance(image,sitk.Image):
        array=convert_image_to_array(image)
    else:
        array=image
    
    resultats=[]
    itterator=range(array.shape[0])
    
    if barre_progression:
        itterator=progress_bar(itterator)
    for z in itterator:
        tranche=array[z,:,:]
        resultats.append(function_to_apply(tranche))
    return resultats




def show_total_object_image_label(image_label):
    if isinstance(image_label, sitk.Image):
        total=sitk.LabelShapeStatisticsImageFilter()
        total.Execute(image_label)
        totales=len(total.GetLabels())
        print("totale des objets dans l'image label est :",totales)
        
    else:
        print("errzeur image n'est pas un objet de simpleITK")
        return None
    
def convert_label_image_to_binary_uint8(image_label):
    image_binaire = sitk.Cast(image_label > 0, sitk.sitkUInt8)

    image_binaire = image_binaire * 255

    image_binaire.CopyInformation(image_label)

    return image_binaire            















