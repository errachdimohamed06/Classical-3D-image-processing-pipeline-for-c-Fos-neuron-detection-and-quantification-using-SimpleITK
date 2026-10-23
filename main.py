"""
Pipeline principal pour la détection des neurones c-Fos+
Utilisation : python main.py --config config.yaml
"""
import sys
import os
import yaml
import argparse
import SimpleITK as sitk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import des modules du pipeline
from src import preprocessing
from src import segmentation
from src import filtering
from src import fusion
from src import statistics
from src import utils

def main(config_path):
    # 1. Charger la configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_full_path = os.path.join(script_dir, config_path)

    with open(config_full_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Récupérer le nom de l’échantillon
    echantillon_nom = config['echantillon']['nom']
    chemin_base = config['chemins']['base']
    chemin_input = config['chemins']['input']
    chemin_output = config['chemins']['output']
    chemin_stats = config['chemins']['stats']
    
    
    # 3. Construire les chemins complets
    input_dir = f"{chemin_base}/{echantillon_nom}/{chemin_input}"
    output_dir = f"{chemin_base}/{echantillon_nom}/{chemin_output}"
    stats_dir = f"{chemin_base}/{echantillon_nom}/{chemin_stats}"
    
    # 4. Créer les dossiers si besoin
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # 5. Lire l’image
    image_path = f"{input_dir}/cfos.nrrd"
    image = utils.read_nrrd_image(image_path)
    utils.show_image_info(image)
    
    # 6. Prétraitement
    if config['preprocessing']['normalize_z']:
        image = preprocessing.normaliser_coupes_z(image)
    
    image = preprocessing.filtre_median_3d(image, rayon=config['preprocessing']['median_radius'])
    image = preprocessing.top_hat_3d(image, rayon=config['preprocessing']['tophat_radius'])
    
    # 7. Seuillage double
    binary = segmentation.seuillage_double(
        image,
        seuil_min=config['segmentation']['threshold_min'],
        seuil_max=config['segmentation']['threshold_max']
    )
    
    # 8. Dilatation
    binary = preprocessing.dilate_binary_image(
        binary,
        rayon_xy=config['segmentation']['dilate_radius_xy'],
        rayon_z=config['segmentation']['dilate_radius_z']
    )
    
    # 9. Création image label
    label_img = segmentation.create_image_label(binary)
    
    # 10. Filtrage neurones
    img_neurons, img_multi, img_deleted, df = filtering.filter_neurons(label_img)
    
    # 11. Sauvegarder les images intermédiaires
    utils.save_image(img_neurons, f"{output_dir}/neurones_isolés.nrrd")
    utils.save_image(img_multi, f"{output_dir}/neurones_colles.nrrd")
    if config['sauvegarde']['save_deleted']:
        utils.save_image(img_deleted, f"{output_dir}/objets_supprimes.nrrd")
    
    # 12. Statistiques globales
    statistics.save_stat_csv(df, f"{stats_dir}/statistiques_globales.csv")
    
    # 13. Vérifier s’il y a des objets collés

    array_multi = sitk.GetArrayFromImage(img_multi)
    
    if np.max(array_multi) > 0:
        print(f" {int(np.max(array_multi))} objets collés détectés. Séparation en cours...")
        
        # Séparer les neurones collés
        img_separated = filtering.separate_collapsed_neurons_by_distance_threshold_Danielson(
            img_multi,
            lower=config['separation']['lower_threshold'],
            upper=config['separation']['upper_threshold'],
            min_object_size=config['separation']['min_object_size'],
            dilate_radius=config['separation']['dilate_radius'],
            use_danielsson=(config['separation']['method'] == "danielsson")
        )
        
        utils.save_image(img_separated, f"{output_dir}/neurones_separes.nrrd")
        
        # 14. Statistiques des séparés
        df_sep = statistics.get_objects_stat(img_separated, image)
        statistics.save_stat_csv(df_sep, f"{stats_dir}/statistiques_separes.csv")
        
        # 15. Fusion finale
        img_final = fusion.combine_neurons(img_neurons, img_separated)
        utils.save_image(img_final, f"{output_dir}/neurones_finaux.nrrd")
        
        # 16. Statistiques finales
        df_final = statistics.get_objects_stat(img_final, image)
        statistics.save_stat_csv(df_final, f"{stats_dir}/statistiques_finales.csv")
        
        # 17. Afficher résumé
        nb_final = len(df_final)
        
    else:
        print(" Aucun objet collé détecté. Séparation et fusion non nécessaires.")
        
        # Pas de séparation, l’image finale = neurones isolés
        img_final = img_neurons
        utils.save_image(img_final, f"{output_dir}/neurones_finaux.nrrd")
        
        # Sauvegarder un DataFrame vide pour les stats des séparés
        import pandas as pd
        df_sep = pd.DataFrame()
        statistics.save_stat_csv(df_sep, f"{stats_dir}/statistiques_separes.csv")
        
        # Statistiques finales = statistiques des isolés
        statistics.save_stat_csv(df, f"{stats_dir}/statistiques_finales.csv")
        
        nb_final = len(df)
    
    # Afficher résumé final
    print(f"\n Pipeline terminé pour {echantillon_nom}")
    print(f" Nombre total de neurones détectés : {nb_final}")
    print(f" Résultats dans : {output_dir}")
    print(f" Statistiques dans : {stats_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Chemin vers config.yaml")
    args = parser.parse_args()
    main(args.config)