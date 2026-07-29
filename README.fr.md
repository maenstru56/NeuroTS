<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.es.md">Español</a> |
  <strong>Français</strong> |
  <a href="README.pt.md">Português</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ro.md">Română</a>
</p>

Ce dépôt contient le code d'entraînement, d'évaluation, d'inférence et de post-traitement de **NeuroTS-Net**, une architecture tridimensionnelle de segmentation sémantique multiclasse conçue pour la segmentation des tumeurs cérébrales pédiatriques en IRM multimodale.

L'architecture NeuroTS-Net a été développée par Darius Peteleaza. Pour toute question concernant l'article ou le code, veuillez contacter [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

Le code accompagne l'article : **NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

L'article est actuellement en cours d'évaluation à MICCAI 2026 dans le cadre du [challenge BraTS](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097).

- Lien vers la publication MICCAI/BraTS : à ajouter.
- Lien vers la prépublication arXiv : à ajouter.

> [**Citation.**](#how-to-cite) Veuillez citer notre article si vous utilisez le code ou l'architecture NeuroTS-Net, ou du contenu provenant de l'article associé. Ce dépôt est distribué sous la [Licence Creative Commons Attribution 4.0 International](LICENSE).

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="Architecture NeuroTS-Net" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="Bloc NeuroTS" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="Mécanisme de sous-échantillonnage NeuroTS" width="32%"></a>
</p>
<p align="center"><em>De gauche à droite : vue d'ensemble de l'architecture NeuroTS-Net, bloc NeuroTS et mécanisme de sous-échantillonnage NeuroTS.</em></p>

## Prérequis

- Systèmes d'exploitation testés : Windows 11 et Ubuntu 22.04 LTS (le code est conçu pour fonctionner sur tout système d'exploitation pris en charge par Python 3.11, PyTorch et les dépendances indiquées).
- Python 3.11.
- Un GPU compatible CUDA est fortement recommandé pour l'entraînement et l'inférence. L'exécution sur CPU est également prise en charge, mais elle est considérablement plus lente.

Pour une installation standard, installez les dépendances d'exécution :

```bash
python -m pip install -r requirements.txt
```

Pour une installation de développement modifiable, incluant les dépendances de test, utilisez :

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Ce projet utilise des fichiers de configuration YAML pour définir les chemins et la répartition des données, l'architecture du modèle, la fonction de perte, l'entraînement, l'augmentation des données, l'inférence et les paramètres de sortie.

Pour une description détaillée de chaque paramètre, consultez la [référence de configuration](configs/CONFIGURATION.md).

## Jeu de données

Les configurations par défaut ciblent les données BraTS-PED avec les modalités et étiquettes suivantes :

| Canal/étiquette | Signification |
|---|---|
| t1n | IRM pondérée en T1 native |
| t1c | IRM pondérée en T1 avec contraste |
| t2w | IRM pondérée en T2 |
| t2f | IRM T2-FLAIR |
| 0 | Arrière-plan |
| 1 | Tumeur rehaussée (ET) |
| 2 | Tumeur non rehaussée (NET) |
| 3 | Composante kystique (CC) |
| 4 | Œdème (ED) |

La structure attendue du jeu de données est la suivante :

~~~text
data/
|-- BraTS26_PED_training/
|   |-- BraTS-PED-00001-000/
|   |   |-- BraTS-PED-00001-000-t1n.nii.gz
|   |   |-- BraTS-PED-00001-000-t1c.nii.gz
|   |   |-- BraTS-PED-00001-000-t2w.nii.gz
|   |   |-- BraTS-PED-00001-000-t2f.nii.gz
|   |   |-- BraTS-PED-00001-000-seg.nii.gz
|   |-- ...
|-- BraTS26_PED_training_Batch2_Release/
|   |-- ...
|-- BraTS26_PED_validation/
    |-- BraTS-PED-XXXXX-XXX/
    |   |-- BraTS-PED-XXXXX-XXX-t1n.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t1c.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2w.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2f.nii.gz
    |-- ...
~~~

Les sujets sont dédupliqués entre les racines d'entraînement selon leur identifiant de cas avant la création des répartitions.

## Prétraitement des données

Le pipeline de prétraitement :

1. Lit les quatre modalités NIfTI sans rééchantillonnage ;
2. Calcule une boîte englobante du premier plan non nul sur l'ensemble des modalités ;
3. Étend cette boîte selon la marge configurée (32 voxels par défaut) ;
4. Écrête chaque modalité selon les percentiles d'intensité du premier plan ;
5. Applique une normalisation z-score uniquement sur le premier plan ;
6. Stocke les images sous forme de tableaux float32, avec les étiquettes et les métadonnées spatiales.

Construisez le cache réutilisable :

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Pour reconstruire les entrées de cache existantes :

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

Les prédictions sont réinsérées dans la forme d'image originale et enregistrées avec la matrice affine, l'espacement, l'orientation et les métadonnées qform et sform de l'image source.

## Entraînement du modèle

Le même point d'entrée d'entraînement prend en charge la validation croisée groupée et une répartition explicite entraînement/validation.

### Validation croisée à cinq plis

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Le fichier de plis est généré à partir des cas d'entraînement mis en cache s'il n'existe pas. Pour n'entraîner qu'un seul pli configuré :

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### Répartition unique entraînement/validation

Créez une répartition stratifiée par étiquette et à signal élevé :

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

Entraînez le modèle avec cette répartition :

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

Lorsque `data.split_mode` vaut `single`, le fichier de répartition est obligatoire. L'entraînement s'arrête avec une erreur explicite au lieu de revenir silencieusement au mode par plis.

## Échantillonnage et fonction objectif

Les configurations publiques reproduisent la stratégie d'entraînement sensible aux régions rares utilisée pour NeuroTS-Net :

- Échantillonnage de patchs équilibré par composante pour ET, NET, CC et ED ;
- Patchs négatifs difficiles centrés sur du tissu tumoral sans ED provenant de cas ED négatifs ;
- Entropie croisée pondérée et Dice par étiquette et par échantillon ;
- Termes Tversky/Dice pour les régions évaluées ET, NET, CC, ED, TC et WT ;
- Pénalité de probabilité de classe absente pour ET, CC et ED ;
- Supervision profonde, décroissance cosinusoïdale du taux d'apprentissage et préchauffage linéaire.

Les statistiques de demande, de réussite, de repli, de présence des classes et de voxels cibles de l'échantillonnage sont écrites dans le journal d'entraînement et dans `patch_sampling_latest.json`.

## Points de contrôle et métriques

Chaque répertoire d'entraînement peut contenir :

- `best_voxel_dice.pt` : meilleur Dice régional moyen lors de la validation par patchs ;
- `best_rare_present_score.pt` : meilleur score de présence des régions rares en volume complet.

L'évaluateur de présence des régions rares rapporte séparément les cas avec vérité terrain présente, prédiction présente, TP, FP et FN. Les cas où une région rare est vide à la fois dans la vérité terrain et dans la prédiction ne servent qu'au diagnostic et n'améliorent pas la sélection du point de contrôle.

Les sorties d'entraînement comprennent des historiques compacts de métriques CSV/JSON, des courbes d'apprentissage, des rapports de validation en volume complet, des journaux et des instantanés de ressources. La télémétrie de sélection des branches n'est volontairement ni affichée ni exportée.

Validez les fichiers YAML, y compris le rejet des clés dupliquées :

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## Inférence

Prétraitez les données de validation avant l'inférence :

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### Point de contrôle brut ou ensemble de plis

Détectez et moyennez tous les points de contrôle de présence des régions rares d'une exécution :

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

Des points de contrôle explicites peuvent être répétés pour former n'importe quel ensemble :

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### Post-traitement des composantes ET/CC

Le post-traitement public ajusté modifie les probabilités et les composantes ET et CC tout en préservant les prédictions ED du modèle principal :

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

Répétez `--checkpoint` pour moyenner plusieurs points de contrôle. Les fichiers NIfTI sont placés dans le répertoire des prédictions et `--zip` crée une archive plate adaptée à l'envoi au challenge.

## Sorties

Les données générées sont écrites sous les racines configurées :

~~~text
cache/       tableaux prétraités et définitions des répartitions/plis
outputs/     points de contrôle, journaux, métriques, prédictions et archives
~~~

Ces répertoires, les jeux de données bruts, les points de contrôle et les images médicales sont exclus par `.gitignore`.

## Tests

Exécutez la suite de tests rapide :

~~~bash
python -m pytest -q
~~~

Exécutez le test rapide de propagation avant/arrière du modèle :

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## Citation

Veuillez citer notre article lorsque vous utilisez ce dépôt :

### Texte brut

```

```

### BibTeX

```

```
