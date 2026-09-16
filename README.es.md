<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <strong>Español</strong> |
  <a href="README.fr.md">Français</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ro.md">Română</a>
</p>

Este repositorio contiene el código de entrenamiento, evaluación, inferencia y posprocesamiento de **NeuroTS-Net**, una arquitectura tridimensional de segmentación semántica multiclase diseñada para segmentar tumores cerebrales pediátricos en MRI multimodal.

La arquitectura NeuroTS-Net fue desarrollada por Darius Peteleaza. Para cualquier consulta sobre el artículo o el código, contacte con [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

El código acompaña al artículo: **NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

El artículo ha sido aceptado en el [desafío BraTS 2026: BraTS-PEDs (Tarea 2)](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097) - MICCAI 2026.

- **Actas de Springer Nature:** aceptado; el enlace se añadirá cuando esté disponible.
- **Prepublicación en arXiv:** [https://arxiv.org/abs/2609.16873](https://arxiv.org/abs/2609.16873)
- **MICCAI 2026 - Póster de BraTS-PEDs:** [Ver el póster](figures/MICCAI-2026_BraTS-PEDs_Poster.png)

> [**Cita.**](#how-to-cite) Cite nuestro artículo si utiliza el código o la arquitectura NeuroTS-Net, o material del artículo asociado. Este repositorio se distribuye bajo la [Licencia Creative Commons Atribución 4.0 Internacional](LICENSE).

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="Arquitectura NeuroTS-Net" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="Bloque NeuroTS" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="Mecanismo de submuestreo NeuroTS" width="32%"></a>
</p>
<p align="center"><em>De izquierda a derecha: descripción general de la arquitectura NeuroTS-Net, bloque NeuroTS y mecanismo de submuestreo NeuroTS.</em></p>

## Requisitos

- Sistemas operativos probados: Windows 11 y Ubuntu 22.04 LTS (el código está diseñado para ejecutarse en cualquier sistema operativo compatible con Python 3.11, PyTorch y las dependencias indicadas).
- Python 3.11.
- Se recomienda encarecidamente una GPU compatible con CUDA para el entrenamiento y la inferencia. También se admite la ejecución en CPU, aunque es considerablemente más lenta.

Para una instalación estándar, instale las dependencias de ejecución:

```bash
python -m pip install -r requirements.txt
```

Para una instalación editable de desarrollo, incluidas las dependencias de pruebas, use:

```bash
python -m pip install -e ".[dev]"
```

## Configuración

Este proyecto utiliza archivos YAML para definir las rutas y particiones de los datos, la arquitectura del modelo, la pérdida, el entrenamiento, el aumento de datos, la inferencia y los ajustes de salida.

Consulte la [referencia de configuración](configs/CONFIGURATION.md) para obtener una descripción campo por campo de todos los ajustes.

## Conjunto de datos

Las configuraciones predeterminadas están dirigidas a datos BraTS-PED con las siguientes modalidades y etiquetas:

| Canal/etiqueta | Significado |
|---|---|
| t1n | MRI ponderada en T1 nativa |
| t1c | MRI ponderada en T1 con contraste |
| t2w | MRI ponderada en T2 |
| t2f | MRI T2-FLAIR |
| 0 | Fondo |
| 1 | Tumor realzado (ET) |
| 2 | Tumor no realzado (NET) |
| 3 | Componente quístico (CC) |
| 4 | Edema (ED) |

La estructura esperada del conjunto de datos es:

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

Los sujetos se deduplican por identificador de caso entre las raíces de entrenamiento antes de crear las particiones.

## Preprocesamiento de datos

El flujo de preprocesamiento:

1. Lee las cuatro modalidades NIfTI sin remuestreo;
2. Calcula una caja delimitadora del primer plano no nulo sobre todas las modalidades;
3. Amplía la caja con el margen configurado (32 vóxeles de forma predeterminada);
4. Recorta cada modalidad según los percentiles de intensidad del primer plano;
5. Aplica normalización z-score únicamente sobre el primer plano;
6. Almacena las imágenes como matrices float32 junto con las etiquetas y los metadatos espaciales.

Construya la caché reutilizable:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Para reconstruir entradas existentes de la caché:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

Las predicciones se restauran a la forma original de la imagen y se guardan con la matriz afín, el espaciado, la orientación y los metadatos qform y sform de la imagen de origen.

## Entrenamiento del modelo

La misma entrada de entrenamiento admite tanto validación cruzada agrupada como una partición explícita de entrenamiento/validación.

### Validación cruzada de cinco pliegues

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Si el archivo de pliegues no existe, se genera a partir de los casos de entrenamiento almacenados en caché. Para entrenar solo un pliegue configurado:

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### Partición única de entrenamiento/validación

Cree una partición estratificada por etiquetas y de alta señal:

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

Entrene con esa partición:

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

Cuando `data.split_mode` es `single`, el archivo de partición es obligatorio. El entrenamiento falla con un error claro en lugar de volver silenciosamente al modo de pliegues.

## Muestreo y función objetivo

Las configuraciones públicas reproducen la estrategia de entrenamiento sensible a regiones poco frecuentes utilizada para NeuroTS-Net:

- Muestreo de parches equilibrado por componentes para ET, NET, CC y ED;
- Parches negativos difíciles centrados en tejido tumoral sin ED de casos negativos para ED;
- Entropía cruzada ponderada y Dice por etiqueta y por muestra;
- Términos Tversky/Dice para las regiones evaluadas ET, NET, CC, ED, TC y WT;
- Penalización de probabilidad para clases ausentes ET, CC y ED;
- Supervisión profunda, decaimiento cosenoidal de la tasa de aprendizaje y calentamiento lineal.

Las estadísticas de solicitudes, éxitos, alternativas, presencia de clases y vóxeles objetivo del muestreo se escriben en el registro de entrenamiento y en `patch_sampling_latest.json`.

## Puntos de control y métricas

Cada directorio de entrenamiento puede contener:

- `best_voxel_dice.pt`: mejor Dice regional medio en la validación por parches;
- `best_rare_present_score.pt`: mejor puntuación de presencia de regiones poco frecuentes en volumen completo.

El evaluador de presencia poco frecuente informa por separado los casos con GT presente, predicción presente, TP, FP y FN. Los casos en los que una región poco frecuente está vacía tanto en GT como en la predicción son únicamente diagnósticos y no mejoran la selección del punto de control.

Las salidas de entrenamiento incluyen historiales compactos de métricas CSV/JSON, curvas de aprendizaje, informes de validación de volumen completo, registros e instantáneas de recursos. La telemetría de selección de ramas no se imprime ni se exporta deliberadamente.

Valide los archivos YAML, incluido el rechazo de claves duplicadas:

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## Inferencia

Preprocese los datos de validación antes de la inferencia:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### Punto de control sin procesar o conjunto de pliegues

Descubra y promedie todos los puntos de control de presencia poco frecuente de una ejecución:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

Se pueden repetir puntos de control explícitos para formar cualquier conjunto:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### Posprocesamiento de componentes ET/CC

El posprocesamiento público ajustado modifica las probabilidades y los componentes de ET y CC, a la vez que conserva las predicciones ED del modelo principal:

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

Repita `--checkpoint` para promediar varios puntos de control. Los archivos NIfTI se escriben en el directorio de predicciones y `--zip` crea un archivo plano apto para su envío al desafío.

## Salidas

Los datos generados se escriben bajo las raíces configuradas:

~~~text
cache/       matrices preprocesadas y definiciones de particiones/pliegues
outputs/     puntos de control, registros, métricas, predicciones y archivos
~~~

Estos directorios, los conjuntos de datos sin procesar, los puntos de control y las imágenes médicas están excluidos mediante `.gitignore`.

## Pruebas

Ejecute el conjunto rápido de pruebas:

~~~bash
python -m pytest -q
~~~

Ejecute la prueba rápida de propagación hacia delante y hacia atrás del modelo:

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## Cita

Si utiliza este repositorio, cite nuestro artículo:

### Texto sin formato

```

```

### BibTeX

```

```
