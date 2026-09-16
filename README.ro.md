<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <a href="README.pt.md">Português</a> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.ja.md">日本語</a> |
  <strong>Română</strong>
</p>

Acest repository conține codul pentru antrenarea, evaluarea, inferența și postprocesarea **NeuroTS-Net**, o arhitectură tridimensională de segmentare semantică multiclasă concepută pentru segmentarea tumorilor cerebrale pediatrice în imagini IRM multimodale.

Arhitectura NeuroTS-Net a fost dezvoltată de Darius Peteleaza. Pentru întrebări despre articol sau cod, vă rugăm să contactați [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

Codul însoțește lucrarea: **NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

Lucrarea a fost acceptată în cadrul [competiției BraTS 2026: BraTS-PEDs (Sarcina 2)](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097) - MICCAI 2026.

- **Volumul de lucrări Springer Nature:** lucrare acceptată; linkul va fi adăugat când va fi disponibil.
- **Preprint arXiv:** [https://arxiv.org/abs/2609.16873](https://arxiv.org/abs/2609.16873)
- **MICCAI 2026 - Poster BraTS-PEDs:** [Vezi posterul](figures/MICCAI-2026_BraTS-PEDs_Poster.png)

> [**Citare.**](#how-to-cite) Vă rugăm să citați lucrarea noastră dacă utilizați codul sau arhitectura NeuroTS-Net ori materiale din lucrarea asociată. Acest repository este distribuit sub [Licența Creative Commons Atribuire 4.0 Internațional](LICENSE).

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="Arhitectura NeuroTS-Net" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="Blocul NeuroTS" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="Mecanismul de downsampling NeuroTS" width="32%"></a>
</p>
<p align="center"><em>De la stânga la dreapta: prezentarea generală a arhitecturii NeuroTS-Net, blocul NeuroTS și mecanismul de downsampling NeuroTS.</em></p>

## Cerințe

- Sisteme de operare testate: Windows 11 și Ubuntu 22.04 LTS (codul este conceput să ruleze pe orice sistem de operare compatibil cu Python 3.11, PyTorch și dependențele specificate).
- Python 3.11.
- Pentru antrenare și inferență este recomandat un GPU compatibil CUDA. Execuția pe CPU este de asemenea acceptată, dar este considerabil mai lentă.

Pentru o instalare standard, instalați dependențele necesare rulării:

```bash
python -m pip install -r requirements.txt
```

Pentru o instalare editabilă de dezvoltare, inclusiv dependențele pentru teste, utilizați:

```bash
python -m pip install -e ".[dev]"
```

## Configurare

Acest proiect utilizează fișiere de configurare YAML pentru a defini căile și împărțirea datelor, arhitectura modelului, funcția de pierdere, antrenarea, augmentarea datelor, inferența și setările de ieșire.

Pentru descrierea fiecărui câmp, consultați [referința configurației](configs/CONFIGURATION.md).

## Setul de date

Configurațiile implicite vizează datele BraTS-PED cu următoarele modalități și etichete:

| Canal/etichetă | Semnificație |
|---|---|
| t1n | IRM nativ ponderat T1 |
| t1c | IRM ponderat T1 cu substanță de contrast |
| t2w | IRM ponderat T2 |
| t2f | IRM T2-FLAIR |
| 0 | Fundal |
| 1 | Tumoră cu captare de contrast (ET) |
| 2 | Tumoră fără captare de contrast (NET) |
| 3 | Componentă chistică (CC) |
| 4 | Edem (ED) |

Structura așteptată a setului de date este:

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

Subiecții sunt deduplicați după ID-ul cazului în toate directoarele rădăcină de antrenare înainte de împărțirea datelor.

## Preprocesarea datelor

Fluxul de preprocesare:

1. Citește cele patru modalități NIfTI fără reeșantionare;
2. Calculează o casetă de delimitare a prim-planului nenul pe toate modalitățile;
3. Extinde caseta cu marginea configurată (implicit 32 de voxeli);
4. Limitează fiecare modalitate la percentilele intensității din prim-plan;
5. Aplică normalizarea z-score exclusiv pe prim-plan;
6. Stochează imaginile ca matrice float32 împreună cu etichetele și metadatele spațiale.

Construiți memoria cache reutilizabilă:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Pentru a reconstrui intrările existente din cache:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

Predicțiile sunt reintegrate în forma originală a imaginii și salvate cu matricea afină, spațierea, orientarea și metadatele qform și sform ale imaginii sursă.

## Antrenarea modelului

Același punct de intrare pentru antrenare acceptă atât validarea încrucișată grupată, cât și o împărțire explicită antrenare/validare.

### Validare încrucișată cu cinci partiții

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Dacă fișierul cu partiții nu există, acesta este generat din cazurile de antrenare stocate în cache. Pentru a antrena o singură partiție configurată:

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### O singură împărțire antrenare/validare

Creați o împărțire stratificată după etichete și cu semnal ridicat:

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

Antrenați modelul pe această împărțire:

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

Când `data.split_mode` este `single`, fișierul de împărțire este obligatoriu. Antrenarea se oprește cu o eroare explicită în loc să revină în mod silențios la modul cu partiții.

## Eșantionare și funcția obiectiv

Configurațiile publice reproduc strategia de antrenare orientată către regiunile rare utilizată pentru NeuroTS-Net:

- Eșantionare echilibrată pe componente a patch-urilor pentru ET, NET, CC și ED;
- Patch-uri negative dificile centrate pe țesut tumoral fără ED din cazuri ED-negative;
- Entropie încrucișată ponderată și Dice pe etichetă, calculat pentru fiecare eșantion;
- Termeni Tversky/Dice pentru regiunile evaluate ET, NET, CC, ED, TC și WT;
- Penalizare a probabilității claselor absente pentru ET, CC și ED;
- Supervizare profundă, descreștere cosinusoidală a ratei de învățare și încălzire liniară.

Statisticile privind cererile de eșantionare, reușitele, revenirile, prezența claselor și voxelii țintă sunt scrise în jurnalul de antrenare și în `patch_sampling_latest.json`.

## Checkpoint-uri și metrici

Fiecare director de antrenare poate conține:

- `best_voxel_dice.pt`: cel mai bun Dice regional mediu la validarea pe patch-uri;
- `best_rare_present_score.pt`: cel mai bun scor de prezență a regiunilor rare la evaluarea volumului complet.

Evaluatorul pentru prezența regiunilor rare raportează separat cazurile cu GT prezent, predicție prezentă, TP, FP și FN. Cazurile în care o regiune rară este absentă atât din GT, cât și din predicție sunt doar diagnostice și nu îmbunătățesc selecția checkpoint-ului.

Rezultatele antrenării includ istorice compacte ale metricilor în format CSV/JSON, curbe de învățare, rapoarte de validare pe volum complet, jurnale și instantanee ale resurselor. Telemetria selecției ramurilor nu este afișată sau exportată în mod intenționat.

Validați fișierele YAML, inclusiv detectarea cheilor duplicate:

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## Inferență

Preprocesați datele de validare înainte de inferență:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### Checkpoint brut sau ansamblu de partiții

Descoperiți și mediați toate checkpoint-urile de prezență a regiunilor rare dintr-o rulare:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

Checkpoint-urile explicite pot fi repetate pentru a forma orice ansamblu:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### Postprocesarea componentelor ET/CC

Postprocesarea publică ajustată modifică probabilitățile și componentele ET și CC, păstrând în același timp predicțiile ED ale modelului principal:

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

Repetați `--checkpoint` pentru a media mai multe checkpoint-uri. Fișierele NIfTI sunt plasate în directorul de predicții, iar `--zip` creează o arhivă plată potrivită pentru încărcarea în cadrul provocării.

## Ieșiri

Datele generate sunt scrise în directoarele rădăcină configurate:

~~~text
cache/       matrice preprocesate și definiții ale împărțirilor/partițiilor
outputs/     checkpoint-uri, jurnale, metrici, predicții și arhive
~~~

Aceste directoare, seturile de date brute, checkpoint-urile și imaginile medicale sunt excluse prin `.gitignore`.

## Teste

Rulați suita rapidă de teste:

~~~bash
python -m pytest -q
~~~

Rulați testul rapid pentru propagarea înainte și înapoi prin model:

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## Citare

Vă rugăm să citați lucrarea noastră atunci când utilizați acest repository:

### Text simplu

```text
Peteleaza, D., Dumitru, R.-G., Neamtu, B., Gellert, A., Sandu, M., & Matei, C. (2026).
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI.
arXiv preprint arXiv:2609.16873. https://doi.org/10.48550/arXiv.2609.16873
```

### BibTeX

```bibtex
@misc{peteleaza2026neurotsnet,
  title         = {{NeuroTS-Net}: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal {MRI}},
  author        = {Peteleaza, Darius and Dumitru, Razvan-Gabriel and Neamtu, Bogdan and Gellert, Arpad and Sandu, Mariana and Matei, Claudiu},
  year          = {2026},
  eprint        = {2609.16873},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2609.16873},
  url           = {https://arxiv.org/abs/2609.16873}
}
```
