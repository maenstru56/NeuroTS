<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <a href="README.pt.md">Português</a> |
  <strong>Deutsch</strong> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ro.md">Română</a>
</p>

Dieses Repository enthält den Trainings-, Evaluierungs-, Inferenz- und Nachverarbeitungscode für **NeuroTS-Net**, eine dreidimensionale semantische Mehrklassensegmentierungsarchitektur zur Segmentierung pädiatrischer Hirntumoren in multimodaler MRT.

Die NeuroTS-Net-Architektur wurde von Darius Peteleaza entwickelt. Bei Fragen zur Veröffentlichung oder zum Code wenden Sie sich bitte an [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

Der Code gehört zur Veröffentlichung: **NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

Die Arbeit wurde für die [BraTS 2026 Challenge: BraTS-PEDs (Aufgabe 2)](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097) - MICCAI 2026 angenommen.

- **Springer-Nature-Tagungsband:** angenommen; der Link wird ergänzt, sobald er verfügbar ist.
- **arXiv-Preprint:** [https://arxiv.org/abs/2609.16873](https://arxiv.org/abs/2609.16873)
- **MICCAI 2026 - BraTS-PEDs-Poster:** [Poster ansehen](figures/MICCAI-2026_BraTS-PEDs_Poster.png)

> [**Zitierung.**](#how-to-cite) Bitte zitieren Sie unsere Veröffentlichung, wenn Sie den NeuroTS-Net-Code, die Architektur oder Material aus der zugehörigen Veröffentlichung verwenden. Dieses Repository steht unter der [Creative Commons Namensnennung 4.0 International Lizenz](LICENSE).

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="NeuroTS-Net-Architektur" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="NeuroTS-Block" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="NeuroTS-Downsampling-Mechanismus" width="32%"></a>
</p>
<p align="center"><em>Von links nach rechts: Überblick über die NeuroTS-Net-Architektur, NeuroTS-Block und NeuroTS-Downsampling-Mechanismus.</em></p>

## Voraussetzungen

- Getestete Betriebssysteme: Windows 11 und Ubuntu 22.04 LTS (der Code ist für die Ausführung auf jedem von Python 3.11, PyTorch und den aufgeführten Abhängigkeiten unterstützten Betriebssystem ausgelegt).
- Python 3.11.
- Für Training und Inferenz wird eine CUDA-kompatible GPU dringend empfohlen. Die Ausführung auf der CPU wird ebenfalls unterstützt, ist jedoch erheblich langsamer.

Installieren Sie für eine Standardinstallation die Laufzeitabhängigkeiten:

```bash
python -m pip install -r requirements.txt
```

Für eine editierbare Entwicklungsinstallation einschließlich der Testabhängigkeiten verwenden Sie:

```bash
python -m pip install -e ".[dev]"
```

## Konfiguration

Dieses Projekt verwendet YAML-Konfigurationsdateien, um Datenpfade und -aufteilungen, Modellarchitektur, Verlustfunktion, Training, Datenaugmentation, Inferenz und Ausgabeeinstellungen festzulegen.

Eine feldweise Beschreibung aller Einstellungen finden Sie in der [Konfigurationsreferenz](configs/CONFIGURATION.md).

## Datensatz

Die Standardkonfigurationen sind für BraTS-PED-Daten mit den folgenden Modalitäten und Labels ausgelegt:

| Kanal/Label | Bedeutung |
|---|---|
| t1n | Native T1-gewichtete MRT |
| t1c | Kontrastmittelverstärkte T1-gewichtete MRT |
| t2w | T2-gewichtete MRT |
| t2f | T2-FLAIR-MRT |
| 0 | Hintergrund |
| 1 | Kontrastmittelanreichernder Tumor (ET) |
| 2 | Nicht kontrastmittelanreichernder Tumor (NET) |
| 3 | Zystische Komponente (CC) |
| 4 | Ödem (ED) |

Die erwartete Datensatzstruktur lautet:

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

Vor der Aufteilung werden Probanden anhand der Fall-ID über alle Trainingsstammverzeichnisse hinweg dedupliziert.

## Datenvorverarbeitung

Die Vorverarbeitungspipeline:

1. Liest die vier NIfTI-Modalitäten ohne Resampling ein;
2. Berechnet über alle Modalitäten eine Begrenzungsbox des von null verschiedenen Vordergrunds;
3. Erweitert die Box um den konfigurierten Rand (standardmäßig 32 Voxel);
4. Begrenzt jede Modalität anhand der Intensitätsperzentile des Vordergrunds;
5. Führt eine z-Score-Normalisierung ausschließlich auf dem Vordergrund durch;
6. Speichert Bilder als float32-Arrays zusammen mit Labels und räumlichen Metadaten.

Erstellen Sie den wiederverwendbaren Cache:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

So erstellen Sie vorhandene Cache-Einträge neu:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

Vorhersagen werden in die ursprüngliche Bildform zurückgeführt und mit der affinen Matrix, dem Voxelabstand, der Orientierung sowie den qform- und sform-Metadaten des Quellbildes gespeichert.

## Modelltraining

Derselbe Trainingseinstieg unterstützt sowohl gruppierte Kreuzvalidierung als auch eine explizite Trainings-/Validierungsaufteilung.

### Fünffache Kreuzvalidierung

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Falls die Fold-Datei nicht existiert, wird sie aus den gecachten Trainingsfällen erzeugt. So trainieren Sie nur einen konfigurierten Fold:

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### Einzelne Trainings-/Validierungsaufteilung

Erstellen Sie eine nach Labels stratifizierte Aufteilung mit hohem Informationsgehalt:

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

Trainieren Sie mit dieser Aufteilung:

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

Wenn `data.split_mode` auf `single` gesetzt ist, ist die Aufteilungsdatei erforderlich. Das Training bricht mit einer eindeutigen Fehlermeldung ab, statt unbemerkt auf den Fold-Modus zurückzufallen.

## Sampling und Zielfunktion

Die öffentlichen Konfigurationen reproduzieren das in NeuroTS-Net verwendete Training mit besonderer Berücksichtigung seltener Regionen:

- Komponentenbalanciertes Patch-Sampling für ET, NET, CC und ED;
- Schwierige Negativ-Patches, die in ED-negativen Fällen auf Tumorgewebe ohne ED zentriert sind;
- Gewichtete Kreuzentropie und Label-Dice pro Stichprobe;
- Tversky-/Dice-Terme für die evaluierten Regionen ET, NET, CC, ED, TC und WT;
- Wahrscheinlichkeitsstrafe für die abwesenden Klassen ET, CC und ED;
- Tiefe Überwachung, Kosinusabfall der Lernrate und lineares Aufwärmen.

Statistiken zu Sampling-Anfragen, Erfolgen, Rückfällen, Klassenpräsenz und Zielvoxeln werden in das Trainingsprotokoll und in `patch_sampling_latest.json` geschrieben.

## Checkpoints und Metriken

Jedes Trainingsverzeichnis kann Folgendes enthalten:

- `best_voxel_dice.pt`: bester mittlerer Regions-Dice der Patch-Validierung;
- `best_rare_present_score.pt`: bester Präsenz-Score seltener Regionen bei Vollvolumenauswertung.

Der Präsenz-Evaluator für seltene Regionen meldet Fälle mit vorhandener Ground Truth, vorhandener Vorhersage, TP, FP und FN getrennt. Fälle, in denen eine seltene Region sowohl in der Ground Truth als auch in der Vorhersage leer ist, dienen nur der Diagnose und verbessern die Checkpoint-Auswahl nicht.

Die Trainingsausgaben umfassen kompakte CSV-/JSON-Metrikverläufe, Lernkurven, Vollvolumen-Validierungsberichte, Protokolle und Ressourcen-Snapshots. Die Telemetrie der Branch-Auswahl wird bewusst weder ausgegeben noch exportiert.

Validieren Sie die YAML-Dateien einschließlich der Erkennung doppelter Schlüssel:

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## Inferenz

Verarbeiten Sie die Validierungsdaten vor der Inferenz vor:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### Unverarbeiteter Checkpoint oder Fold-Ensemble

Ermitteln und mitteln Sie alle Rare-Present-Checkpoints eines Laufs:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

Explizite Checkpoints können für ein beliebiges Ensemble mehrfach angegeben werden:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### ET-/CC-Komponentennachverarbeitung

Die abgestimmte öffentliche Nachverarbeitung passt ET- und CC-Wahrscheinlichkeiten beziehungsweise Komponenten an und erhält zugleich die ED-Vorhersagen des Hauptmodells:

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

Wiederholen Sie `--checkpoint`, um mehrere Checkpoints zu mitteln. NIfTI-Dateien werden im Vorhersageverzeichnis abgelegt und `--zip` erstellt ein flaches Archiv für die Challenge-Einreichung.

## Ausgaben

Erzeugte Daten werden unterhalb der konfigurierten Stammverzeichnisse gespeichert:

~~~text
cache/       vorverarbeitete Arrays und Definitionen der Aufteilungen/Folds
outputs/     Checkpoints, Protokolle, Metriken, Vorhersagen und Archive
~~~

Diese Verzeichnisse sowie Rohdatensätze, Checkpoints und medizinische Bilder werden durch `.gitignore` ausgeschlossen.

## Tests

Führen Sie die schnelle Testsuite aus:

~~~bash
python -m pytest -q
~~~

Führen Sie den Smoke-Test für Vorwärts- und Rückwärtsdurchlauf des Modells aus:

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## Zitierung

Bitte zitieren Sie unsere Veröffentlichung, wenn Sie dieses Repository verwenden:

### Klartext

```

```

### BibTeX

```

```
