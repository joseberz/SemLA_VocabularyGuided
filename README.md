# Vokabulargeleitete Selektion von LoRA-Adaptern mittels CLIP für domänenadaptive Open-Vocabulary-Segmentierung

Dieses Repository enthält den Code zur gleichnamigen Masterarbeit. Die Arbeit baut auf SemLA [1] auf und untersucht, ob sich die dort vorgeschlagene, rein visuelle Adapterauswahl durch ein zusätzliches Vokabularsignal verbessern lässt.

[SemLA-Code](https://github.com/rezaqorbani/SemLA) | [SemLA-Paper](https://doi.org/10.1109/CVPR52734.2025.00916)

## Kurzbeschreibung

SemLA wählt zur Testzeit passende LoRA-Adapter für ein Segmentierungsmodell (CAT-Seg) allein anhand der visuellen Ähnlichkeit zwischen Eingabebild und den Trainingsdatensätzen der verfügbaren Adapter. Diese Arbeit erweitert die Adapterauswahl um eine zusätzliche Vokabular-Distanz, indem aus dem Eingabebild über CLIP ein Vokabular abgeleitet und mit dem Klassenvokabular der Adapter verglichen. Beide Distanzsignale (visuell und vokabularbasiert) werden gewichtet kombiniert, um die Adapterselektion zu bestimmen.

Untersucht werden drei Strategien zur Ableitung des Bildvokabulars (Global, Patch, Objekterkennung), die gegen die unveränderte SemLA-Baseline sowie eine normalisierte Variante davon verglichen werden.

## Setup

Das grundlegende Projekt-Setup, die Beschaffung der SemLA-Datensätze sowie der Aufbau der Adapterbibliothek sind in [README_SemLA.md](README_SemLA.md) beschrieben. Bitte zuerst diese Anleitung durchgehen. \
_Hinweis: Die Installation von detectron2 kann je nach Betriebssystem / Python-Version variieren. Für die Installation von detectron2 kann die offizielle [Dokumentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) verwendet werden._

Nach dem SemLA-Setup:
Um die Datensatz-Zentroid-Embeddings, sowie die Adapter-Vokabular-Embeddings zu erstellen, bitte folgende Befehle aus dem Root-Verzeichnis ausführen:
   ```bash
   cd catseg
   python ../generate_embeddings.py --source_domains_file ../config/source_domains.yaml --lora_library_path ./loradb
   ```

### Vokabularübergreifender Testdatensatz

Zusätzlich zu den SemLA-Datensätzen wird für die Evaluation ein eigens erstellter, vokabularübergreifender Testdatensatz benötigt:

1. Das zugehörige Repository klonen:
   ```bash
   git clone https://github.com/joseberz/VokabularuebergreifenderTestdatensatz
   ```
2. Im geklonten Projekt die Bildordner mit dem dort enthaltenen Skript befüllen (siehe [README](https://github.com/joseberz/VokabularuebergreifenderTestdatensatz/blob/main/README.md) des Projekts).
3. Die fertig befüllten Ordner aus dem geklonten Projekt in den `detectron2`-Datensatz-Ordner dieses Projekts verschieben.

## Verwendung

Alle Experimente werden über `experiments.py` gestartet.

### Einzelner Durchlauf einer Methode auf den Standard-Datensatz:

```bash
python experiments.py --experiment semla \
    --source_domains config/source_domains.yaml \
    --target_domains config/target_domains.yaml \
    --semla_config config/semla_config_best_none.yaml \
    --voc_distance_method none \
    --output_dir results/none
```

### Einzelner Durchlauf einer Methode auf dem Vokabularübergreifenden Testdatensatz:

```bash
python experiments.py --experiment semla \
    --source_domains config/source_domains_novel.yaml \
    --target_domains config/target_domains_novel.yaml \
    --semla_config config/semla_config_best_none.yaml \
    --voc_distance_method none \
    --output_dir results/none_voc_extension_dataset
```

Für die anderen Methoden `--voc_distance_method` entsprechend anpassen (`none_normalized`, `global`, `patch`, `objectdetection`) und eine passende `--semla_config` übergeben.
Unter dem Ordner configs/ liegen die passenden Konfigurationen mit bereits definierten Hyperparametern.

### Bayessche Optimierung

```bash
python experiments.py --experiment bo_optimize \
    --source_domains config/source_domains.yaml \
    --target_domains config/target_domains.yaml \
    --semla_config config/semla_config_best_patch.yaml \
    --voc_distance_method patch \
    --output_dir results/bo_patch \
    --bo_init_points 5 \
    --bo_n_iter 15
```

Der Zustand des Optimizers wird nach jeder Iteration automatisch gespeichert (`optimizer_state.json` im Ausgabeordner). Ein unterbrochener Lauf kann mit `--bo_resume_state <pfad>` fortgesetzt werden.

### Ablationsstudie

```bash
python experiments.py --experiment grid_norm_ablation \
    --source_domains config/source_domains.yaml \
    --target_domains config/target_domains.yaml \
    --semla_config config/semla_config_best_patch.yaml \
    --voc_distance_method patch \
    --normalization_method zscore \
    --output_dir results/ablation_norm
```

Die übrigen Ablationen (`grid_centroid_ablation`, `grid_distance_metric_ablation`, `grid_topq_frac_ablation`, `grid_tau_k_sensitivity_ablation`, `grid_search_weighting_ablation`) folgen demselben Muster; die jeweils relevanten Parameter sind in `experiments.py` als `GRID_*`-Konstanten hinterlegt.

Alle verfügbaren Parameter: `python experiments.py --help`.

## Qualitative Vokabularerweiterung

Für die qualitativen Free-Vocabulary-Beispiele (siehe Fazit der Arbeit) steht `vocab_expansion_experiments.py` bereit. Das Skript segmentiert eine Auswahl an Bildern mit einem um die Adaptervokabulare erweiterten Eingabevokabular, statt auf das ursprüngliche Vokabular des Testdatensatzes beschränkt zu sein.

Die verwendeten Pfade (Quell-/Zieldomänen, SemLA-Konfiguration, Bildliste, Ausgabeordner) werden am Ende der Datei im `main`-Block gesetzt und dort vor dem Aufruf angepasst:

```bash
python vocab_expansion_experiments.py
```

`image_list_file` verweist auf eine YAML-Datei mit den Pfaden der auszuwertenden Testbilder. \
`top_x` bestimmt, wie viele zusätzliche Vokabularkandidaten pro Bild einbezogen werden sollen.


## Visualisierung

Auswertungsplots (Adapter-Gewichtungs-Heatmaps, Coverage-Korrelationen) werden über `visualize.py` erzeugt:

```bash
python visualize.py --plot heatmaps
python visualize.py --plot domain-bar
python visualize.py --plot coverage-bins
```

Die benötigten Pfade zu den `correlation_log.json`-Dateien werden am Kopf von `visualize.py` konfiguriert.
Die `correlation_log.json`-Dateien werden im Zuge der Hauptexperimente im Ergebnisordner erstellt. Für die BO- und Ablationsexperimente werden diese Dateien nicht erstellt.

# Lizenz
## Lizenz

Dieses Repository steht, wie die zugrundeliegende Arbeit SemLA [1], unter der Apache-2.0-Lizenz, siehe [LICENSE](LICENSE) für die Details. Einzelne Teile unterliegen abweichenden Lizenzbedingungen, die aus SemLA übernommen wurden: [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) steht unter der MIT-Lizenz, einsehbar unter [catseg/LICENSE](catseg/LICENSE). Darüber hinaus werden einzelne Dateien aus [Detectron2](https://github.com/facebookresearch/detectron2) und [FC-CLIP](https://github.com/bytedance/fc-clip) (Apache-2.0-Lizenz) sowie aus [Mask2Former](https://github.com/facebookresearch/Mask2Former) (MIT-Lizenz) verwendet.

# Quellen
> [1] R. Qorbani, G. Villani, T. Panagiotakopoulos, M. B. Colomer, L. Härenstam-Nielsen, M. Segu, P. L. Dovesi, J. Karlgren, D. Cremers, F. Tombari, M. Poggi, "Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation," in *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025, S. 9804–9815, doi: 10.1109/CVPR52734.2025.00916.