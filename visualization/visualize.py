import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict

from vis_utils import get_class_names_from_ids, normalize_class_list, ADAPTER_VOCAB_JSONS, \
    aggregate_weights_cross_vocab, GOOD_ADAPTERS_FOR_OBJECT, is_good_adapter, avg_per_adapter, normalize_class_name

# Neue Voc-Coverage-Metrik auf Basis der Klassen, nicht der kompletten GTs
# TODO prüfen
def weighted_class_coverage(
        class_name: str,
        adapter_weights: Dict[str, float],
        adapter_vocabs: Dict[str, List[str]],
) -> float:
    class_name_normalized = normalize_class_name(class_name)[0]

    coverage_score = 0
    for n, adapter_name in enumerate(adapter_weights.keys()):
        if adapter_name not in adapter_vocabs:
            raise KeyError(
                f"Adapter '{adapter_name}' nicht in Adaptervokabularen gefunden.\n"
            )
        adapter_classes = adapter_vocabs[adapter_name]

        if class_name_normalized in adapter_classes:
            coverage_score += adapter_weights[adapter_name]
    return coverage_score


def correlate_match_vs_mismatch_boxplot(df, mismatch_threshold):
    """
    # Boxplot
    # Daten einteilen in Mismatch und Match, also neue Boolean Spalte (wenn coverage > 0 => Match)
    # Dann die Zwei Gruppen in Abhängigkeit des mIoU vergleichen
    # Ebene: Global und Pro-Domäne
    # Immer ein Plot pro Klasse, da der Schwierigkeitsgrad der Klasse den mIoU-Wert beeinflusst!
    :param df:
    :param mismatch_threshold:
    :return:
    """
    new_df = df.assign(match=lambda x: x.coverage_1 > mismatch_threshold)
    unique_domains = new_df['domain'].unique()

    for domain_name in unique_domains:
        domain_df = new_df[new_df['domain'] == domain_name]
        sns.boxplot(data=domain_df, x="class_name", y="miou_1", hue="match")

        plt.xlabel("Coverage")
        plt.ylabel("mIoU")
        plt.title(f"Boxplot für Mismatch Vergleich in der Domain {domain_name}")
        plt.show()

def correlate_methods_match_vs_mismatch_boxplot(df):
    # delta_miou = Verbesserung durch PATCH gegenüber NONE
    new_df = df.assign(match=df["delta_cov"] > 0)
    unique_domains = new_df["domain"].unique()
    unique_classes = new_df["class_name"].unique()

    for domain_name in unique_domains:
        domain_df = new_df[new_df["domain"] == domain_name]

        # Pro Klasse einen Boxplot
        for class_name in unique_classes:
            class_df = domain_df[domain_df["class_name"] == class_name]
            if len(class_df) == 0:
                print("yxes")
                continue

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=class_df, x="match", y="delta_miou", hue="match", width=0.3, ax=ax)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xticklabels(["Match", "Mismatch"])
            ax.set_xlabel("")
            ax.set_ylabel("Diff mIoU (PATCH − NONE)")
            ax.set_title(f"{domain_name}: {class_name}")
            plt.tight_layout()
            plt.show()

def correlate_general_coverage_vs_class_iou(df):
    """
    # Alle Zeilen (classes) plotten, eine Zeile entspricht einer Klasse auf einem spezifischen Bild
    # x = coverage_1 oder coverage_2
    # y = miou_1 oder miou_2
    # als Farbe zB nach unterschdl Domänen (oder Klassennamen) einfärben
    # Ebene: Global und Pro-Domäne
    # Immer ein Plot pro Klasse, da der Schwierigkeitsgrad der Klasse den mIoU-Wert beeinflusst!
    """
    # df columns:
    # domain
    # class_name
    # coverage_1
    # coverage_2
    # miou_1
    # miou_2
    # delta_cov
    # delta_miou
    unique_domains = df['domain'].unique()
    unique_classes = df['class_name'].unique()

    sns.regplot(x="coverage_1", y="miou_1", data=df, scatter_kws={"alpha": 0.4})

    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.xlabel("Coverage")
    plt.ylabel("mIoU")
    plt.title(f"Coverage vs. mIoU")
    plt.show()

    for domain_name in unique_domains:
        for class_name in unique_classes:
            new_df = df[(df['domain'] == domain_name) & (df['class_name'] == class_name)]
            if len(new_df) == 0:
                continue

            sns.regplot(x="coverage_1", y="miou_1", data=new_df, scatter_kws={"alpha": 0.4})
            plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")

            pad = 0.05 * (new_df["coverage_1"].max() - new_df["coverage_1"].min())
            plt.xlim(new_df["coverage_1"].min() - pad, new_df["coverage_1"].max() + pad)

            plt.xlabel("Coverage")
            plt.ylabel("mIoU")
            plt.title(f"Coverage vs. mIoU: Domäne {domain_name}, Klasse {class_name}")
            plt.show()
            plt.close()

    for class_name in unique_classes:
        new_df = df[df['class_name'] == class_name]
        if len(new_df) == 0:
            continue

        sns.regplot(x="coverage_1", y="miou_1", data=new_df, scatter_kws={"alpha": 0.4})
        plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")

        pad = 0.05 * (new_df["coverage_1"].max() - new_df["coverage_1"].min())
        plt.xlim(new_df["coverage_1"].min() - pad, new_df["coverage_1"].max() + pad)

        plt.xlabel("Coverage")
        plt.ylabel("mIoU")
        plt.title(f"Coverage vs. mIoU: Klasse {class_name}, domänenübergreifend")
        plt.show()
        plt.close()

def get_df_class_coverage_vs_iou(evaluation_data_1: list[Dict],
                                    evaluation_data_2: list[Dict]) -> pd.DataFrame:
    sorted_method1 = evaluation_data_1.copy()
    sorted_method2 = evaluation_data_2.copy()
    assert all(a["image_path"] == b["image_path"] for a, b in zip(sorted_method1, sorted_method2)), \
        "Reihenfolge zwischen Bildern in Methode 1 und Methode 2 stimmt nicht überein!"

    all_adapters_and_classes = load_all_adapter_classes()
    adapters_classes_normalized = {adapter: normalize_class_list(classes) for adapter, classes in all_adapters_and_classes.items()}


    # per_class_miou hat form: {"class_name": miou}}
    # class_miou_1 hat form: [{"class_name": miou}] => ein dict pro Bild
    class_miou_1 = [eval_result["per_class_miou"] for eval_result in sorted_method1]
    class_miou_2 = [eval_result["per_class_miou"] for eval_result in sorted_method2]
    adapter_weights_1 = [eval_result["adapter_weights"] for eval_result in sorted_method1]
    adapter_weights_2 = [eval_result["adapter_weights"] for eval_result in sorted_method2]
    domains = [eval_result["domain"] for eval_result in sorted_method1]

    class_coverage_score_1 = []
    class_coverage_score_2 = []
    for i, class_miou_dict in enumerate(class_miou_1):
        class_coverage_dict_1 = {}
        class_coverage_dict_2 = {}
        for class_name in class_miou_dict.keys():
            class_coverage_dict_1[class_name] = weighted_class_coverage(class_name, adapter_weights_1[i], adapters_classes_normalized)
            class_coverage_dict_2[class_name] = weighted_class_coverage(class_name, adapter_weights_2[i], adapters_classes_normalized)
        class_coverage_score_1.append(class_coverage_dict_1)
        class_coverage_score_2.append(class_coverage_dict_2)

    rows = []
    for i in range(len(sorted_method1)):
        for class_name, miou_1 in class_miou_1[i].items():
            miou_2 = class_miou_2[i].get(class_name, np.nan)
            rows.append({
                "domain":      domains[i],
                "class_name":  class_name,
                "coverage_1":  class_coverage_score_1[i].get(class_name, 0),
                "coverage_2":  class_coverage_score_2[i].get(class_name, 0),
                "miou_1":      miou_1['miou'],
                "miou_2":      miou_2['miou'],
                "delta_cov":   class_coverage_score_2[i].get(class_name, 0) - class_coverage_score_1[i].get(class_name, 0),
                "delta_miou":  miou_2['miou'] - miou_1['miou'],
            })
    df = pd.DataFrame(rows)

    df["miou_1_z"] = df.groupby("class_name")["miou_1"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df["miou_2_z"] = df.groupby("class_name")["miou_2"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    print(df["coverage_1"].describe())
    print(df["coverage_1"].value_counts(normalize=True).head(10))
    print(df["delta_cov"].describe())

    return df


def voc_coverage_score(adapter_set_per_image: List[Dict], domains: List[str], gt_ids: List[List[int]]) -> list[float]:
    all_adapters_and_classes = load_all_adapter_classes()

    gt_classes = [
        get_class_names_from_ids(domain, ids, all_adapters_and_classes)
        for domain, ids in zip(domains, gt_ids)
    ]

    # Normalisierung der Klassenbezeichnungen
    gt_classes = [normalize_class_list(classes) for classes in gt_classes]
    # Normalisierung der Adapter-Vokabulare
    adapters_classes_normalized = {adapter: normalize_class_list(classes) for adapter, classes in all_adapters_and_classes.items()}

    coverage_scores = []
    for i, adapter_name_and_weight in enumerate(adapter_set_per_image):
        gt_classes_image_set = set(gt_classes[i])

        if len(gt_classes_image_set) == 0:
            coverage_scores.append(float("nan"))
            continue

        coverage_score_image = 0
        for n, adapter_name in enumerate(adapter_name_and_weight):
            if adapter_name not in adapters_classes_normalized:
                raise KeyError(
                    f"Adapter '{adapter_name}' nicht in Adaptervokabularen gefunden.\n"
                )
            adapter_classes = adapters_classes_normalized[adapter_name]

            overlap = len(gt_classes_image_set.intersection(set(adapter_classes)))
            coverage_score_image += (adapter_name_and_weight[adapter_name]*10) * (overlap / len(gt_classes_image_set))
        coverage_scores.append(coverage_score_image)

    return coverage_scores

def adapter_weight_shift(adapter_weights_1: list[Dict], adapter_weights_2: list[Dict]) -> list[float]:
    print("Adapter Weights 1: ", len(adapter_weights_1))
    print("Adapter Weights 2: ", len(adapter_weights_2))

    adapter_weight_shift_values = []
    for i, adapter_weight in enumerate(adapter_weights_1):
        # Bilde Vereinigungsmenge der Adapter
        all_keys = set(adapter_weight.keys()) | set(adapter_weights_2[i].keys())
        # Berechne TDV als statistische Differenz der Adaptergewichtsverteilung
        shift = 0
        for key in all_keys:
            shift += abs(adapter_weights_2[i].get(key, 0) - adapter_weight.get(key, 0))
        adapter_weight_shift_values.append(shift)
    return adapter_weight_shift_values

def correlation_adapter_weight_iou(evaluation_data_1: list[Dict], evaluation_data_2: list[Dict], specific_domain="", calculate_per_domain=False):
    sorted_method1 = evaluation_data_1.copy()
    sorted_method2 = evaluation_data_2.copy()

    if specific_domain:
        filtered_indices = [
            i for i, eval_result in enumerate(sorted_method1)
            if eval_result["domain"] == specific_domain
        ]
        sorted_method1 = [sorted_method1[i] for i in filtered_indices]
        sorted_method2 = [sorted_method2[i] for i in filtered_indices]

        if len(sorted_method1) == 0:
            print(f"Keine Daten für Domäne '{specific_domain}' gefunden.")
            return None

    assert all(a["image_path"] == b["image_path"] for a, b in zip(sorted_method1, sorted_method2)), \
        "Reihenfolge zwischen Bildern in Methode 1 und Methode 2 stimmt nicht überein!"

    domains = [eval_result["domain"] for eval_result in sorted_method1]

    adapter_weights_1 = [eval_result["adapter_weights"] for eval_result in sorted_method1]
    adapter_weights_2 = [eval_result["adapter_weights"] for eval_result in sorted_method2]

    image_iou_1 = [eval_result["per_image_miou"] for eval_result in sorted_method1]
    image_iou_2 = [eval_result["per_image_miou"] for eval_result in sorted_method2]

    adapter_weight_shift_values = adapter_weight_shift(adapter_weights_1, adapter_weights_2)
    delta_miou = [m2 - m1 for m1, m2 in zip(image_iou_1, image_iou_2)]

    df = pd.DataFrame({
        "domain": domains,
        "adapter_weight_shift": adapter_weight_shift_values,
        "delta_miou": delta_miou,
    })
    df = df.dropna(subset=["adapter_weight_shift", "delta_miou"])

    if calculate_per_domain:
        correlation_adapter_weight_shift_iou_per_domain(df)

    sns.regplot(x="adapter_weight_shift", y="delta_miou", data=df, scatter_kws={"alpha": 0.4})
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    title_suffix = f"Domäne: {specific_domain}" if specific_domain else ""
    plt.xlabel("Adapter Weight Shift (Methode 2 − Methode 1)")
    plt.ylabel("Diff. mIoU (Methode 2 − Methode 1)")
    plt.title(f"Adapter Weight Shift vs. Diff. mIoU{title_suffix}")
    plt.show()

    return df

def prepare_correlation_data(data_method1: List[Dict], data_method2: List[Dict]) -> tuple[list[Dict], list[Dict]]:
    target_key = "image_path"

    # baue hifsdict: {"image1.png": {"image_path": "image1.png", "weights": [...]"}}
    by_path_1 = {item[target_key]: item for item in data_method1 if target_key in item}
    by_path_2 = {item[target_key]: item for item in data_method2 if target_key in item}

    shared_paths = sorted(by_path_1.keys() & by_path_2.keys())

    # Bilder ohne gültige GT-IDs ausschließen
    # (z.B. Masken, die komplett aus ignore_label bestehen
    valid_paths = [
        p for p in shared_paths
        if by_path_1[p].get("gt_ids") != [] and by_path_2[p].get("gt_ids") != []
    ]

    filtered_sorted_method1 = [by_path_1[p] for p in valid_paths]
    filtered_sorted_method2 = [by_path_2[p] for p in valid_paths]

    return filtered_sorted_method1, filtered_sorted_method2

def load_all_adapter_classes() -> Dict[str, List[str]]:
    classes = {}

    for adapter_name, json_path in ADAPTER_VOCAB_JSONS.items():
        with open(json_path) as f:
            class_names = json.load(f)
            class_names = [class_name.lower() for class_name in class_names]
            classes[adapter_name] = class_names
    return classes

def correlation_voccoveragescore_iou(evaluation_data_1: list[Dict], evaluation_data_2: list[Dict], calculate_per_domain=False):
    sorted_method1 = evaluation_data_1.copy()
    sorted_method2 = evaluation_data_2.copy()
    assert all(a["image_path"] == b["image_path"] for a, b in zip(sorted_method1, sorted_method2)), \
        "image_path-Reihenfolge zwischen Methode 1 und Methode 2 stimmt nicht überein!"

    adapter_weights_1 = [eval_result["adapter_weights"] for eval_result in sorted_method1]
    adapter_weights_2 = [eval_result["adapter_weights"] for eval_result in sorted_method2]
    domains = [eval_result["domain"] for eval_result in sorted_method1]
    gt_ids = [eval_result["gt_ids"] for eval_result in sorted_method1]

    voc_coverages_1 = voc_coverage_score(adapter_weights_1, domains, gt_ids)
    voc_coverages_2 = voc_coverage_score(adapter_weights_2, domains, gt_ids)

    image_iou_1 = [eval_result["per_image_miou"] for eval_result in sorted_method1]
    image_iou_2 = [eval_result["per_image_miou"] for eval_result in sorted_method2]

    print(sum(image_iou_1) / len(image_iou_1))
    print(sum(image_iou_2) / len(image_iou_2 ))

    delta_voc_coverage = [c2 - c1 for c1, c2 in zip(voc_coverages_1, voc_coverages_2)]
    delta_miou = [m2 - m1 for m1, m2 in zip(image_iou_1, image_iou_2)]

    df = pd.DataFrame({
        "domain": domains,
        "delta_voc_coverage_score": delta_voc_coverage,
        "delta_miou": delta_miou,
    })
    df = df.dropna(subset=["delta_voc_coverage_score", "delta_miou"])

    if calculate_per_domain:
        correlation_voccoveragescore_iou_per_domain(df)

    sns.regplot(x="delta_voc_coverage_score", y="delta_miou", data=df, scatter_kws={"alpha":0.4})
    plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.xlabel("Diff. Vocabulary Coverage Score (Methode 2 − Methode 1)")
    plt.ylabel("Diff. mIoU (Methode 2 − Methode 1)")
    plt.show()

    return df

def correlation_voccoveragescore_iou_per_domain(df: pd.DataFrame):
    df_domain = df.copy()
    df_domain = df_domain.dropna(subset=["delta_voc_coverage_score", "delta_miou"])

    domains = sorted(df_domain["domain"].unique())

    for domain_name in domains:
        sub = df_domain[df_domain["domain"] == domain_name]
        if len(sub) == 0:
            continue

        plt.figure(figsize=(6, 5))
        sns.regplot(
            x="delta_voc_coverage_score", y="delta_miou", data=sub,
            scatter_kws={"alpha": 0.4, "s": 15},
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)

        plt.xlabel("Diff. Voc Coverage Score")
        plt.ylabel("Diff. mIoU")
        plt.title(f"Diff. Vocabulary Coverage Score vs. Diff. mIoU: {domain_name}")
        plt.tight_layout()
        plt.show()
        plt.close()

    return df_domain

def correlation_adapter_weight_shift_iou_per_domain(df: pd.DataFrame):
    df_domain = df.copy()
    df_domain = df_domain.dropna(subset=["adapter_weight_shift", "delta_miou"])

    domains = sorted(df_domain["domain"].unique())

    for domain_name in domains:
        sub = df_domain[df_domain["domain"] == domain_name]
        if len(sub) == 0:
            continue

        x = sub["adapter_weight_shift"]

        plt.figure(figsize=(6, 5))
        sns.regplot(
            x="adapter_weight_shift", y="delta_miou", data=sub,
            scatter_kws={"alpha": 0.4, "s": 15},
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)

        pad = 0.05 * (x.max() - x.min())
        xlo, xhi = x.min() - pad, x.max() + pad
        if xlo <= 0 <= xhi:
            plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.xlim(xlo, xhi)

        plt.xlabel("Diff. Adapter Weight Shift")
        plt.ylabel("Diff. mIoU")
        plt.title(f"Adapter Weight Shift vs. Diff. mIoU: {domain_name}")
        plt.tight_layout()
        plt.show()
        # TODO save
        plt.close()

    return df_domain

def plot_heatmap_cross_voc_dataset_all_objects(weights_list, labels):
    available = sorted(o for o in GOOD_ADAPTERS_FOR_OBJECT if o in weights_list[0])
    for obj in available:
        plot_heatmap_cross_voc_dataset_specific_object(obj, weights_list, labels)

def plot_heatmap_cross_voc_dataset_specific_object(specific_object, weights_list, labels):
    assert specific_object in weights_list[0], print(f"'{specific_object}' nicht in weights.json gefunden.")

    adapters = sorted({a for w in weights_list if specific_object in w for a in w[specific_object].keys()})

    # Bauen des Dataframes für seaborn
    rows = []
    for w, label in zip(weights_list, labels):
        if specific_object not in w:
            continue
        avgs = avg_per_adapter(w[specific_object])
        for a in adapters:
            rows.append({"Adapter": a, "Run": label, "Gewicht": avgs.get(a, 0.0)})
    df = pd.DataFrame(rows)
    print(df)

    # Baue den Plot in Abhängigkleit von der Anzahl individueller Adapter
    n_adapt = len(adapters)
    fig, ax = plt.subplots(figsize=(max(9, n_adapt * 1.1 + 2), 5))

    # Hebe die GT-Adapter welche das Objekt enthalten farblich hervor
    for j, a in enumerate(adapters):
        if is_good_adapter(a, specific_object):
            ax.axvspan(j - 0.5, j + 0.5, color="#d4edda", alpha=0.45, zorder=0)

    # Seaborn Plot
    sns.barplot(
        data=df, x="Adapter", y="Gewicht", hue="Run",
        order=adapters, hue_order=labels,
        palette="muted", alpha=0.85, ax=ax, zorder=2,
    )

    # Füge die durchschnitt. Werte der Adapter über den Balken ein
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, color="#333333", padding=2)

    # Label an der X-Achse / Namen der Adapter / "Gute" Adapter sollen grün und fett hervorgehoben werden
    ax.set_xticklabels(adapters, rotation=45, ha="right", fontsize=9)
    for ticklabel, a in zip(ax.get_xticklabels(), adapters):
        if is_good_adapter(a, specific_object):
            ticklabel.set_color("#1a7a1a")
            ticklabel.set_fontweight("bold")
        else:
            ticklabel.set_color("#555555")

    # TODO die neuen Klassen ergänzen
    source_domain = None
    if specific_object in ["bag", "bench", "flag", "river", "trash can", "umbrella"]:
        source_domain = "IDD"
    if specific_object in ["bridge", "fan", "picture", "refrigerator", "traffic light"]:
        source_domain = "PC59"
    if specific_object in ["bicycle", "dog", "keyboard", "potted plant", "teddy bear"]:
        source_domain = "NYU"

    ax.set_ylabel("Durchschn. Adapter-Gewicht", fontsize=10)
    ax.set_xlabel("Adapter (Source Domäne)", fontsize=10)
    ax.set_title(f"Neues Objekt '{specific_object}' in {source_domain} ", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legende
    good_patch = mpatches.Patch(color="#d4edda", label="Objekt im Adapter-Vokabular")
    run_handles, run_labs = ax.get_legend_handles_labels()
    ax.legend(handles=[good_patch] + run_handles,
              labels=["OObjekt im Adapter-Vokabular"] + run_labs,
              fontsize=9, loc="upper right")

    plt.tight_layout()

    plt.show()
    plt.close()

def plot_adapter_contribution_heatmap(weights: List[Dict], threshold: float = 0.1):
    order_columns = ["acdc-rain", "acdc-snow", "acdc-fog", "acdc-night",
                     "muses-clear-day", "muses-clear-night", "muses-rain-day", "muses-rain-night", "muses-fog-day", "muses-fog-night", "muses-snow-day", "muses-snow-night",
                     "cs-normal", "bdd", "mv", "idd",
                     "a133", "pc59", "nyu", "coconutL"]

    weights_data = weights.copy()

    rows = []
    for entry in weights_data:
        domain = entry["domain"]
        for adapter, weight in entry["adapter_weights"].items():
            rows.append({"domain": domain, "adapter": adapter, "weight": weight})

    df = pd.DataFrame(rows)

    matrix = df.pivot_table(index="domain", columns="adapter", values="weight", aggfunc="mean", fill_value=0)
    rows_order = [d for d in order_columns if d in matrix.index]
    cols_order = [d for d in order_columns if d in matrix.columns]

    # nach reihenfolge sortieren
    matrix = matrix.reindex(index=rows_order, columns=cols_order)

    display_matrix = matrix.mask(matrix < threshold)

    fig, ax = plt.subplots(figsize=(max(10, len(matrix.columns) * 0.7),
                                    max(8, len(matrix.index) * 0.5)))

    sns.heatmap(
        display_matrix,
        annot=True, fmt=".1f",
        cmap="Blues",
        linewidths=0.5, linecolor="white",
        mask=display_matrix.isna(),
        ax=ax,
        robust=True,
        square=True,
    )

    ax.set_xlabel("LoRA-Adapter")
    ax.set_ylabel("Test-Datensatz")
    ax.set_title("Adapter-Anteil-Heatmap")
    plt.tight_layout()
    plt.show()

    pass

def load_evaluation_data(path: str) -> list[Dict]:
    with open(path) as json_file:
        return json.load(json_file)


def main():
    ################################
    ###### Correlation Data!! ######
    ################################
    path_1 = "/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/final_A/none_3/correlation_log.json"
    path_2 = "/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/final_A/patch_3/correlation_log.json"

    evaluation_data_1 = load_evaluation_data(path_1)
    evaluation_data_2 = load_evaluation_data(path_2)

    evaluation_data_1, evaluation_data_2 = prepare_correlation_data(evaluation_data_1, evaluation_data_2)

    print(f"Length method 1: {len(evaluation_data_1)}\n")
    print(f"Length method 2: {len(evaluation_data_2)}\n")

    #correlation_voccoveragescore_iou(evaluation_data_1, evaluation_data_2, calculate_per_domain=True)

    #correlation_adapter_weight_iou(evaluation_data_1, evaluation_data_2, calculate_per_domain=True)

    df = get_df_class_coverage_vs_iou(evaluation_data_1, evaluation_data_2)

    # Alle Zeilen (classes) plotten, eine Zeile entspricht einer Klasse auf einem spezifischen Bild
    # x = coverage_1 oder coverage_2
    # y = miou_1 oder miou_2
    # als Farbe zB nach unterschdl Domänen (oder Klassennamen) einfärben
    # Ebene: Global und Pro-Domäne
    # Immer ein Plot pro Klasse, da der Schwierigkeitsgrad der Klasse den mIoU-Wert beeinflusst!
    # Pro Domäne könnte aufschlussreicher sein!
    #correlate_general_coverage_vs_class_iou(df)

    # Boxplot
    # Daten einteilen in Mismatch und Match, also neue Boolean Spalte (wenn coverage > 0 => Match)
    # Dann die Zwei Gruppen in Abhängigkeit des mIoU vergleichen
    # Ebene: Global und Pro-Domäne
    # Immer ein Plot pro Klasse, da der Schwierigkeitsgrad der Klasse den mIoU-Wert beeinflusst!
    # Pro Domäne könnte aufschlussreicher sein!
    #correlate_match_vs_mismatch_boxplot(df, mismatch_threshold=0.1)
    #correlate_methods_match_vs_mismatch_boxplot(df)


    # df[(df["coverage_1"] == 0) & (df["coverage_2"] > 0)]
    # => Klassen die bei semla gar nicht abgedeckt waren, aber bei patch
    # => wie ist da der miou gewinn ?


    ###################################
    ###### Cross Vocab "Heatmap"!! ####
    ###################################

    cross_vocab_weights = ["/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/archive/semla_novelclasses_none/weights.json",
                           "/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/archive/semla_novelclasses_global/weights.json",
                           "/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/archive/semla_novelclasses_patch/weights.json",
                           "/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/archive/semla_novelclasses_object/weights.json"]

    labels = ["None", "Global", "Patch", "Object"]

    specific_object = []

    weights_list = [aggregate_weights_cross_vocab(load_evaluation_data(path)) for path in cross_vocab_weights]

    if len(specific_object) == 0:
        plot_heatmap_cross_voc_dataset_all_objects(weights_list, labels)
    else:
        plot_heatmap_cross_voc_dataset_specific_object(specific_object, weights_list, labels)


    ###################################
    ############ Heatmap!! ############
    ###################################

    weight_path = "/home/joshi/Desktop/HSB/SemLA_VocabularyGuided/results/final_A/none/correlation_log_none.json"
    weight_data = load_evaluation_data(weight_path)
    plot_adapter_contribution_heatmap(weight_data)

if __name__ == "__main__":
    main()