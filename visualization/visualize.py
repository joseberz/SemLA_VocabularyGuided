#!/usr/bin/env python3
# Autor: Joshua Ritter
# Teil der Masterarbeit "Vokabulargeleitete Selektion von LoRA-Adaptern
# mittels CLIP für domänenadaptive Open-Vocabulary-Segmentierung" (2026)
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats


# Konfig

FONT_PATH = "/usr/share/fonts/newcomputermodern/NewCM08-Regular.otf"
OUTPUT_DIR = "./plots"

# Pfade zu den correlation_log.json Dateien
LOG_NONE = "results/final_A/none_FINALMENTE/correlation_log.json"
LOG_NONE_NORM = "results/final_A/none_normalized_FINALMENTE/correlation_log.json"
LOG_PATCH = "results/final_A/patch_FINALMENTE/correlation_log.json"

# Für den Domain-Level-Barplot
DOMAIN_VOCAB_PATH = "visualization/domain_vocab.json"
OFFICIAL_MIOU_PATH = None  # optional, sonst None
TOP_K = 3

# Für die Coverage-Bins
VOCAB_DIR = "catseg/datasets"
MIN_UNION = 1.0
N_BINS = 9

# Vokabular-Dateien pro Adapter
VOCAB_FILES = {
    "mv": "mv65.json", "cs19": "cs19.json", "pc59": "pc59.json",
    "a133": "ade133.json", "coconutL": "coconut.json",
    "nyu": "nyu40.json", "idd": "idd30.json",
}

ADAPTER_TO_VOCAB = {
    "mv": "mv", "cs-normal": "cs19", "pc59": "pc59", "a133": "a133",
    "coconutL": "coconutL", "nyu": "nyu", "idd": "idd", "bdd": "cs19",
    "acdc-night": "cs19", "acdc-rain": "cs19", "acdc-snow": "cs19", "acdc-fog": "cs19",
    "muses-clear-day": "cs19", "muses-clear-night": "cs19",
    "muses-rain-day": "cs19", "muses-rain-night": "cs19",
    "muses-fog-day": "cs19", "muses-fog-night": "cs19",
    "muses-snow-day": "cs19", "muses-snow-night": "cs19",
}

DOMAIN_TO_VOCAB = {
    "mv": "mv", "cs-normal": "cs19", "pc59": "pc59", "a133": "a133",
    "coconutL": "coconutL", "nyu": "nyu", "idd": "idd", "bdd": "cs19",
}

DOMAIN_GROUPS = {
    "acdc-rain": "acdc", "acdc-snow": "acdc", "acdc-fog": "acdc", "acdc-night": "acdc",
    "muses-clear-day": "muses", "muses-clear-night": "muses",
    "muses-rain-day": "muses", "muses-rain-night": "muses",
    "muses-fog-day": "muses", "muses-fog-night": "muses",
    "muses-snow-day": "muses", "muses-snow-night": "muses",
}

ORDER_COLUMNS = ["acdc", "muses", "cs-normal", "bdd", "mv", "idd", "a133", "pc59", "nyu", "coconutL"]

ORDER_COLUMNS_FULL = [
    "acdc-rain", "acdc-snow", "acdc-fog", "acdc-night",
    "muses-clear-day", "muses-clear-night", "muses-rain-day", "muses-rain-night",
    "muses-fog-day", "muses-fog-night", "muses-snow-day", "muses-snow-night",
    "cs-normal", "bdd", "mv", "idd",
    "a133", "pc59", "nyu", "coconutL",
]


# Hilfsfunktionen

def setup_style():
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "axes.grid": False, "grid.alpha": 0.3})
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_log_indexed(path):
    data = load_json(path)
    return {d["image_path"]: d for d in data}


def norm(name):
    return name.strip().lower()


def normalize_class_name(name):
    return set(tok.strip().lower() for tok in name.split(",") if tok.strip())


def in_vocab(class_name, vocab):
    for part in norm(class_name).split(","):
        if part.strip() in vocab:
            return True
    return False


# Heatmaps

def plot_heatmap(weights_data, method_name, aggregated=False,
                 threshold=0.05, annot_fs=14, label_fs=15, title_fs=16, tick_fs=14):

    rows = []
    for entry in weights_data:
        if aggregated:
            domain = DOMAIN_GROUPS.get(entry["domain"], entry["domain"])
        else:
            domain = entry["domain"]
        for adapter, weight in entry["adapter_weights"].items():
            if aggregated:
                adapter = DOMAIN_GROUPS.get(adapter, adapter)
            rows.append({"domain": domain, "adapter": adapter, "weight": weight})

    df = pd.DataFrame(rows)
    order = ORDER_COLUMNS if aggregated else ORDER_COLUMNS_FULL

    matrix = df.pivot_table(index="domain", columns="adapter", values="weight",
                            aggfunc="mean", fill_value=0)
    rows_order = [d for d in order if d in matrix.index]
    cols_order = [d for d in order if d in matrix.columns]
    matrix = matrix.reindex(index=rows_order, columns=cols_order)

    display_matrix = matrix.mask(matrix < threshold)

    if aggregated:
        figsize = (max(6, len(matrix.columns) * 0.6), max(5, len(matrix.index) * 0.5))
    else:
        figsize = (max(10, len(matrix.columns) * 0.7), max(8, len(matrix.index) * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        display_matrix,
        annot=True, fmt=".1f",
        annot_kws={"size": annot_fs},
        cmap="Blues",
        vmin=0, vmax=display_matrix.max().max() * 0.7,
        linewidths=0.5, linecolor="white",
        mask=display_matrix.isna(),
        ax=ax, square=True, cbar=False,
    )

    ax.set_xlabel("LoRA-Adapter", fontsize=label_fs)
    ax.set_ylabel("Test-Datensatz", fontsize=label_fs)
    ax.set_title(f"Adapter-Gewichtung für\n{method_name}", fontsize=title_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    if not aggregated:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    prefix = "heatmap" if aggregated else "heatmap_full"
    out_path = os.path.join(OUTPUT_DIR, f"{prefix}_{method_name}.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_heatmaps():
    data_none = load_json(LOG_NONE)
    data_none_norm = load_json(LOG_NONE_NORM)
    data_patch = load_json(LOG_PATCH)

    fs = dict(annot_fs=20, label_fs=24, title_fs=26, tick_fs=20)

    # Volle Heatmaps (alle Subdomänen)
    plot_heatmap(data_none, "(a) SemLA (unnorm.)", aggregated=False, **fs)
    plot_heatmap(data_none_norm, "(b) SemLA (norm.)", aggregated=False, **fs)
    plot_heatmap(data_patch, "(c) Patch", aggregated=False, **fs)

    # Aggregierte Heatmaps (ACDC/MUSES zusammengefasst)
    plot_heatmap(data_none, "(a) SemLA (unnorm.)", aggregated=True)
    plot_heatmap(data_none_norm, "(b) SemLA (norm.)", aggregated=True)
    plot_heatmap(data_patch, "(c) Patch", aggregated=True)


# Domain-Level Barplot / Mismatch-Reduktion vs. mIoU-Gewinn

def get_top_k_adapters(weight_dict, k):
    sorted_adapters = sorted(weight_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in sorted_adapters[:k]]


def load_domain_vocab(path):
    raw = load_json(path)
    vocab = {}
    for domain, classes in raw.items():
        tokens = set()
        for c in classes:
            tokens |= normalize_class_name(c)
        vocab[domain] = tokens
    return vocab


def collect_class_level_stats(none_index, patch_index):
    stats_dict = defaultdict(lambda: {
        "intersection_none": 0, "union_none": 0,
        "intersection_patch": 0, "union_patch": 0,
        "n_images": 0,
    })

    common_paths = set(none_index.keys()) & set(patch_index.keys())
    for path in common_paths:
        none_entry = none_index[path]
        patch_entry = patch_index[path]
        domain = none_entry["domain"]

        none_pc = none_entry.get("per_class_miou", {})
        patch_pc = patch_entry.get("per_class_miou", {})
        all_classes = set(none_pc.keys()) | set(patch_pc.keys())

        for cls in all_classes:
            s = stats_dict[(domain, cls)]
            s["n_images"] += 1
            if cls in none_pc:
                s["intersection_none"] += none_pc[cls]["intersection"]
                s["union_none"] += none_pc[cls]["union"]
            if cls in patch_pc:
                s["intersection_patch"] += patch_pc[cls]["intersection"]
                s["union_patch"] += patch_pc[cls]["union"]

    return stats_dict


def collect_class_level_coverage(none_index, patch_index, domain_vocab, top_k):
    cov = defaultdict(lambda: {"n_images": 0, "n_covered_baseline": 0, "n_covered_patch": 0})

    common_paths = set(none_index.keys()) & set(patch_index.keys())
    for path in common_paths:
        none_entry = none_index[path]
        patch_entry = patch_index[path]
        domain = none_entry["domain"]

        baseline_top_k = get_top_k_adapters(none_entry["adapter_weights"], top_k)
        patch_top_k = get_top_k_adapters(patch_entry["adapter_weights"], top_k)

        baseline_vocab = set()
        for a in baseline_top_k:
            baseline_vocab |= domain_vocab.get(a, set())
        patch_vocab = set()
        for a in patch_top_k:
            patch_vocab |= domain_vocab.get(a, set())

        gt_classes = list(none_entry.get("per_class_miou", {}).keys())
        for cls in gt_classes:
            tokens = normalize_class_name(cls)
            c = cov[(domain, cls)]
            c["n_images"] += 1
            if tokens & baseline_vocab:
                c["n_covered_baseline"] += 1
            if tokens & patch_vocab:
                c["n_covered_patch"] += 1

    return cov


def build_domain_table(none_index, patch_index, domain_vocab, top_k, official_miou=None):
    miou_stats = collect_class_level_stats(none_index, patch_index)
    cov_stats = collect_class_level_coverage(none_index, patch_index, domain_vocab, top_k)

    # Klassen-Ebene aufbauen
    class_rows = []
    for key, s in miou_stats.items():
        domain, cls = key
        miou_none = s["intersection_none"] / s["union_none"] if s["union_none"] > 0 else np.nan
        miou_patch = s["intersection_patch"] / s["union_patch"] if s["union_patch"] > 0 else np.nan
        gain = (miou_patch - miou_none) if not (np.isnan(miou_none) or np.isnan(miou_patch)) else np.nan

        c = cov_stats.get(key, {"n_images": 0, "n_covered_baseline": 0, "n_covered_patch": 0})
        n_img = c["n_images"]
        cov_baseline = c["n_covered_baseline"] / n_img if n_img > 0 else np.nan
        cov_patch = c["n_covered_patch"] / n_img if n_img > 0 else np.nan
        cov_improvement = (cov_patch - cov_baseline) if n_img > 0 else np.nan

        class_rows.append({
            "domain": domain, "class_name": cls,
            "miou_none": miou_none, "miou_patch": miou_patch, "miou_gain": gain,
            "coverage_baseline": cov_baseline, "coverage_patch": cov_patch,
            "coverage_improvement": cov_improvement,
        })

    class_df = pd.DataFrame(class_rows)

    # Bilder pro Domäne zählen
    common_paths = set(none_index.keys()) & set(patch_index.keys())
    image_counts = defaultdict(int)
    for path in common_paths:
        image_counts[none_index[path]["domain"]] += 1

    # Domänen-Ebene aggregieren
    domain_rows = []
    for domain, sub in class_df.groupby("domain"):
        if official_miou is not None and domain in official_miou:
            miou_none_agg = official_miou[domain]["none"]
            miou_patch_agg = official_miou[domain]["patch"]
        else:
            valid = sub.dropna(subset=["miou_none", "miou_patch"])
            miou_none_agg = valid["miou_none"].mean() * 100 if len(valid) else np.nan
            miou_patch_agg = valid["miou_patch"].mean() * 100 if len(valid) else np.nan

        miou_gain_agg = miou_patch_agg - miou_none_agg

        cov_valid = sub.dropna(subset=["coverage_baseline", "coverage_patch"])
        reduction_macro = cov_valid["coverage_improvement"].mean() if len(cov_valid) else np.nan

        domain_rows.append({
            "domain": domain,
            "n_images": image_counts.get(domain, 0),
            "miou_none_agg": miou_none_agg,
            "miou_patch_agg": miou_patch_agg,
            "miou_gain_agg": miou_gain_agg,
            "mismatch_reduction_macro": reduction_macro,
        })

    return pd.DataFrame(domain_rows).sort_values("miou_gain_agg", ascending=False)


def plot_domain_bar(domain_df):
    plt.rcParams.update({
        "font.size": 13.5, "axes.titlesize": 17, "axes.labelsize": 15,
        "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 13,
    })

    df = domain_df.sort_values("mismatch_reduction_macro", ascending=False)
    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(df))
    width = 0.4

    b1 = ax1.bar(x - width / 2, df["mismatch_reduction_macro"], width,
                 color="#9467bd", label="Mismatch-Reduktion")
    ax1.set_ylabel("Mismatch-Reduktion", color="#9467bd", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#9467bd")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width / 2, df["miou_gain_agg"], width,
                 color="#1f77b4", label="mIoU-Gewinn")
    ax2.set_ylabel("mIoU-Gewinn", color="#1f77b4", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["domain"], rotation=45, ha="right", fontsize=12)

    fig.suptitle("Mismatch-Reduktion in Relation zur mIoU-Steigerung",
                 fontweight="bold", y=0.95, fontsize=16)
    fig.legend(handles=[b1, b2], loc="upper center", bbox_to_anchor=(0.5, 0.87),
               ncol=2, frameon=False, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(OUTPUT_DIR, "domain_level_bar.pdf")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Gespeichert: unter {out_path}")


def run_domain_bar():
    none_index = load_log_indexed(LOG_NONE)
    patch_index = load_log_indexed(LOG_PATCH)
    domain_vocab = load_domain_vocab(DOMAIN_VOCAB_PATH)

    official_miou = None
    if OFFICIAL_MIOU_PATH:
        official_miou = load_json(OFFICIAL_MIOU_PATH)

    domain_df = build_domain_table(none_index, patch_index, domain_vocab, TOP_K, official_miou)

    csv_path = os.path.join(OUTPUT_DIR, "domain_level.csv")
    domain_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print(domain_df.to_string(index=False))

    plot_domain_bar(domain_df)


# Coverage-Bins

def load_vocabs():
    raw = {}
    for key, fname in VOCAB_FILES.items():
        path = Path(VOCAB_DIR) / fname
        if path.exists():
            raw[key] = {norm(c) for c in json.loads(path.read_text())}
        else:
            print(f"{path} fehlt")

    result = {}
    for adapter, vocab_key in ADAPTER_TO_VOCAB.items():
        if vocab_key in raw:
            result[adapter] = raw[vocab_key]
    return result


def coverage(class_name, weights, vocabs):
    total = 0
    for adapter, weight in weights.items():
        if adapter in vocabs and in_vocab(class_name, vocabs[adapter]):
            total += weight
    return total


def build_pairs(data_a, data_b, vocabs):
    by_path_b = {r["image_path"]: r for r in data_b}
    rows = []

    for a in data_a:
        b = by_path_b.get(a["image_path"])
        if b is None:
            continue

        cache_a = {}
        cache_b = {}
        for cls, va in a["per_class_miou"].items():
            vb = b["per_class_miou"].get(cls)
            if vb is None or max(va["union"], vb["union"]) < MIN_UNION:
                continue

            if cls not in cache_a:
                cache_a[cls] = coverage(cls, a["adapter_weights"], vocabs)
            if cls not in cache_b:
                cache_b[cls] = coverage(cls, b["adapter_weights"], vocabs)

            rows.append({
                "image_path": a["image_path"],
                "domain": a["domain"],
                "class_name": norm(cls),
                "cov_none": cache_a[cls],
                "cov_patch": cache_b[cls],
                "miou_none": va["miou"] * 100,
                "miou_patch": vb["miou"] * 100,
                "inter_none": va["intersection"],
                "union_none": va["union"],
                "inter_patch": vb["intersection"],
                "union_patch": vb["union"],
            })

    df = pd.DataFrame(rows)
    df["delta_cov"] = df.cov_patch - df.cov_none
    df["delta_miou"] = df.miou_patch - df.miou_none
    return df


def pooled_class_deltas(sub):
    # Pro Klasse: Intersection/Union aufsummieren, dann erst Verhältnis bilden
    g = sub.groupby("class_name").agg(
        inter_none=("inter_none", "sum"), union_none=("union_none", "sum"),
        inter_patch=("inter_patch", "sum"), union_patch=("union_patch", "sum"),
        n_rows=("class_name", "size"),
    )
    g = g[(g.union_none > 0) & (g.union_patch > 0)]
    g["miou_none"] = g.inter_none / g.union_none * 100
    g["miou_patch"] = g.inter_patch / g.union_patch * 100
    g["delta"] = g.miou_patch - g.miou_none
    return g


def plot_coverage_bins(df, suffix=""):
    changed = df[df.delta_cov.abs() > 1e-6].copy()
    if len(changed) < 100:
        print("Zu wenige Zeilen mit Coverage-Änderung")
        return None

    r_p, p_p = stats.pearsonr(changed.delta_cov, changed.delta_miou)
    r_s, p_s = stats.spearmanr(changed.delta_cov, changed.delta_miou)

    changed["bin"] = pd.qcut(changed.delta_cov, q=N_BINS, duplicates="drop")

    agg = []
    for b, sub in changed.groupby("bin", observed=True):
        class_delta = pooled_class_deltas(sub)["delta"]
        agg.append({
            "bin": b,
            "center": sub.delta_cov.mean(),
            "mean": class_delta.mean(),
            "n": len(sub),
            "n_classes": class_delta.shape[0],
        })
    agg = pd.DataFrame(agg)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    xs = np.arange(len(agg))

    colors = []
    for m in agg["mean"]:
        if m < 0:
            colors.append("#e34948")
        else:
            colors.append("#1baf7a")

    ax.bar(xs, agg["mean"], width=0.68, color=colors, alpha=0.9)
    ax.axhline(0, color="0.4", linewidth=0.9, linestyle="--")
    ax.set_xticks(xs)
    ax.tick_params(axis="y", labelsize=12)

    labels = []
    for b in agg["bin"]:
        labels.append(f"{b.left:+.2f}\n{b.right:+.2f}")
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_xlabel(r"$\Delta$ Vokabular-Coverage (Patch $-$ Baseline)", fontsize=14)
    ax.set_ylabel(r"$\Delta$ Klassen-IoU", fontsize=14)

    fig.suptitle("Effekt der Coverage-Änderung auf den Klassen-IoU-Gewinn nach Bins",
                 y=0.98, fontsize=16)
    fig.tight_layout()

    filename = f"coverage_bins{suffix}.pdf"
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")

    result = {
        "pearson_r": r_p, "pearson_p": p_p,
        "spearman_r": r_s, "spearman_p": p_s,
        "n": len(changed),
    }
    print(result)
    return result


def run_coverage_bins():
    data_none = load_json(LOG_NONE)
    data_patch = load_json(LOG_PATCH)
    vocabs = load_vocabs()

    print(f"Methode None Einträge: {len(data_none)} / Methode Patch Einträge: {len(data_patch)}")

    df = build_pairs(data_none, data_patch, vocabs)
    if df.empty:
        print("Keine gemeinsamen (Bild, Klasse)-Paare gefunden.")
        return

    changed_rows = (df.delta_cov.abs() > 1e-6).sum()
    print(f"{len(df)} gepaarte Beobachtungen, {df.class_name.nunique()} Klassen,"
          f"{df.domain.nunique()} Domänen")
    print(f"Veränderter Coverage unter diesen beobachtungen: {changed_rows}")

    # Gesamtplot
    plot_coverage_bins(df)

    # Pro Domäne
    for domain_name in sorted(df.domain.unique()):
        domain_df = df[df.domain == domain_name]
        changed = (domain_df.delta_cov.abs() > 1e-6).sum()
        if changed < 50:
            print(f"  {domain_name}: zu wenige Zeilen ({changed}), überspringe")
            continue
        print(f"\n{domain_name}")
        plot_coverage_bins(domain_df, suffix=f"-{domain_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", required=True,
                        choices=["heatmaps", "domain-bar", "coverage-bins"],
                        help="Welcher Plot erzeugt werden soll")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    if args.plot == "heatmaps":
        run_heatmaps()
    elif args.plot == "domain-bar":
        run_domain_bar()
    elif args.plot == "coverage-bins":
        run_coverage_bins()