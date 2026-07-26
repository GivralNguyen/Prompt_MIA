import argparse
import math
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


BATCH_ORDER = [1, 4, 8, 16, 32, 64, 128, 256]


def infer_dataset(model_path: str) -> str:
    lower = model_path.lower()
    if "cifar100" in lower:
        return "cifar100"
    if "cifar10" in lower:
        return "cifar10"
    return "unknown"


def infer_defense(model_path: str) -> str:
    lower = model_path.lower()
    if "defsparse" in lower:
        return "Gradient Sparse (80%)"
    if "defmix_up" in lower:
        return "Mixup ( alpha = 0.2)"
    if "defnone" in lower and "aug1" in lower:
        return "DataAug"
    if "defnone" in lower and "aug0" in lower:
        return "No Defense"
    return "Other"


def parse_log(
    log_path: str,
    forced_dataset: str | None = None,
    forced_defense: str | None = None,
):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    records = []
    current_model = None
    current_loop = None

    for line in lines:
        line = line.strip()

        m = re.search(r"Model loaded from\s+(.*)$", line)
        if m:
            current_model = m.group(1).strip()
            current_loop = None
            continue

        m = re.search(r"MIA Attack Loop\s+(\d+)", line)
        if m:
            current_loop = int(m.group(1))
            continue

        m = re.search(
            r"mia batch size:\s*(\d+).*?Attack Accuracy:\)\s*([0-9.]+)\s*%",
            line,
        )
        if m and current_model is not None and current_loop is not None:
            batch_size = int(m.group(1))
            asr = float(m.group(2))
            dataset = forced_dataset if forced_dataset is not None else infer_dataset(current_model)
            defense = forced_defense if forced_defense is not None else infer_defense(current_model)
            records.append((dataset, defense, current_model, current_loop, batch_size, asr))

    return records


def compute_stats(records, ci95: bool = True):
    # Collect per-loop values so interrupted loops only drop missing points.
    grouped = defaultdict(lambda: defaultdict(list))
    # key: (dataset, defense), inner key: batch_size -> list[asr]
    for dataset, defense, _model, _loop, batch, asr in records:
        grouped[(dataset, defense)][batch].append(asr)

    stats = {}
    for key, batch_dict in grouped.items():
        xs = []
        means = []
        lows = []
        highs = []
        ns = []

        for b in BATCH_ORDER:
            vals = batch_dict.get(b, [])
            if not vals:
                continue

            arr = np.array(vals, dtype=np.float64)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            if ci95 and len(arr) > 1:
                half = 1.96 * std / math.sqrt(len(arr))
            else:
                half = std

            low = max(0.0, mean - half)
            high = min(100.0, mean + half)

            xs.append(b)
            means.append(mean)
            lows.append(low)
            highs.append(high)
            ns.append(len(arr))

        stats[key] = {
            "x": np.array(xs, dtype=np.int64),
            "mean": np.array(means, dtype=np.float64),
            "low": np.array(lows, dtype=np.float64),
            "high": np.array(highs, dtype=np.float64),
            "n": np.array(ns, dtype=np.int64),
        }

    return stats


def plot(stats, output_path: str, ci95: bool):
    color_map = {
        "No Defense": "#1f77b4",
        "Gradient Sparse (80%)": "#ff7f0e",
        "Mixup ( alpha = 0.2)": "#2ca02c",
        "DataAug": "#d62728",
        "Other": "#7f7f7f",
    }
    marker_map = {
        "No Defense": "o",
        "Gradient Sparse (80%)": "s",
        "Mixup ( alpha = 0.2)": "^",
        "DataAug": "d",
        "Other": "x",
    }
    linestyle_map = {
        "No Defense": "--",
        "Gradient Sparse (80%)": "--",
        "Mixup ( alpha = 0.2)": "-.",
        "DataAug": ":",
        "Other": "-",
    }

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    batch_to_pos = {b: i for i, b in enumerate(BATCH_ORDER)}
    x_tick_pos = np.arange(len(BATCH_ORDER))
    datasets = ["cifar10", "cifar100"]
    titles = {
        "cifar10": "CIFAR-10 PromptMIA ASR",
        "cifar100": "CIFAR-100 PromptMIA ASR",
    }

    for ax, ds in zip(axes, datasets):
        for defense in [
            "Gradient Sparse (80%)",
            "Mixup ( alpha = 0.2)",
            "DataAug",
            "Other",
            "No Defense",
        ]:
            key = (ds, defense)
            if key not in stats:
                continue

            d = stats[key]
            label = f"{ds.upper()} {defense}"
            color = color_map.get(defense, "#7f7f7f")
            x_pos = np.array([batch_to_pos[int(b)] for b in d["x"]], dtype=np.float64)
            band_low = d["low"].copy()
            band_high = d["high"].copy()

            if defense == "No Defense":
                # Visualization-only floor: CI can be sub-pixel at this y-range.
                min_half = 0.15
                band_low = np.maximum(0.0, np.minimum(band_low, d["mean"] - min_half))
                band_high = np.minimum(100.0, np.maximum(band_high, d["mean"] + min_half))

            ax.plot(
                x_pos,
                d["mean"],
                marker=marker_map.get(defense, "o"),
                linestyle=linestyle_map.get(defense, "-"),
                color=color,
                linewidth=1.5,
                markersize=5,
                label=label,
                zorder=3,
            )
            ax.fill_between(
                x_pos,
                band_low,
                band_high,
                color=color,
                alpha=0.28 if defense == "No Defense" else 0.16,
                zorder=2,
            )
            if defense == "No Defense":
                ax.plot(x_pos, band_low, color=color, linewidth=0.9, alpha=0.55, zorder=2.2)
                ax.plot(x_pos, band_high, color=color, linewidth=0.9, alpha=0.55, zorder=2.2)

        ax.set_title(titles[ds])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Attack Success Rate (%)")
        ax.set_xticks(x_tick_pos)
        ax.set_xticklabels([str(b) for b in BATCH_ORDER])
        ax.set_xlim(-0.35, len(BATCH_ORDER) - 0.65)
        ax.set_ylim(50, 101)
        ax.grid(True, alpha=0.45)
        ax.legend(loc="lower left")

    band_name = "95% CI" if ci95 else "std"
    fig.suptitle("PromptMIA ASR with Uncertainty Band")
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot defenses ASR with uncertainty bands.")
    parser.add_argument(
        "--log_path",
        type=str,
        default="/home/quannm/code/icml_promptmia/result_log/defenses.txt",
        help="Path to defenses.txt log file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/quannm/code/icml_promptmia/result_log/defenses_with_uncertainty.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="ci95",
        choices=["ci95", "std"],
        help="Type of uncertainty band.",
    )
    parser.add_argument(
        "--cifar10_no_defense_log",
        type=str,
        default="/home/quannm/code/icml_promptmia/result_log/cifar10_vit_stealth.txt",
        help="Optional CIFAR-10 no-defense log to overlay.",
    )
    parser.add_argument(
        "--cifar100_no_defense_log",
        type=str,
        default="/home/quannm/code/icml_promptmia/result_log/cifar100_vit_stealth.txt",
        help="Optional CIFAR-100 no-defense log to overlay.",
    )
    args = parser.parse_args()

    records = parse_log(args.log_path)
    records.extend(
        parse_log(
            args.cifar10_no_defense_log,
            forced_dataset="cifar10",
            forced_defense="No Defense",
        )
    )
    records.extend(
        parse_log(
            args.cifar100_no_defense_log,
            forced_dataset="cifar100",
            forced_defense="No Defense",
        )
    )
    if not records:
        raise RuntimeError("No parsable records found in log file.")

    stats = compute_stats(records, ci95=(args.band == "ci95"))
    plot(stats, args.output_path, ci95=(args.band == "ci95"))


if __name__ == "__main__":
    main()