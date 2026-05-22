import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    from matplotlib.lines import Line2D  # type: ignore[import-not-found]
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime environment dependent
    plt = None
    Line2D = None
    FormatStrFormatter = None
    MaxNLocator = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create F1-focused publication figures from tabulated CSV outputs."
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier3/wandb_last_runs_report_tables",
        help="Directory containing tabulated CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for figures (defaults to <tables-dir>/figures).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=("accuracy", "precision", "recall", "f1"),
        help="Metric to visualize.",
    )
    parser.add_argument(
        "--context-sizes",
        type=str,
        default="all",
        help="Optional comma-separated context sizes to include (e.g. '0,100,500').",
    )
    parser.add_argument(
        "--min-task-number",
        type=int,
        default=None,
        help="Optional minimum task number to include.",
    )
    parser.add_argument(
        "--max-task-number",
        type=int,
        default=None,
        help="Optional maximum task number to include.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required table missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def classify_family(task_family: str) -> str:
    lowered = task_family.lower()
    if "codeact" in lowered:
        return "CodeAct"
    if "rlm" in lowered:
        return "RLM"
    return task_family


def short_model(model_name: str) -> str:
    return model_name.split("/")[-1]


def compact_model(model_name: str) -> str:
    name = short_model(model_name)
    replacements = {
        "gpt-5-mini": "gpt5-mini",
        "grok-4-fast": "grok4-fast",
    }
    return replacements.get(name, name)


def compact_family(family: str) -> str:
    if family == "CodeAct":
        return "CA"
    return family


def config_label(row: dict[str, object]) -> str:
    return (
        f"{compact_family(str(row['family']))} | {compact_model(str(row['model_name']))} | "
        f"ctx={row['context_size']}"
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
        }
    )


def save_figure(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def include_row(
    row: dict[str, object],
    allowed_contexts: set[int],
    min_task: Optional[int],
    max_task: Optional[int],
) -> bool:
    context_size = parse_int(row.get("context_size"))
    task_number = parse_int(row.get("task_number"))
    if context_size is None or task_number is None:
        return False
    if allowed_contexts and context_size not in allowed_contexts:
        return False
    if min_task is not None and task_number < min_task:
        return False
    if max_task is not None and task_number > max_task:
        return False
    return True


def load_task_rows(
    tables_dir: Path,
    metric: str,
    allowed_contexts: set[int],
    min_task: Optional[int],
    max_task: Optional[int],
) -> list[dict[str, object]]:
    task_perf_rows = read_csv_rows(tables_dir / "task_averages.csv")
    task_cost_rows = read_csv_rows(tables_dir / "task_cost_averages.csv")

    cost_index: dict[tuple[str, str, int, str, int], float] = {}
    for row in task_cost_rows:
        task_number = parse_int(row.get("task_number"))
        context_size = parse_int(row.get("context_size"))
        if task_number is None or context_size is None:
            continue
        key = (
            str(row.get("project", "")),
            str(row.get("task_family", "")),
            task_number,
            str(row.get("model_name", "")),
            context_size,
        )
        cost = parse_float(row.get("avg_total_cost_per_run_usd"))
        if cost is not None and cost > 0.0:
            cost_index[key] = cost

    merged: list[dict[str, object]] = []
    for row in task_perf_rows:
        if row.get("metric") != metric:
            continue
        if not include_row(row, allowed_contexts, min_task, max_task):
            continue
        task_number = parse_int(row.get("task_number"))
        context_size = parse_int(row.get("context_size"))
        score = parse_float(row.get("task_avg_over_samples"))
        if task_number is None or context_size is None or score is None:
            continue
        key = (
            str(row.get("project", "")),
            str(row.get("task_family", "")),
            task_number,
            str(row.get("model_name", "")),
            context_size,
        )
        cost_usd = cost_index.get(key)
        if cost_usd is None:
            continue
        family = classify_family(str(row.get("task_family", "unknown")))
        merged.append(
            {
                "project": str(row.get("project", "")),
                "task_number": task_number,
                "model_name": str(row.get("model_name", "")),
                "context_size": context_size,
                "family": family,
                "score": score,
                "cost_usd": cost_usd,
            }
        )
    return merged


def plot_f1_heatmap(out_dir: Path, rows: list[dict[str, object]], metric: str) -> None:
    tasks = sorted({int(r["task_number"]) for r in rows})
    configs = sorted(
        {
            (str(r["family"]), str(r["model_name"]), int(r["context_size"]))
            for r in rows
        },
        key=lambda x: (x[0], x[1], x[2]),
    )
    config_to_col = {cfg: i for i, cfg in enumerate(configs)}
    task_to_row = {task: i for i, task in enumerate(tasks)}
    matrix = [[float("nan") for _ in configs] for _ in tasks]

    for row in rows:
        cfg = (str(row["family"]), str(row["model_name"]), int(row["context_size"]))
        matrix[task_to_row[int(row["task_number"])]] [config_to_col[cfg]] = float(row["score"])

    fig, ax = plt.subplots(figsize=(max(9.0, len(configs) * 1.4), 5.0), constrained_layout=True)
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(metric.upper())

    xticklabels = [f"{compact_family(cfg[0])}\n{compact_model(cfg[1])}\nctx={cfg[2]}" for cfg in configs]
    ax.set_xticks(list(range(len(configs))))
    ax.set_xticklabels(xticklabels, rotation=12, ha="right", rotation_mode="anchor")
    ax.set_yticks(list(range(len(tasks))))
    ax.set_yticklabels([f"Task {t}" for t in tasks])
    ax.set_title(f"{metric.upper()} by task and configuration")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Task")
    ax.set_xticks([x - 0.5 for x in range(1, len(configs))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(tasks))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0, alpha=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(tasks)):
        for j in range(len(configs)):
            value = matrix[i][j]
            if not math.isnan(value):
                text_color = "white" if value < 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    save_figure(fig, out_dir, f"{metric}_task_configuration_heatmap")


def plot_task_trends(out_dir: Path, rows: list[dict[str, object]], metric: str) -> None:
    family_colors = {"RLM": "#0072B2", "CodeAct": "#D55E00"}
    model_markers = {"gpt-5-mini": "o", "grok-4-fast": "s"}
    context_linestyles = {0: "-", 100: "--", 500: ":"}

    grouped: dict[tuple[str, str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), str(row["model_name"]), int(row["context_size"]))
        grouped[key].append((int(row["task_number"]), float(row["score"])))

    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    config_handles: list[Line2D] = []
    for (family, model_name, context_size), values in sorted(grouped.items()):
        values_sorted = sorted(values)
        xs = [v[0] for v in values_sorted]
        ys = [v[1] for v in values_sorted]
        line = ax.plot(
            xs,
            ys,
            color=family_colors.get(family, "#444444"),
            marker=model_markers.get(short_model(model_name), "D"),
            linestyle=context_linestyles.get(context_size, "-."),
            linewidth=1.8,
            markersize=5.5,
        )[0]
        config_handles.append(
            Line2D(
                [0],
                [0],
                color=line.get_color(),
                marker=line.get_marker(),
                linestyle=line.get_linestyle(),
                linewidth=1.8,
                markersize=5.5,
                label=f"{compact_family(family)} | {compact_model(model_name)} | ctx={context_size}",
            )
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Task number")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} trend across tasks")
    ax.set_xticks(sorted({int(r['task_number']) for r in rows}))
    ax.legend(
        handles=config_handles,
        title="Configuration",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=2,
        borderaxespad=0.0,
    )
    save_figure(fig, out_dir, f"{metric}_task_trends")


def plot_mean_f1_vs_cost(out_dir: Path, rows: list[dict[str, object]], metric: str) -> None:
    grouped_scores: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    grouped_costs: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["family"]), str(row["model_name"]), int(row["context_size"]))
        grouped_scores[key].append(float(row["score"]))
        grouped_costs[key].append(float(row["cost_usd"]))

    family_colors = {"RLM": "#0072B2", "CodeAct": "#D55E00"}
    context_markers = {0: "o", 100: "s", 500: "^"}
    model_fill = {"gpt-5-mini": "filled", "grok-4-fast": "hollow"}

    fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
    x_values: list[float] = []
    y_values: list[float] = []
    present_contexts: set[int] = set()
    for key in sorted(grouped_scores):
        family, model_name, context_size = key
        scores = grouped_scores[key]
        costs = grouped_costs[key]
        y = mean(scores)
        yerr = pstdev(scores) if len(scores) > 1 else 0.0
        x = mean(costs)
        x_values.append(x)
        y_values.append(y)
        present_contexts.add(context_size)
        marker = context_markers.get(context_size, "D")
        short = short_model(model_name)
        face_color = (
            family_colors.get(family, "#444444")
            if model_fill.get(short, "filled") == "filled"
            else "white"
        )
        edge_color = family_colors.get(family, "#444444")
        ax.errorbar(
            [x],
            [y],
            yerr=[yerr],
            fmt="none",
            ecolor=edge_color,
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=85,
            facecolors=face_color,
            edgecolors=edge_color,
            linewidths=1.4,
            zorder=3,
        )
    if x_values:
        x_min = min(x_values)
        x_max = max(x_values)
        x_pad = max(0.01, (x_max - x_min) * 0.10)
        ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    if MaxNLocator is not None and FormatStrFormatter is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        lower = max(0.0, y_min - 0.12)
        upper = min(1.05, y_max + 0.10)
        if upper - lower < 0.35:
            center = (upper + lower) / 2
            lower = max(0.0, center - 0.18)
            upper = min(1.05, center + 0.18)
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Mean cost per run (USD)")
    ax.set_ylabel(f"Mean {metric.upper()} across tasks")
    ax.set_title(f"Mean {metric.upper()} vs cost (error bars = across-task std)")

    family_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=family,
        )
        for family, color in sorted(family_colors.items())
    ]
    context_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"ctx={ctx}",
        )
        for ctx, marker in sorted(context_markers.items())
        if ctx in present_contexts
    ]
    model_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label="gpt5-mini (filled)",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="grok4-fast (hollow)",
        ),
    ]

    legend1 = ax.legend(
        handles=family_handles,
        title="Family",
        loc="upper left",
        bbox_to_anchor=(0.0, -0.14),
        borderaxespad=0.0,
    )
    ax.add_artist(legend1)
    if context_handles:
        legend2 = ax.legend(
            handles=context_handles,
            title="Context",
            loc="upper right",
            bbox_to_anchor=(1.0, -0.14),
            borderaxespad=0.0,
        )
        ax.add_artist(legend2)
    ax.legend(
        handles=model_handles,
        title="Model fill",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        borderaxespad=0.0,
    )
    save_figure(fig, out_dir, f"{metric}_mean_vs_cost")


def plot_efficiency_bars(out_dir: Path, rows: list[dict[str, object]], metric: str) -> None:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_costs: dict[str, list[float]] = defaultdict(list)
    group_family: dict[str, str] = {}

    for row in rows:
        label = config_label(row)
        grouped_scores[label].append(float(row["score"]))
        grouped_costs[label].append(float(row["cost_usd"]))
        group_family[label] = str(row["family"])

    summary: list[dict[str, object]] = []
    for label in grouped_scores:
        avg_score = mean(grouped_scores[label])
        avg_cost = mean(grouped_costs[label])
        if avg_cost <= 0.0:
            continue
        summary.append(
            {
                "label": label,
                "family": group_family[label],
                "avg_score": avg_score,
                "avg_cost": avg_cost,
                "efficiency": avg_score / avg_cost,
            }
        )
    summary.sort(key=lambda r: float(r["efficiency"]), reverse=True)

    colors = {"RLM": "#0072B2", "CodeAct": "#D55E00"}
    fig, ax = plt.subplots(figsize=(10.8, max(4.6, 0.6 * len(summary))), constrained_layout=True)
    y_pos = list(range(len(summary)))
    ax.barh(
        y_pos,
        [float(r["efficiency"]) for r in summary],
        color=[colors.get(str(r["family"]), "#888888") for r in summary],
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(r["label"]) for r in summary])
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric.upper()} per USD (higher is better)")
    ax.set_title(f"Cost efficiency ranking ({metric.upper()})")

    for idx, row in enumerate(summary):
        ax.text(
            float(row["efficiency"]),
            idx,
            f"  {float(row['avg_score']):.3f} @ ${float(row['avg_cost']):.3f}",
            va="center",
            fontsize=8,
        )

    ax.legend(
        handles=[
            Line2D([0], [0], color=colors["RLM"], lw=6, label="RLM"),
            Line2D([0], [0], color=colors["CodeAct"], lw=6, label="CodeAct"),
        ],
        title="Family",
        loc="lower right",
    )
    save_figure(fig, out_dir, f"{metric}_efficiency_ranking")


def main() -> None:
    args = parse_args()
    if plt is None or Line2D is None:
        raise RuntimeError("matplotlib is not installed. Install it with: pip install matplotlib")

    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir) if args.output_dir else tables_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_contexts: set[int] = set()
    if args.context_sizes.strip().lower() != "all":
        allowed_contexts = {
            int(v.strip())
            for v in args.context_sizes.split(",")
            if v.strip()
        }

    setup_style()
    rows = load_task_rows(
        tables_dir=tables_dir,
        metric=args.metric,
        allowed_contexts=allowed_contexts,
        min_task=args.min_task_number,
        max_task=args.max_task_number,
    )
    if not rows:
        raise RuntimeError("No rows available for plotting after filters.")
    family_counts = defaultdict(int)
    for row in rows:
        family_counts[str(row["family"])] += 1
    print("Loaded rows by family:", dict(sorted(family_counts.items())))

    plot_f1_heatmap(output_dir, rows, args.metric)
    plot_task_trends(output_dir, rows, args.metric)
    plot_mean_f1_vs_cost(output_dir, rows, args.metric)
    plot_efficiency_bars(output_dir, rows, args.metric)

    print(f"Wrote figures to: {output_dir}")
    print(f"  - {output_dir / f'{args.metric}_task_configuration_heatmap.png'}")
    print(f"  - {output_dir / f'{args.metric}_task_trends.png'}")
    print(f"  - {output_dir / f'{args.metric}_mean_vs_cost.png'}")
    print(f"  - {output_dir / f'{args.metric}_efficiency_ranking.png'}")


if __name__ == "__main__":
    main()
