import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - runtime environment dependent
    plt = None

try:
    import wandb
except ImportError:  # pragma: no cover - runtime environment dependent
    wandb = None


METRICS = ("precision", "recall", "f1")
SAMPLE_METRIC_PATTERN = re.compile(r"^sample/(\d+)/(precision|recall|f1)$")
PROJECT_PATTERN = re.compile(r'project\s*=\s*"([^"]+)"')


@dataclass
class RunSamples:
    project: str
    run_id: str
    run_name: str
    created_at: str
    sample_metrics: dict[int, dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load last N W&B runs for each tier3 task project and plot "
            "sample/{i}/precision|recall|f1 metrics."
        )
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="W&B entity/username that owns the projects.",
    )
    parser.add_argument(
        "--tier3-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier3",
        help="Directory containing rlm_task*.py files.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=5,
        help="Number of latest runs to pull per project.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier3/wandb_last_runs_report",
        help="Directory where plots and CSV summaries will be written.",
    )
    return parser.parse_args()


def discover_task_projects(tier3_dir: Path) -> list[str]:
    projects: set[str] = set()
    for path in sorted(tier3_dir.glob("rlm_task*.py")):
        text = path.read_text(encoding="utf-8")
        for match in PROJECT_PATTERN.finditer(text):
            projects.add(match.group(1))
    def task_key(name: str) -> tuple[int, str]:
        m = re.search(r"Task(\d+)$", name)
        if m:
            return (int(m.group(1)), name)
        return (10**9, name)
    return sorted(projects, key=task_key)


def extract_sample_metrics(summary: dict) -> dict[int, dict[str, float]]:
    per_sample: dict[int, dict[str, float]] = {}
    for key, value in summary.items():
        m = SAMPLE_METRIC_PATTERN.match(key)
        if not m:
            continue
        if not isinstance(value, (int, float)):
            continue
        sample_i = int(m.group(1))
        metric = m.group(2)
        per_sample.setdefault(sample_i, {})[metric] = float(value)
    return per_sample


def fetch_last_runs(entity: str, project: str, last_n: int) -> list[RunSamples]:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    result: list[RunSamples] = []
    for run in runs:
        summary = dict(run.summary)
        sample_metrics = extract_sample_metrics(summary)
        if not sample_metrics:
            continue
        result.append(
            RunSamples(
                project=project,
                run_id=run.id,
                run_name=run.name or run.id,
                created_at=run.created_at or "",
                sample_metrics=sample_metrics,
            )
        )
        if len(result) >= last_n:
            break
    return result


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def write_project_summary_csv(project_dir: Path, runs: list[RunSamples]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for metric in METRICS:
            values = [
                metrics[metric]
                for metrics in run.sample_metrics.values()
                if metric in metrics and isinstance(metrics[metric], (int, float))
            ]
            row = {
                "project": run.project,
                "run_id": run.run_id,
                "run_name": run.run_name,
                "created_at": run.created_at,
                "metric": metric,
                "sample_count": len(values),
                "mean": safe_mean(values),
                "std": safe_std(values),
                "min": min(values) if values else float("nan"),
                "max": max(values) if values else float("nan"),
            }
            rows.append(row)

    out_csv = project_dir / "run_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "project",
                "run_id",
                "run_name",
                "created_at",
                "metric",
                "sample_count",
                "mean",
                "std",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_sample_rows(project: str, runs: list[RunSamples]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for sample_id, sample_metrics in run.sample_metrics.items():
            for metric in METRICS:
                value = sample_metrics.get(metric)
                if not isinstance(value, (int, float)):
                    continue
                rows.append(
                    {
                        "project": project,
                        "run_id": run.run_id,
                        "run_name": run.run_name,
                        "created_at": run.created_at,
                        "sample_id": sample_id,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def write_project_sample_points_csv(project_dir: Path, rows: list[dict[str, object]]) -> None:
    out_csv = project_dir / "sample_points.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "project",
                "run_id",
                "run_name",
                "created_at",
                "sample_id",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_project_samples(project_dir: Path, project: str, runs: list[RunSamples]) -> None:
    sample_ids = sorted({i for run in runs for i in run.sample_metrics})
    if not sample_ids:
        return

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(sample_ids) * 0.5), 12), sharex=True)
    bar_group_width = 0.8
    bar_width = bar_group_width / max(1, len(runs))
    x = list(range(len(sample_ids)))

    for metric_idx, metric in enumerate(METRICS):
        ax = axes[metric_idx]
        for run_idx, run in enumerate(runs):
            offset = -bar_group_width / 2 + (run_idx + 0.5) * bar_width
            values = []
            for sample_id in sample_ids:
                value = run.sample_metrics.get(sample_id, {}).get(metric, float("nan"))
                values.append(value)
            xs = [v + offset for v in x]
            label = f"{run.run_name} ({run.run_id[:8]})"
            ax.bar(xs, values, width=bar_width, label=label, alpha=0.85)

        ax.set_ylabel(metric.title())
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_title(f"{project}: sample/{'{i}'}/{metric}")

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(i) for i in sample_ids], rotation=90)
    axes[-1].set_xlabel("Sample index i")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(project_dir / "sample_metrics_bar.png", dpi=200)
    plt.close(fig)


def plot_project_strip(project_dir: Path, project: str, runs: list[RunSamples]) -> None:
    sample_ids = sorted({i for run in runs for i in run.sample_metrics})
    if not sample_ids:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(sample_ids) * 0.6), 6))
    metric_colors = {"precision": "#1f77b4", "recall": "#ff7f0e", "f1": "#2ca02c"}
    metric_offsets = {"precision": -0.24, "recall": 0.0, "f1": 0.24}

    for metric in METRICS:
        xs: list[float] = []
        ys: list[float] = []
        for sample_pos, sample_id in enumerate(sample_ids):
            for run_idx, run in enumerate(runs):
                value = run.sample_metrics.get(sample_id, {}).get(metric)
                if not isinstance(value, (int, float)):
                    continue

                center_x = sample_pos + metric_offsets[metric]
                if len(runs) > 1:
                    spread = 0.18
                    jitter = -spread / 2 + run_idx * (spread / (len(runs) - 1))
                else:
                    jitter = 0.0
                xs.append(center_x + jitter)
                ys.append(float(value))

        if xs:
            ax.scatter(
                xs,
                ys,
                s=45,
                alpha=0.85,
                color=metric_colors[metric],
                label=metric.title(),
                edgecolors="none",
            )

    ax.set_xticks(list(range(len(sample_ids))))
    ax.set_xticklabels([str(i) for i in sample_ids], rotation=90)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Question / sample index i")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{project}: strip plot of last {len(runs)} runs by question and metric")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Metric", loc="upper right")
    fig.tight_layout()
    fig.savefig(project_dir / "sample_metrics_strip.png", dpi=200)
    plt.close(fig)


def plot_global_strip(output_dir: Path, sample_rows: list[dict[str, object]], last_n: int) -> None:
    projects = sorted({str(r["project"]) for r in sample_rows if "project" in r})
    if not projects:
        return

    ncols = min(2, len(projects))
    nrows = math.ceil(len(projects) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(12, 8 * ncols), max(4, 3.8 * nrows)),
        sharey=True,
    )
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    else:
        axes_list = [axes]

    metric_colors = {"precision": "#1f77b4", "recall": "#ff7f0e", "f1": "#2ca02c"}
    metric_offsets = {"precision": -0.24, "recall": 0.0, "f1": 0.24}

    for ax_idx, project in enumerate(projects):
        ax = axes_list[ax_idx]
        project_rows = [r for r in sample_rows if r.get("project") == project]
        sample_ids = sorted(
            {
                int(r["sample_id"])
                for r in project_rows
                if isinstance(r.get("sample_id"), int)
            }
        )
        sample_pos_by_id = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        run_ids = sorted({str(r["run_id"]) for r in project_rows if "run_id" in r})
        run_index = {run_id: idx for idx, run_id in enumerate(run_ids)}

        for metric in METRICS:
            xs: list[float] = []
            ys: list[float] = []
            for row in project_rows:
                if row.get("metric") != metric:
                    continue
                sample_id = row.get("sample_id")
                value = row.get("value")
                run_id = row.get("run_id")
                if not isinstance(sample_id, int) or not isinstance(value, float):
                    continue

                sample_pos = sample_pos_by_id.get(sample_id)
                if sample_pos is None:
                    continue
                center_x = sample_pos + metric_offsets[metric]
                if len(run_ids) > 1 and isinstance(run_id, str):
                    spread = 0.18
                    idx = run_index.get(run_id, 0)
                    jitter = -spread / 2 + idx * (spread / (len(run_ids) - 1))
                else:
                    jitter = 0.0
                xs.append(center_x + jitter)
                ys.append(value)

            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=35,
                    alpha=0.85,
                    color=metric_colors[metric],
                    label=metric.title(),
                    edgecolors="none",
                )

        ax.set_title(project)
        ax.set_xticks(list(range(len(sample_ids))))
        ax.set_xticklabels([str(i) for i in sample_ids], rotation=90)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes_list[len(projects):]:
        ax.set_visible(False)

    for row_idx in range(nrows):
        axes_list[row_idx * ncols].set_ylabel("Metric value")
    for col_idx in range(ncols):
        ax = axes_list[(nrows - 1) * ncols + col_idx]
        if ax.get_visible():
            ax.set_xlabel("Question / sample index i")

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Metric", loc="upper right")
    fig.suptitle(
        f"All projects: strip plots of last {last_n} runs by question and metric",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "all_projects_strip.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_global_summary_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    out_csv = output_dir / "all_projects_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "project",
                "run_id",
                "run_name",
                "created_at",
                "metric",
                "sample_count",
                "mean",
                "std",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_global_sample_points_csv(output_dir: Path, rows: list[dict[str, object]]) -> None:
    out_csv = output_dir / "all_sample_points.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "project",
                "run_id",
                "run_name",
                "created_at",
                "sample_id",
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "value_count": 0.0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "value_count": float(len(values)),
        "mean": safe_mean(values),
        "std": safe_std(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def write_global_statistics_csv(output_dir: Path, sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    stats_rows: list[dict[str, object]] = []

    all_values = [r["value"] for r in sample_rows if isinstance(r.get("value"), float)]
    overall = summarize_values(all_values)
    stats_rows.append(
        {
            "level": "overall",
            "project": "all",
            "metric": "all",
            **overall,
        }
    )

    for metric in METRICS:
        values = [
            r["value"]
            for r in sample_rows
            if r.get("metric") == metric and isinstance(r.get("value"), float)
        ]
        stats = summarize_values(values)
        stats_rows.append(
            {
                "level": "metric",
                "project": "all",
                "metric": metric,
                **stats,
            }
        )

    projects = sorted({str(r["project"]) for r in sample_rows if "project" in r})
    for project in projects:
        for metric in METRICS:
            values = [
                r["value"]
                for r in sample_rows
                if r.get("project") == project
                and r.get("metric") == metric
                and isinstance(r.get("value"), float)
            ]
            stats = summarize_values(values)
            stats_rows.append(
                {
                    "level": "project_metric",
                    "project": project,
                    "metric": metric,
                    **stats,
                }
            )

    out_csv = output_dir / "global_statistics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "level",
                "project",
                "metric",
                "value_count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerows(stats_rows)
    return stats_rows


def fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()

    if wandb is None:
        raise RuntimeError(
            "wandb is not installed. Install it with: pip install wandb"
        )
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        )
    tier3_dir = Path(args.tier3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = discover_task_projects(tier3_dir)
    if not projects:
        raise RuntimeError(f"No W&B projects discovered from {tier3_dir}/rlm_task*.py")

    print(f"Discovered projects: {', '.join(projects)}")
    all_rows: list[dict[str, object]] = []
    all_sample_rows: list[dict[str, object]] = []

    for project in projects:
        runs = fetch_last_runs(args.entity, project, args.last_n)
        if not runs:
            print(f"[{project}] No runs with sample precision/recall/f1 found.")
            continue

        project_dir = output_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)

        plot_project_samples(project_dir, project, runs)
        plot_project_strip(project_dir, project, runs)
        rows = write_project_summary_csv(project_dir, runs)
        sample_rows = build_sample_rows(project, runs)
        write_project_sample_points_csv(project_dir, sample_rows)
        all_rows.extend(rows)
        all_sample_rows.extend(sample_rows)

        print(f"\n[{project}] using {len(runs)} run(s):")
        for run in runs:
            print(f"  - {run.run_name} ({run.run_id}) created_at={run.created_at}")
        print(f"  Wrote: {project_dir / 'sample_metrics_bar.png'}")
        print(f"  Wrote: {project_dir / 'sample_metrics_strip.png'}")
        print(f"  Wrote: {project_dir / 'run_summary.csv'}")
        print(f"  Wrote: {project_dir / 'sample_points.csv'}")

        for metric in METRICS:
            metric_rows = [r for r in rows if r["metric"] == metric]
            metric_means = [r["mean"] for r in metric_rows if isinstance(r["mean"], float)]
            print(
                f"  Aggregate {metric}: mean_of_run_means={fmt(safe_mean(metric_means))} "
                f"runs={len(metric_rows)}"
            )

    if all_rows:
        write_global_summary_csv(output_dir, all_rows)
        print(f"\nGlobal summary written to: {output_dir / 'all_projects_summary.csv'}")
    if all_sample_rows:
        write_global_sample_points_csv(output_dir, all_sample_rows)
        global_stats_rows = write_global_statistics_csv(output_dir, all_sample_rows)
        plot_global_strip(output_dir, all_sample_rows, args.last_n)
        print(f"Global sample points written to: {output_dir / 'all_sample_points.csv'}")
        print(f"Global statistics written to: {output_dir / 'global_statistics.csv'}")
        print(f"Global strip plot written to: {output_dir / 'all_projects_strip.png'}")
        for metric in METRICS:
            metric_stats = next(
                (
                    row
                    for row in global_stats_rows
                    if row.get("level") == "metric" and row.get("metric") == metric
                ),
                None,
            )
            if metric_stats:
                print(
                    f"  Global {metric}: mean={fmt(metric_stats['mean'])} "
                    f"std={fmt(metric_stats['std'])} median={fmt(metric_stats['median'])} "
                    f"n={int(metric_stats['value_count'])}"
                )
    else:
        print("\nNo rows collected. Check entity/project access and metric availability.")


if __name__ == "__main__":
    main()
