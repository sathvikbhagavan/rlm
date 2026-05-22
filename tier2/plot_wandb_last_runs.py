import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tier2 wrapper for publication-style plotting (same recipe as tier3)."
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="/home/bhagavan/rlms/rlm/tier2/wandb_last_runs_report_tables",
        help="Directory containing tabulated CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory for figures.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="accuracy",
        choices=("accuracy", "precision", "recall", "f1"),
        help="Metric to visualize.",
    )
    parser.add_argument(
        "--context-sizes",
        type=str,
        default="all",
        help="Optional context-size filter (e.g. '0,100').",
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


def main() -> None:
    args = parse_args()
    tier3_script = Path("/home/bhagavan/rlms/rlm/tier3/plot_wandb_last_runs.py")
    if not tier3_script.exists():
        raise FileNotFoundError(f"Missing tier3 script: {tier3_script}")

    cmd = [
        sys.executable,
        str(tier3_script),
        "--tables-dir",
        args.tables_dir,
        "--metric",
        args.metric,
        "--context-sizes",
        args.context_sizes,
    ]
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.min_task_number is not None:
        cmd.extend(["--min-task-number", str(args.min_task_number)])
    if args.max_task_number is not None:
        cmd.extend(["--max-task-number", str(args.max_task_number)])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
