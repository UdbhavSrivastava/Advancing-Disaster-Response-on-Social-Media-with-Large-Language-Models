import argparse
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def load_tasks(config_path, include_token=False):
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)

    tasks = []
    templates = config.get("task_templates", {})
    for name, task in config.get("tasks", {}).items():
        template = templates.get(task.get("template"), {})
        task_type = task.get("type", template.get("type"))
        if task_type == "token_classification" and not include_token:
            continue
        tasks.append(name)
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Run configured tasks for multiple seeds.")
    parser.add_argument("--tasks", nargs="*", default=["all"], help="Task names, or `all`.")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output_root", default="outputs/runs")
    parser.add_argument("--data_dir", required=True, help="CrisisMMD root containing Informative/Humanitarian/Damage.")
    parser.add_argument("--images_dir", required=True, help="CrisisMMD image root.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--include_token", action="store_true", help="Also attempt LMR token tasks.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.tasks == ["all"]:
        tasks = load_tasks(repo_root / "configs" / "data.yaml", include_token=args.include_token)
    else:
        tasks = args.tasks

    output_root = Path(args.output_root)
    metrics_csv = output_root / "all_runs.csv"

    for task in tasks:
        for seed in args.seeds:
            run_dir = output_root / task / f"seed_{seed}"
            metrics_json = run_dir / "metrics.json"
            if args.skip_existing and metrics_json.exists():
                print(f"Skipping existing {task} seed {seed}: {metrics_json}")
                continue

            cmd = [
                sys.executable,
                str(repo_root / "main.py"),
                "--task",
                task,
                "--seed",
                str(seed),
                "--output_dir",
                str(run_dir),
                "--data_dir",
                args.data_dir,
                "--images_dir",
                args.images_dir,
                "--metrics_csv",
                str(metrics_csv),
            ]
            if args.epochs is not None:
                cmd.extend(["--epochs", str(args.epochs)])
            if args.lr is not None:
                cmd.extend(["--lr", str(args.lr)])
            if args.batch_size is not None:
                cmd.extend(["--batch_size", str(args.batch_size)])

            print("Running:", " ".join(cmd))
            result = subprocess.run(cmd, cwd=repo_root)
            if result.returncode != 0:
                message = f"Run failed for {task} seed {seed} with exit code {result.returncode}"
                if args.keep_going:
                    print(message)
                    continue
                raise SystemExit(message)

    summary_csv = output_root / "ci_summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tools" / "aggregate_ci.py"),
            "--root",
            str(output_root),
            "--out_csv",
            str(summary_csv),
            "--per_run_csv",
            str(metrics_csv),
        ],
        cwd=repo_root,
        check=True,
    )
    print(f"Wrote CI summary: {summary_csv}")


if __name__ == "__main__":
    main()
