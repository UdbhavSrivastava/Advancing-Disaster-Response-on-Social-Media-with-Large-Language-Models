import argparse
import csv
import json
import sys
import importlib
import yaml
from pathlib import Path

def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_configs(task_name):
    base_dir = Path(__file__).parent

    try:
        with open(base_dir / 'configs' / 'data.yaml', 'r') as f:
            data_config_all = yaml.safe_load(f)
        with open(base_dir / 'configs' / 'models.yaml', 'r') as f:
            model_config_all = yaml.safe_load(f)
        with open(base_dir / 'configs' / 'training.yaml', 'r') as f:
            training_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Error: Missing configuration file. {e}")
        sys.exit(1)

    try:
        task_config_raw = data_config_all['tasks'][task_name]
    except KeyError:
        print(f"Error: Task '{task_name}' not found in data configuration.")
        sys.exit(1)

    template_name = task_config_raw.get("template")
    template_config = data_config_all.get("task_templates", {}).get(template_name, {})
    task_config = _deep_merge(template_config, task_config_raw)
    task_config["name"] = task_name
    task_config["template_name"] = template_name
    task_config["base_paths"] = data_config_all.get("base_paths", {})

    model_config = {}
    model_config.update(model_config_all.get("base_models", {}))
    model_config["fusion_models"] = model_config_all.get("fusion_models", {})

    return data_config_all, task_config, model_config, training_config

def write_metrics(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Run a specific machine learning task.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task to run (e.g., 'text_informative').")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs to override config.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate to override config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size to override config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for this run.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for this run.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override CrisisMMD text/TSV root.")
    parser.add_argument("--images_dir", type=str, default=None, help="Override CrisisMMD image root.")
    parser.add_argument("--lmr_base_dir", type=str, default=None, help="Override LMR dataset root.")
    parser.add_argument("--metrics_csv", type=str, default=None, help="Optional CSV to append one row of metrics.")
    
    args = parser.parse_args()
    
    data_config_all, task_config, model_config, training_config = load_configs(args.task)

    if args.data_dir:
        task_config["base_paths"]["data_dir"] = args.data_dir
    if args.images_dir:
        task_config["base_paths"]["images_dir"] = args.images_dir
    if args.lmr_base_dir:
        task_config["base_paths"]["lmr_base_dir"] = args.lmr_base_dir
    
    training_config_base = training_config['base']
    if args.epochs:
        training_config_base['num_train_epochs'] = args.epochs
    if args.lr:
        training_config_base['learning_rate'] = args.lr
    if args.batch_size:
        training_config_base['per_device_train_batch_size'] = args.batch_size
        training_config_base['per_device_eval_batch_size'] = args.batch_size
    training_config_base['seed'] = args.seed
    training_config_base['data_seed'] = args.seed
    if args.output_dir:
        training_config_base['output_dir'] = args.output_dir
    
    try:
        task_type = task_config['type']
        if task_type in ["text_only", "token_classification"]:
            module_name = "src.tasks.text_tasks"
        elif task_type == "image_only":
            module_name = "src.tasks.image_tasks"
        elif task_type == "multimodal":
            module_name = "src.tasks.multimodal_tasks"
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        task_module = importlib.import_module(module_name)
        runner_function = getattr(task_module, 'run_task')
        
        metrics = runner_function(task_config, model_config, training_config_base)
        row = {
            "task": args.task,
            "seed": args.seed,
            "output_dir": training_config_base["output_dir"],
        }
        row.update(metrics or {})

        metrics_path = Path(training_config_base["output_dir"]) / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w") as f:
            json.dump(row, f, indent=2, sort_keys=True)

        if args.metrics_csv:
            write_metrics(args.metrics_csv, row)
        
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find or run the task script for '{args.task}'. Please ensure the file and function exist. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
