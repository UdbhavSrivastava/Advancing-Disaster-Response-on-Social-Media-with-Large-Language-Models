# main.py
import argparse
import sys
import importlib
import yaml
from pathlib import Path

def load_configs(task_name):
    # Using Path to get the parent directory of this script
    base_dir = Path(__file__).parent
    
    # Load and merge configurations
    try:
        with open(base_dir / 'configs' / 'data.yaml', 'r') as f:
            data_config_all = yaml.safe_load(f)
            task_config = data_config_all['tasks'][task_name]
        with open(base_dir / 'configs' / 'models.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        with open(base_dir / 'configs' / 'training.yaml', 'r') as f:
            training_config = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Error: Missing configuration file. {e}")
        sys.exit(1)
    except KeyError:
        print(f"Error: Task '{task_name}' not found in data configuration.")
        sys.exit(1)
        
    return data_config_all, task_config, model_config, training_config

def main():
    parser = argparse.ArgumentParser(description="Run a specific machine learning task.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task to run (e.g., 'text_informative').")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs to override config.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate to override config.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size to override config.")
    
    args = parser.parse_args()
    
    data_config_all, task_config, model_config, training_config = load_configs(args.task)
    
    # Apply command-line overrides
    training_config_base = training_config['base']
    if args.epochs:
        training_config_base['num_train_epochs'] = args.epochs
    if args.lr:
        training_config_base['learning_rate'] = args.lr
    if args.batch_size:
        training_config_base['per_device_train_batch_size'] = args.batch_size
        training_config_base['per_device_eval_batch_size'] = args.batch_size
    
    try:
        task_type = task_config['type']
        if task_type in ["text_only", "token_classification"]:
            module_name = "src.tasks.run_text_tasks"
        elif task_type == "image_only":
            module_name = "src.tasks.run_image_tasks"
        elif task_type == "multimodal":
            module_name = "src.tasks.run_multimodal_tasks"
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        task_module = importlib.import_module(module_name)
        runner_function = getattr(task_module, 'run_task')
        
        runner_function(task_config, model_config, training_config_base)
        
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not find or run the task script for '{args.task}'. Please ensure the file and function exist. Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()