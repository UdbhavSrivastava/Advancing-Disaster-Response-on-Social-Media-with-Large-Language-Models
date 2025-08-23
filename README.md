# Advancing Disaster Response on Social Media with Large Language Models

This project provides a unified and modular framework for training and evaluating machine learning models on crisis-related data. It supports various tasks, including text-only, image-only, and multimodal classification, by leveraging large language models (LLMs) like Llama and vision models like ResNet and ViT.

The codebase is designed for reproducibility and ease of use, allowing you to define and run experiments with simple command-line arguments.

***

## Features

* **Modular Architecture**: All core logic for data loading, modeling, and training is separated into reusable modules.
* **Single Entry Point**: All experiments are run from a single `main.py` script, which simplifies the workflow.
* **Configurable Experiments**: Hyperparameters, model choices, and task-specific settings are managed in YAML configuration files.
* **Multi-Modality Support**: The framework supports tasks on text, images, and multimodal data.
* **Extensible Design**: Easily add new tasks, models, or data sources by updating the configuration files or adding new modules to the `src/` directory.

***

## Project Structure

The project is organized into a clean, hierarchical structure to separate concerns.

Markdown

# Multimodal Crisis Management Classification

This project provides a unified and modular framework for training and evaluating machine learning models on crisis-related data. It supports various tasks, including text-only, image-only, and multimodal classification, by leveraging large language models (LLMs) like Llama and vision models like ResNet and ViT.

The codebase is designed for reproducibility and ease of use, allowing you to define and run experiments with simple command-line arguments.

***

## Features

* **Modular Architecture**: All core logic for data loading, modeling, and training is separated into reusable modules.
* **Single Entry Point**: All experiments are run from a single `main.py` script, which simplifies the workflow.
* **Configurable Experiments**: Hyperparameters, model choices, and task-specific settings are managed in YAML configuration files.
* **Multi-Modality Support**: The framework supports tasks on text, images, and multimodal data.
* **Extensible Design**: Easily add new tasks, models, or data sources by updating the configuration files or adding new modules to the `src/` directory.

***

## Project Structure

The project is organized into a clean, hierarchical structure to separate concerns.

.
├── configs/                  # All configuration files
├── data/                     # Raw data files
├── src/                      # All reusable source code
│   ├── data/
│   │   ├── base_dataloader.py
│   │   └── text_preprocessing.py
│   ├── models/
│   │   ├── base_models.py
│   │   ├── fusion_models.py
│   │   └── modeling_llama.py
│   ├── tasks/
│   │   ├── run_image_tasks.py
│   │   ├── run_multimodal_tasks.py
│   │   └── run_text_tasks.py
│   └── utils/
│       └── metrics.py
├── .gitignore
├── main.py                   # Main entry point for all tasks
├── README.md                 # This file
└── requirements.txt

***

## Setup

Follow these steps to set up the project locally.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment** (optional but recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data and Configurations**
    * Place your raw data files in the `data/` directory.
    * 
    * **Edit the configuration files** in the `configs/` directory to match your data paths and define your experiments. This is a crucial step!

***

## Running Experiments

All experiments are run using the `main.py` script. You can specify a task and override default hyperparameters from the command line.

### Basic Usage

The `--task` argument is required. Its value must match a task name defined in `configs/data.yaml`.

```bash
# Example: Run the default text-only informative task
python main.py --task text_informative
