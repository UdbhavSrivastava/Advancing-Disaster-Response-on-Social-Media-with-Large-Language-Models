# Advancing Disaster Response on Social Media with Large Language Models

This project provides a unified and modular framework for training and evaluating machine learning models on crisis-related data. It supports various tasks, including text-only, image-only, and multimodal classification, by leveraging large language models (LLMs) like Llama and vision models like ResNet and ViT.

The codebase is designed for reproducibility and ease of use, allowing you to define and run experiments with simple command-line arguments.

***

## Project Structure

The project is organized into a clean, hierarchical structure to separate concerns.

```text
.
├── configs/
│   ├── data.yaml
│   ├── models.yaml
│   └── training.yaml
├── data/
│   └── (raw data files from CrisisMMD_v2.0, LMR, etc.)
├── src/
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
├── LICENSE
├── main.py
├── README.md
└── requirements.txt
***

```
## Setup

Follow these steps to set up the project locally.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/UdbhavSrivastava/Advancing-Disaster-Response-on-Social-Media-with-Large-Language-Models
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

```
***

## Acknowledgments

This project relies on several key resources and open-source contributions. We extend our gratitude to the following:

* **CrisisMMD Dataset**: https://crisisnlp.qcri.org/crisismmd
* **Idrisi LMR Dataset**: https://github.com/rsuwaileh/IDRISI
* **Hugging Face**: For the excellent `transformers` and `datasets` libraries that form the foundation of our modeling pipeline.
* **Pre-trained Models**: We thank the creators of the pre-trained models used in this work, including:
    * **Llama** for the text-based models and embeddings. Specially LS-LLama - https://github.com/4AI/LS-LLaMA
    * **ResNet** for the image feature extraction models.
    * **ViT** for the advanced vision transformer models.
