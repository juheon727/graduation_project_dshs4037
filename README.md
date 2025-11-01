# Few-Shot Keypoint Detection for Soccer Video Analysis

This project is a deep learning pipeline for few-shot keypoint detection, implemented in PyTorch and Lightning. The model is designed to identify novel keypoints in images given only a few (N-shot) annotated examples.

The core architecture uses a meta-learning approach:

1.  A powerful backbone feature extractor (`ResNet50-DeepLabv3+`) is trained to generate rich spatial features.
2.  A lightweight `HeatmapRegressor` head is instantiated *per task*.
3.  This head is rapidly adapted (fine-tuned) on the N-shot support set.
4.  The adapted head is then used to predict keypoint heatmaps for the query set.

The project also includes a complete "pseudo-dataset" generation pipeline that can download videos from YouTube, filter relevant frames using CLIP, and generate segmentation and keypoint-prompted annotations using YOLO and the Segment Anything Model (SAM).

## Features

  * **Model:** `ResNet50-DeepLabv3+` backbone with an adaptable `HeatmapRegressor` head.
  * **Method:** Few-shot, task-based meta-learning (N-shot, K-query).
  * **Training:** Managed by PyTorch Lightning (`train/train.py`) with logging to Weights & Biases (`wandb`).
  * **Loss Function:** KL Divergence (`kl_loss`) for regressing heatmap distributions.
  * **Dataset:** A custom `FSKeypointDataset` class (`train/dataset.py`) that serves few-shot tasks from COCO-formatted annotations.
  * **Evaluation:** `train/predict.py` script to load a checkpoint, run inference, and generate Percentage of Correct Keypoints (PCK) curves.
  * **Data Pipeline:** A `pseudodataset/downloader.py` script to automatically create datasets from YouTube URLs, using YOLO, SAM, and CLIP for annotation.

## Project Structure

```
.
├── config.yaml                   # Main configuration file for all scripts
├── requirements.txt              # Python dependencies
├── pseudodataset/
│   └── downloader.py             # Script to download and auto-annotate YouTube videos
├── train/
│   ├── architectures.py          # Defines the ResNet50-DeepLabv3+ model
│   ├── dataset.py                # PyTorch Dataset and Collator for few-shot tasks
│   ├── model.py                  # Defines the LitModule, MultiRegHead, and adapt logic
│   ├── train.py                  # Main script to start training
│   ├── predict.py                # Script to run evaluation and generate PCK curves
│   └── visualize_dataset.py      # Utility to debug and visualize data loader output
└── ...
```

## Setup

1.  Clone the repository:

    ```bash
    git clone <repo-url>
    cd <your-repo-name>
    ```

2.  Create and activate a virtual environment (recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## How to Use

### 1\. Configuration

All parameters for training, data, and evaluation are controlled by `config.yaml`. Before running any script, you **must** review and edit this file to set the correct paths and parameters:

  * `train.dataset_dir_train`: Path to your training dataset.
  * `train.dataset_dir_val`: Path to your validation dataset.
  * `train.model`: Model architecture to use.
  * `train.n_shot`, `train.n_query`, `train.batch_size`: Task parameters.
  * `pseudodataset.output_dir`: Where to save auto-annotated data.

### 2\. Dataset Generation (Optional)

If you do not have a dataset, you can use the provided pipeline to create one from YouTube videos.

1.  Add YouTube URLs to `pseudodataset_urls.yaml` or `config.yaml`.
2.  Configure the prompts (`clip_prompt`, `yolo_prompt`) in `config.yaml`.
3.  Run the downloader script:
    ```bash
    python -m pseudodataset.downloader
    ```
    *Note: This script assumes the `config.yaml` is in the parent directory.*

### 3\. Training

Once your dataset is ready and `config.yaml` is configured, you can start training:

```bash
python -m train.train
```

The script will use the parameters from `config.yaml` and log results to Weights & Biases (if configured).

### 4\. Evaluation (Prediction)

After training, you can evaluate a model checkpoint (`.ckpt`) on the validation or test set. This script will generate heatmap visualizations and a PCK curve.

```bash
python -m train.predict --ckpt_path /path/to/your/lightning_logs/model.ckpt
```

Additional options:

  * `--outdir`: Directory to save visualizations (default: `./visualizations`).
  * `--dataset_split`: Which dataset to use, 'train' or 'val' (default: `val`).
  * `--num_batches`: Number of batches to accumulate PCK over (default: `1`).

### 5\. Visualize Data

To verify your dataset and dataloader are working correctly, you can run `visualize_dataset.py`. This will save a batch of support/query images and their corresponding heatmaps as overlays.

```bash
python -m train.visualize_dataset --outdir ./data_visuals
```

## Key Dependencies

  * `torch` & `torchvision`
  * `lightning` (PyTorch Lightning)
  * `ultralytics` (for YOLO)
  * `transformers` (for CLIP)
  * `pytubefix` (for YouTube downloading)
  * `opencv-python`
  * `pycocotools`
  * `wandb`

## Dataset

Dataset used for this project is available at:
https://drive.google.com/drive/folders/1-_dBLxepIJCjxVqLNrAOWnKNS_0cDEdd?usp=sharing

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
