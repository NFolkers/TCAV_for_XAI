# TCAV for XAI

## Description

In this project we try to find out whether ResNet-18 is sensitive for the concept of feathers when classifying different bird classes.

## Dataset

To run this project, you need to download the dataset and unzip the contents to the `data` folder.
The data can be downloaded here:
https://www.kaggle.com/datasets/ambityga/imagenet100?resource=download

## Installing Requirements

Set up your virtual environment of choice or go ahead and install the required packages globally by installing the requirements:

```bash
pip install -r requirements.txt
```

# Execution

## 1. **tcav.py**

### Arguments:

- `--layer_name`: Specifies the name of the layer to analyze. Default is `'layer4'`.
- `--log_filename`: Specifies the log filename to store results. Default is `'tcav_ranking_log_cosine.csv'`.

```bash
python tcav.py --layer_name <layer_name> --log_filename <log_filename>
```

Examples:

```bash
python tcav.py --layer_name layer4 --log_filename tcav_results.csv
```

## 2. **analytics.py**

Arguments:

- csv_file: Path to the input CSV file (required, no default).
- -c, `--concept`: Name of the concept analyzed, used in plot titles (default: 'feathers').
- -s, `--sort_by`: Column name to sort results by, such as 'TCAV Score', 'Avg Rank Score', or 'Target Class' (default: 'Avg Rank Score').
- -a, `--ascending`: Sorts results in ascending order if set (default: False for descending).
- -n, `--top_n`: Number of top or bottom rows to display (default: 10).
- -o, `--output_csv`: Path to save the sorted results as a CSV file (default: None, optional).
- -p, `--plot`: Generates summary plots if set (default: False).

```bash
python analytics.py <csv_file> --concept <concept> --sort_by <sort_by> --ascending --top_n <top_n> --output_csv <output_csv> --plot
```

Example:

```bash
python analytics.py tcav_ranking_log_cosine.csv --concept feathers --sort_by TCAV Score --top_n 5 --plot
```

## 3. **datasampler.py**

- `--data_root`: Root directory of the dataset (default: './data').
- `--split`: Data split to sample from, e.g., 'train.X' (default: 'train.X').
- `--num_samples`: Number of random samples to draw (default: 250).
- `--output_dir`: Directory to save sampled images (default: './concepts').

```bash
python datasampler.py --data_root <data_root> --split <split> --num_samples <num_samples> --output_dir <output_dir>
```

Example:

```bash
python datasampler.py --data_root ./data --split train.X --num_samples 100 --output_dir ./sampled_concepts
```
