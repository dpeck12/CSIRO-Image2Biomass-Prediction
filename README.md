# CSIRO - Image2Biomass Prediction

Baseline local training + submission pipeline to compete in the Kaggle competition: https://www.kaggle.com/competitions/csiro-biomass

This repo provides:
- Data download scripts (Windows PowerShell)
- A PyTorch baseline model combining image + simple tabular features
- Training loop with weighted R² validation metric
- Inference script that generates the required `submission.csv`
- Configurable settings via `configs/config.yaml`

## Quick Start (Windows)

1) Set up Kaggle API (one-time)

Use the script to place your `kaggle.json`:

```powershell
pwsh scripts/setup_kaggle.ps1
```

2) Create Python environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3) Download competition data

```powershell
pwsh scripts/download_data.ps1
```

This will populate the `data/` folder with `train.csv`, `test.csv`, and image directories referenced therein.

4) Train the baseline

```powershell
python src/train.py
```

This saves a checkpoint to `outputs/model.pt` and prints validation weighted R² each epoch.

5) Run inference and create submission

```powershell
python src/infer.py
```

This writes `outputs/submission.csv` in the required long format with columns: `sample_id,target`.

## Configuration

Edit settings in [configs/config.yaml](configs/config.yaml):
- **data_root**: folder containing CSVs and images (default `data`)
- **train_csv** / **test_csv**: CSV file names inside `data_root`
- **image_size**, **batch_size**, **epochs**, **lr**: training hyperparameters
- **use_tabular**: include simple tabular features (date, NDVI, height, state, species)
- **backbone**: timm model name (default `efficientnet_b0`)
- **out_dir**: outputs directory

## Notes and Next Steps
- The competition’s scoring uses a globally weighted R² over all targets; validation computes this metric over the dev split.
- The current baseline pivots `train.csv` to a wide format and trains a 5-output regressor. Stronger results may come from advanced augmentations, backbones, and richer tabular encoding.
- Kaggle requires Notebook submissions; after local development, port `src/train.py` and `src/infer.py` logic to a Kaggle Notebook, ensure internet is disabled, and output `submission.csv`.

## Repository Layout

```
configs/
	config.yaml           # project settings
data/                   # populated via download script
notebooks/              # optional EDA and experiments
scripts/
	setup_kaggle.ps1      # set up Kaggle API credentials
	download_data.ps1     # download and extract competition data
src/
	biomass/
		data.py             # datasets for train/test
		metrics.py          # weighted R²
		model.py            # image+tabular regressor
		utils.py            # CSV utilities and pivot
	train.py              # training loop
	infer.py              # submission generation
requirements.txt        # Python dependencies
```

## Troubleshooting
- Torch install issues on Windows: use official wheels or `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision` matching your CUDA/CPU.
- If images are not found, check that `image_path` in CSV is relative to `data_root` and that `data/` contains the extracted `train/` and (hidden at scoring) `test/` directories.
