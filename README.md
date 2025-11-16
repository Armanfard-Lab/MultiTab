<div align="center">
<h1>[AAAI'26] MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data</h1>
</div>

[![ArXiv](https://img.shields.io/badge/ArXiv-2511.09970-red)](https://arxiv.org/abs/2511.09970)

<div align="center">
<img src="figures/multitab.png" style="width: 4000px; height: auto;">
</div>

## Environment Setup

Follow these steps to set up the conda environment for the MultiTab project:

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system
- NVIDIA GPU with CUDA support (recommended for training)

### Installation Steps

1. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate multitab
   ```

3. **Verify the installation:**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Dataset Setup Instructions

This project supports three datasets for multitask learning experiments:

- **Higgs Dataset**: High-energy physics dataset for binary classification with additional regression targets
- **ACS Income Dataset**: American Community Survey data for income prediction and demographic analysis  
- **AliExpress Dataset**: E-commerce dataset for click-through rate and conversion prediction

### Quick Setup (Recommended)

The easiest way to set up all datasets is to use our preprocessed H5 files available on Hugging Face:

1. **Configure the data root directory:**
   Edit `download_data.sh` and set your desired data root:
   ```bash
   DATA_ROOT="/path/to/your/data/"
   ```

2. **Make the script executable:**
   ```bash
   chmod +x download_data.sh
   ```

3. **Run the download script:**
   ```bash
   ./download_data.sh
   ```

   This will automatically download all three preprocessed datasets (Higgs, AliExpress, and ACS Income) in H5 format and organize them in the correct directory structure.

### Manual Dataset Setup (Optional)

If you prefer to download the datasets from their original sources and perform the preprocessing yourself, please refer to the [manual dataset setup instructions](data/manual_dataset_setup.md).

## Running Experiments

Once you have set up your datasets, you can run experiments using the provided training script:

1. **Make the script executable:**
   ```bash
   chmod +x run.sh
   ```

2. **Configure the experiment parameters:**
   Edit the variables at the top of `run.sh`:
   ```bash
   DATA_ROOT="/path/to/data/"  # Path to your processed datasets
   MODEL_NAME="mtt"            # Model to use (mtt, mmoe, ple, etc.)
   DATASET="acs_income"        # Dataset name (acs_income, higgs, etc.)
   GPU_ID=0                    # GPU ID for training
   SEED=42                     # Random seed for reproducibility
   PATIENCE=5                  # Early stopping patience
   ```

3. **Run the experiment:**
   ```bash
   ./run.sh
   ```

The script will automatically start training with the specified configuration and save results to the logs directory.

## **📚 Citation**
If you use this code or find our work helpful, please cite:
```bibtex
@misc{sinodinos2025multitabscalablefoundationmultitask,
      title={MultiTab: A Scalable Foundation for Multitask Learning on Tabular Data}, 
      author={Dimitrios Sinodinos and Jack Yi Wei and Narges Armanfard},
      year={2025},
      eprint={2511.09970},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.09970}, 
}
```
