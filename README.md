# PatchMamS2S

**PatchMamS2S: A Mamba-based Multivariate Time Series Forecasting Model in Power Domain with Patch Mining via Prior-based Weight Fusion**

## Overview

PatchMamS2S is an official implementation of a novel multivariate time series (MTS) forecasting model specifically designed for power load prediction. Published in 2026, this model integrates a patching strategy with the Mamba state space architecture to achieve superior performance on long-term forecasting tasks.

### Key Features

- **Multi-scale Patching**: Decomposes time series into multiple patches with different lengths (48, 24, 12, 6)
- **Mamba Architecture**: Utilizes efficient state space models for sequence modeling
- **Series Decomposition**: Separates seasonal and trend components using moving averages
- **Moving Average Fusion**: Combines multiple decomposition strategies with configurable weights
- **RevIN Normalization**: Supports Reversible Instance Normalization

## Project Structure

```
PatchMamS2S/
├── README.md                 # Project documentation
├── main.py                  # Main entry point for training
├── eval.py                  # Evaluation script
├── data/                    # Data loading utilities
│   ├── __init__.py
│   └── data_loader.py       # Dataset classes for MTS data
├── exp/                     # Experiment modules
│   ├── __init__.py
│   ├── exp_basic.py         # Base experiment class
│   └── exp_long_term_forecasting.py # Long-term forecasting experiments
├── models/                  # Model implementations
│   ├── PatchMamS2S.py       # Main PatchMamS2S model
│   ├── Crossformer.py       # Crossformer baseline
│   ├── Crossmamba.py        # Crossmamba model
│   ├── Mamba.py             # Simple Mamba model
│   ├── PatchMLP.py          # Patch MLP model
│   └── xPatch.py            # xPatch model
├── modules/                 # Core modules
│   ├── Mamba2Local.py       # Mamba2 implementation
│   ├── Transformer_EncDec.py
│   └── mamba_local/         # Mamba local implementation
└── utils/                   # Utility modules
    ├── tools.py             # Helper functions
    ├── metrics.py           # Evaluation metrics
    ├── normalizes.py        # Normalization techniques
    ├── c_adamw.py           # Custom AdamW optimizer
    └── efficiency_statistics.py # Performance metrics
```

## Installation

### Requirements

- Python 3.x
- PyTorch
- NumPy
- Pandas
- einops
- torchsummary
- thop
- tqdm

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd PatchMamS2S
```

2. Install required dependencies:
```bash
pip install torch numpy pandas einops torchsummary thop tqdm
```

## Data Preparation

The project supports multiple multivariate time series datasets. Download the datasets and place them in the `./datasets/` directory:

- **ETT datasets**: ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
- **Weather**: weather.csv
- **Traffic**: traffic.csv
- **Exchange**: exchange_rate.csv
- **Solar**: solar_AL.txt
- **ECL**: ECL.csv
- **ILI**: national_illness.csv
- **Australia**: Australia.csv
- **EP**: electric_price.csv

**Note**: Datasets need to be provided by users. Please ensure the data files are in CSV format and placed in the correct directory.

## Usage

### Training

Train a model using `main.py`:

```bash
python main.py --model PatchMamS2S --data ETTh1 --in_len 96 --out_len 96 --d_model 256 --d_state 1 --headdim 8 --k 13 --ma_type ema
```

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model name (PatchMamS2S, Crossformer, Crossmamba, Mamba, PatchMLP, xPatch) | - |
| `--data` | Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, etc.) | - |
| `--in_len` | Input sequence length (T) | 96 |
| `--out_len` | Output sequence length (tau) | 96 |
| `--train_epochs` | Number of training epochs | 20 |
| `--batch_size` | Batch size | 32 |
| `--learning_rate` | Initial learning rate | 0.0001 |
| `--d_model` | Dimension of hidden states | 256 |
| `--d_state` | Mamba state dimension | 1 |
| `--headdim` | Mamba2 head dimension | 8 |
| `--k` | Moving average kernel size | 13 |
| `--ma_type` | Moving average type (reg, ema, dema) | ema |
| `--alpha` | Alpha parameter for EMA/DEMA | 0.3 |
| `--beta` | Beta parameter for DEMA | 0.3 |
| `--revin` | Use RevIN normalization (1=True, 0=False) | 1 |
| `--use_gpu` | Use GPU if available | True |
| `--save_pred` | Save predicted future MTS | False |

### Evaluation

Evaluate a trained model using `eval.py`:

```bash
python eval.py --checkpoint_root ./checkpoints --setting_name PatchMamS2S_paper_ETTh1_96_96_dm256_1_8_lrtype1_decomp13_itr0_0514_1234_ema --inverse --save_pred
```

### Supported Models

1. **PatchMamS2S**: Main model with patching strategy and Mamba
2. **Crossformer**: Transformer-based baseline
3. **Crossmamba**: Mamba with cross-temporal and cross-dimensional dependencies
4. **Mamba**: Simple Mamba implementation
5. **PatchMLP**: Patch-based MLP
6. **xPatch**: Alternative patch-based model

## Model Architecture

The PatchMamS2S model consists of:

1. **Patch Embedding**: Multi-scale patch embedding with different patch lengths
2. **Series Decomposition**: Moving average-based decomposition into seasonal and trend components
3. **Mamba Encoder**: State space model-based encoder for sequence modeling
4. **Decoder**: Generates predictions for future time steps

## Evaluation Metrics

The following metrics are used to evaluate forecasting performance:

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)

## Example Commands

### Train PatchMamS2S on ETTh1
```bash
python main.py --model PatchMamS2S --data ETTh1 --in_len 96 --out_len 96 --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --d_model 256 --d_state 8 --headdim 8
```

### Train with Different Moving Average Type
```bash
python main.py --model PatchMamS2S --data ETTh1 --in_len 96 --out_len 96
```

### Evaluate Trained Model
```bash
python eval.py --checkpoint_root ./checkpoints --setting_name PatchMamS2S_paper_ETTh1_96_96_dm256_1_8_lrtype1_decomp13_itr0_time1234_ema --inverse --save_pred
```

## Citation

If you find our work useful in your research, please consider citing our papers:

```bibtex
@article{lin2025crossmamba,
  title={Crossmamba: multivariate time series forecasting model for cross-temporal and cross-dimensional dependencies with Mamba},
  author={Lin, Yuhan and Xiong, Liping and Hong, Zhiyong and Zeng, Zhiqiang and Zeng, Jian and Zeng, Guoqiang},
  journal={Data Mining and Knowledge Discovery},
  volume={39},
  number={5},
  pages={68},
  year={2025},
  publisher={Springer}
}

@article{lin2026patchmams2s,
  title={PatchMamS2S: A Mamba-based Multivariate Time Series Forecasting Model in Power Domain with Patch Mining via Prior-based Weight Fusion},
  author={Lin, Yuhan and Ouyang, Haoyuan and Chen, Hongyu and Xiong, Liping and Hong, Zhiyong and Wang, Zhishuang},
  year={2026}
}
```

## Contact

If you have any questions or concerns, please contact us:

- Email: YuhanLin4038@outlook.com
- Email: 2679146671@qq.com
- Or submit an issue on GitHub

## Related Work

If you are interested in this area, we recommend our previous work:

**Crossmamba: multivariate time series forecasting model for cross-temporal and cross-dimensional dependencies with Mamba**

Published in Data Mining and Knowledge Discovery, 2025

[Read the paper](https://link.springer.com/article/10.1007/s10618-025-01149-9)

## License

Please refer to the project license file for details.

Note: This README is intelligently generated by LLM