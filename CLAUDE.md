# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational data science lab focused on PyTorch fundamentals and linear regression. The project consists of Jupyter notebooks with hands-on exercises.

## Running the Notebooks

```bash
pip install torch numpy scikit-learn matplotlib
jupyter notebook
```

Open `.ipynb` files and run cells sequentially.

## Architecture

- **pytorch_preliminary.ipynb** — PyTorch fundamentals: tensor creation, operations, reshaping, broadcasting, NumPy/PyTorch conversions
- **linear_regression.ipynb** — Linear regression with PyTorch: data generation via sklearn, `nn.Linear(1,1)` model, MSELoss, SGD optimizer, 100-epoch training loop, Matplotlib visualization

Both notebooks are self-contained and independent.

## Conventions

- Exercise tasks are marked with `# TODO:` comments
- Standard PyTorch training pattern: forward pass → loss → backward → optimizer step
- Data generated with `sklearn.datasets.make_regression()`, converted to tensors via `torch.from_numpy()`
