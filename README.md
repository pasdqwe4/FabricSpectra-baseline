# FabricSpectra: A Large-Scale Benchmark Dataset for Real-World Multi-Task Fabric Component Analysis

## Repository Structure

- **`data_utils/`**: Utility scripts for data preprocessing.
- **`model/`**: Definitions of model architectures used in experiments.
- **`baseline_train_test.py`**: Main script for training and evaluating models.
- **`model_utils.py`**: Helper functions for model operations.
- **`requirements.txt`**: Python dependencies required for the project.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

The project has been tested on:
- **OS**: Ubuntu 20.04.4 LTS
- **PyTorch**: 2.5.1 or higher
- **CUDA**: 12.0 or higher

## Train Test code

Use the main script baseline_train_test.py for training and evaluation. You can select different models by modifying the script.

Example: Using ConvNeXtV2-Huge

```setup
model = convnextv2_huge()
```
Example: Using Swin Transformer-Tiny

```setup
model = swin_t()
```
## ⚠️ Notice on Code Availability

Please note that **some components of this project** (e.g., specific model implementations or pretrained weights) are currently being used in **commercial applications**. Due to **licensing and confidentiality constraints**, we are **unable to publicly release** those parts of the codebase.

However, we have provided:

- ✅ All essential components needed to **reproduce the main experimental results** presented in the paper.

We appreciate your understanding.  
If you have questions or are interested in collaboration, feel free to open an issue or contact us directly.



## Reference

1. https://github.com/timeseriesAI/tsai/tree/main
2. https://github.com/meraks/Swin-Transformer-1D/tree/main
3. https://github.com/microsoft/Swin-Transformer/tree/main
4. https://github.com/amazon-science/chronos-forecasting

