# RealFabric-NIR: A Multi-Task Benchmark Dataset for Real-World Challenges in Fabric Component Analysis


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Recommend pytorch > 2.5.1, CUDA > 12.0

## Train Test code

baseline_train_test.py

Can adjust model: ex) model = convnextv2_huge() or swin_t

## ⚠️ Notice on Code Availability

Please note that **some components of this project** (e.g., specific model implementations or pretrained weights) are currently being used in **commercial applications**. Due to **licensing and confidentiality constraints**, we are **unable to publicly release** those parts of the codebase.

However, we have provided:

- ✅ All essential components needed to **reproduce the main experimental results** presented in the paper.
- 🧩 **Placeholder or stub code** with clear documentation for restricted modules, so users can integrate their own alternatives if needed.

We appreciate your understanding.  
If you have questions or are interested in collaboration, feel free to open an issue or contact us directly.



## Reference

1. https://github.com/timeseriesAI/tsai/tree/main
2. https://github.com/meraks/Swin-Transformer-1D/tree/main
3. https://github.com/microsoft/Swin-Transformer/tree/main

