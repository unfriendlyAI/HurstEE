
http://dx.doi.org/10.1088/1751-8121/ae17f9

Kaggle code:

https://www.kaggle.com/code/unfriendlyai/hurstee-layer-for-estimating-hurst

https://www.kaggle.com/code/unfriendlyai/hurstautograd


# HurstEE: Differentiable Hurst Exponent Estimation Layer

This repository contains the implementation of **HurstEE**, a differentiable neural network layer that estimates the Hurst exponent () and the anomalous diffusion exponent () using the Time-Averaged Mean Squared Displacement (TA-MSD) method.

It is available for both **TensorFlow** and **PyTorch**.

## Features

* **Differentiable:** Can be integrated directly into Neural Networks as a layer.
* **Non-trainable:** Does not require gradient updates; it calculates statistical properties deterministically.
* **Robust:** Handles `NaN` values (missing data) and short trajectories.
* **Plug-and-Play:** Simple integration into existing architectures.

## Input Data Format

For both frameworks, the expected input tensor shape is:


---

## 1. Using with TensorFlow

### How to Import

1. Open `hurstee.ipynb`.
2. Locate the cell titled **"HurstEE with TensorFlow"** (Cell 4).
3. Copy the entire `class HurstEE(tf.keras.layers.Layer):` definition into your project.

### Usage Example

```python
import tensorflow as tf
import numpy as np

# --- PASTE THE HurstEE CLASS DEFINITION HERE --- 
# class HurstEE(tf.keras.layers.Layer): ...

# 1. Instantiate the layer
# use_correction=True helps with bias in very short trajectories
hurst_layer = HurstEE(use_correction=False) 

# 2. Create dummy data (Batch=32, Time=100, Channels=1)
input_data = np.random.randn(32, 100, 1).astype(np.float32)

# 3. Apply the layer
# Returns the estimated Hurst exponent (H) for each sample
estimated_hurst = hurst_layer(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {estimated_hurst.shape}") # (32,)

```

---

## 2. Using with PyTorch

### How to Import

1. Open `hurstee.ipynb`.
2. Locate the cell titled **"HurstEE with PyTorch"** (Cell 5).
3. Copy the entire `class PyHurstEE(nn.Module):` definition into your project.

### Usage Example

```python
import torch
import numpy as np

# --- PASTE THE PyHurstEE CLASS DEFINITION HERE ---
# class PyHurstEE(nn.Module): ...

# 1. Instantiate the module
hurst_layer = PyHurstEE(use_correction=False)

# 2. Create dummy data (Batch=32, Time=100, Channels=1)
input_data = torch.randn(32, 100, 1)

# 3. Apply the layer
estimated_hurst = hurst_layer(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {estimated_hurst.shape}") # torch.Size([32])

```

---

## Parameters

Both classes accept the following initialization parameter:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `use_correction` | `bool` | `False` | If `True`, applies a TEA-MSD & variance correction factor. Useful for heterogeneous or very short trajectories to reduce estimation bias. |

## Methodology

The layer computes the Hurst exponent based on the logarithmic slope of the Mean Squared Displacement (MSD) over time lags. It automatically filters out `NaN` values, making it suitable for datasets with variable-length trajectories (e.g., the AnDi Challenge datasets).

The output is clipped to the range .

# HurstEE
Differentiable Neural Network Layer for Estimating Hurst and Anomalous Diffusion Exponents, TensorFlow and PyTorch

Roman Lavrynenko, Lyudmyla Kirichenko, Nataliya Ryabova and Sophia Lavrynenko

Neural networks have shown excellent performance in the task of estimating the Hurst exponent, but their primary drawback is a lack of explainability. We introduce a specialized neural network layer to address this. This layer is an implementation of the second-order Generalized Hurst Exponent method, designed as a non-trainable, differentiable layer compatible with any deep learning framework. We have implemented it for both TensorFlow and PyTorch. The layer also seamlessly handles missing values, which facilitates its integration into complex neural network architectures designed for analyzing heterogeneous trajectories. Our differentiable Hurst exponent estimation layer offers simplicity in deployment, as it eliminates the need for training and is ready to process time series of any length. While the convenience of not requiring training and its flexibility with series length are clear advantages, the key novelty of our work is the ability to provide interpretability by successfully translating a classical statistical method into a core deep learning component.
