# Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion

This repository implements a physics-guided deep learning workflow for reconstructing 3D density contrast models from gravity observations. The method combines a 2D residual encoder, a channel-to-depth lifting module, a 3D Transformer bottleneck, cross-dimensional attention, and a differentiable gravity forward operator. The methodological description below follows the manuscript draft `A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion(2).docx`, while the commands, defaults, and implementation notes are taken from the current codebase in this repository.

The repository already includes:

- a pretrained checkpoint: `best_model.pth`
- synthetic `.vti` examples in `examples/`
- saved folder-based validation outputs in `folder_validation_results/`
- saved field-data outputs in `batch_test_results/`

<p align="center">
  <img src="batch_test_results/examples/synthetic_model_inversion/comparison.png" width="92%" alt="Synthetic example comparison">
</p>

<p align="center">
  <img src="batch_test_results/field_data/gzz/prediction.png" width="92%" alt="Field-data prediction">
</p>

## Highlights

- Hybrid 2D-to-3D inversion network for volumetric density reconstruction
- Joint input mode using `Gz`, `Gzz`, and normalized depth encoding
- Channel-to-depth lifting to convert 2D latent features into a 3D seed volume
- 3D Transformer bottleneck with sinusoidal position encoding and depth attention
- Physics-guided training with a differentiable gravity forward operator
- On-the-fly synthetic data generation with curriculum learning and augmentation
- Included checkpoint, example models, validation metrics, and visualization outputs

## Repository Structure

```text
.
|-- source code/
|   |-- config.py               # Default configuration and hyperparameters
|   |-- data_preparation.py     # Synthetic model generation, augmentation, forward modeling
|   `-- train_code.py           # Network, loss, trainer, and visualization
|-- examples/                   # Bundled .vti synthetic examples and figure assets
|-- Field data example/         # Example field Gzz grid and predicted density figure
|-- folder_validation_test.py   # Evaluate a checkpoint on .vti models in folders
|-- folder_validation_results/  # Saved validation outputs for bundled examples
|-- batch_test_results/         # Saved synthetic and field-data prediction outputs
|-- best_model.pth              # Included pretrained checkpoint
`-- README.md
```

## Method Overview

### 1. Input representation

The default training mode is `joint`. For each sample, the code constructs a volumetric input tensor with shape `[B, 3, D, H, W]`:

- channel 1: normalized `Gz`
- channel 2: normalized `Gzz`
- channel 3: normalized depth coordinate `z in [0, 1]`

With the default configuration:

- `D = 16`
- `H = W = 32`
- `dx = dz = 100 m`

In `gz` mode, the input shape becomes `[B, 2, D, H, W]` and omits the `Gzz` channel.

### 2. Network architecture

The current implementation in [`source code/train_code.py`](source%20code/train_code.py) follows this high-level flow:

1. A shared 2D residual encoder processes each depth slice of the repeated input volume.
2. Slice-wise latent features are aggregated across depth.
3. `ChannelToDepthLifting` converts the 2D latent map into a coarse 3D feature volume.
4. A 3D Transformer bottleneck refines the volume with global spatial context.
5. `CrossDimAttention` injects high-level 2D features back into the 3D pathway.
6. A 3D decoder upsamples laterally to produce the final density volume.
7. An unsharp-mask module sharpens the predicted boundaries.

Default architectural settings from [`source code/config.py`](source%20code/config.py):

| Parameter | Value |
|---|---:|
| `grid_shape` | `(16, 32, 32)` |
| `encoder_channels` | `(32, 64, 128, 256)` |
| `decoder_channels` | `(128, 64, 32)` |
| `lifting_channels` | `64` |
| `num_transformer_layers` | `4` |
| `num_heads` | `8` |
| `use_position_encoding` | `True` |
| `use_depth_attention` | `True` |

### 3. Training objective

The manuscript emphasizes five principal loss terms: physics consistency, depth weighting, focal weighting, gradient-difference regularization, and edge enhancement. The current implementation keeps those core terms and adds several extra regularizers.

The implemented total loss is:

```text
L_total =
    w_depth    * L_depth
  + w_focus    * L_focus
  + w_gdl      * L_gdl
  + w_physics  * L_physics
  + w_edge     * L_edge
  + w_l1       * L_l1
  + w_morph    * L_morph
  + w_boundary * L_boundary
```

Core terms, following the manuscript and the code:

- `L_physics`: mean-squared error between observed gravity and forward-modeled gravity from the predicted density volume
- `L_depth`: depth-weighted MSE with weights proportional to `(z + epsilon)^beta`
- `L_focus`: anomaly-focused weighted MSE using `1 + beta_f * |rho_true|`
- `L_gdl`: gradient-difference loss on 3D density gradients
- `L_edge`: edge-enhancement loss that emphasizes boundary mismatch along the three axes

Additional implementation terms:

- `L_l1`: sparsity-style penalty on predicted density magnitude
- `L_morph`: morphology regularization using differentiable opening and closing
- `L_boundary`: explicit boundary sharpening loss with normalized edge weights

Implementation detail: the code also uses an `AdaptiveLossBalancer` to keep different loss components on comparable scales throughout training.

Default loss-related parameters:

| Parameter | Value |
|---|---:|
| `w_depth` | `1.0` |
| `w_focus` | `1.0` |
| `w_gdl` | `1.5` |
| `w_physics` | `0.3` |
| `w_edge` | `0.5` |
| `w_l1` | `0.05` |
| `w_morph` | `0.05` |
| `w_boundary` | `0.5` |
| `depth_beta` | `2.0` |
| `focus_beta` | `10.0` |

Important note: in the current code, the differentiable physics-consistency term is computed against `Gz`. In `joint` mode, `Gzz` is used as an additional input channel, but the physics loss itself is driven by the forward-modeled `Gz` response.

## Data Generation and Training Data

Training data are generated on the fly in [`source code/data_preparation.py`](source%20code/data_preparation.py), so no prebuilt training dataset is required.

The generator mixes a wide range of synthetic geological bodies, including:

- prisms
- spheres
- trapezoids
- staircase bodies
- nested bodies
- multiscale blocky structures
- faults
- Perlin-like terrain models
- fractal bodies
- figure-eight staircase patterns
- separated prisms
- hollow boxes
- layered density models
- scattered anomalies
- more realistic mixed anomalies

The dataset logic also includes:

- curriculum learning for the first `30` epochs
- random flips
- random density scaling
- elastic deformation
- depth shifting
- geological noise injection
- `1%` to `5%` observation noise on synthetic gravity data

## Training Defaults

The main training defaults from [`source code/config.py`](source%20code/config.py) are:

| Parameter | Value |
|---|---:|
| `data_mode` | `joint` |
| `lr` | `5e-4` |
| `batch_size` | `8` |
| `epochs` | `400` |
| `steps_per_epoch` | `500` |
| optimizer | `AdamW` |
| weight decay | `1e-4` |
| scheduler | `CosineAnnealingLR` |
| scheduler minimum LR | `1e-6` |
| mixed precision | enabled on CUDA |

## Installation

The project does not currently ship with a `requirements.txt`, so install the main dependencies manually:

```bash
pip install torch numpy scipy matplotlib vtk
```

Notes:

- `vtk` is required for reading bundled `.vti` models during folder-based validation.
- A CUDA-enabled PyTorch build is recommended for training.

## Usage

### 1. Set a writable checkpoint directory

Before training, update `save_dir` in [`source code/config.py`](source%20code/config.py), because the current default path is machine-specific:

```python
save_dir: str = "/home/jszxgx/ysw/deeplearn/checkpoints_v4_new"
```

Change it to a valid local path on your machine.

### 2. Train the model

From the repository root:

```bash
python "source code/train_code.py" --epochs 400 --batch_size 8 --lr 5e-4
```

To run only the gradient-flow check for the physics loss:

```bash
python "source code/train_code.py" --verify-only
```

### 3. Evaluate the bundled `.vti` examples

The tracked evaluation script is [`folder_validation_test.py`](folder_validation_test.py). It loads `best_model.pth`, reads all `.vti` models under `examples/`, and writes per-case arrays, metrics, and summaries.

```bash
python folder_validation_test.py ^
  --checkpoint best_model.pth ^
  --models-dir examples ^
  --output-dir folder_validation_results ^
  --device auto
```

On Linux or macOS:

```bash
python folder_validation_test.py \
  --checkpoint best_model.pth \
  --models-dir examples \
  --output-dir folder_validation_results \
  --device auto
```

Generated outputs include:

- `pred_density.npy`
- `true_density.npy`
- `obs_gravity.npy`
- `pred_gravity.npy`
- `metrics.json`
- `summary.csv`
- `summary.json`
- `vis_highres/*.png`

### 4. Use your own data

The current tracked repository is strongest for:

- training from synthetic data
- evaluating `.vti` synthetic examples with `folder_validation_test.py`

The repository also contains a field-data example:

- input grid: `Field data example/Gzz.txt`
- saved prediction artifacts: `batch_test_results/field_data/gzz/`

If you want to adapt the model to your own data, keep the following conventions in mind:

- match the configured grid size or change `grid_shape`, `dx`, and `dz`
- normalize each gravity channel by its maximum absolute value
- in `joint` mode, build the input stack as `[Gz, Gzz, depth_encoding]`
- density targets are trained in normalized contrast space, typically clipped to `[-1, 1]`

## Bundled Results

### Synthetic example metrics

The repository already includes saved evaluation summaries in `folder_validation_results/summary.csv`. The main synthetic cases report:

| Example | Deep IoU | Density RMSE | Density Corr. | Gravity Corr. |
|---|---:|---:|---:|---:|
| Synthetic Model Inversion | `0.8252` | `0.0798` | `0.9661` | `0.9996` |
| Complex Geological Structure | `0.7627` | `0.1099` | `0.9544` | `0.9994` |
| Complex Terrain Model | `0.4635` | `0.0651` | `0.8317` | `0.9939` |

`Deep IoU` is computed in the code on deeper layers only (`depth_threshold = 8`) with `|density| > 0.3`.

### Field-data example

Saved field-data outputs in `batch_test_results/summary.csv` report the following `Gzz` fit for the included example:

- `gzz_rmse = 9.0842`
- `gzz_mae = 6.2498`
- `gzz_corr = 0.8339`

The manuscript draft associates the field-data benchmark with the Vinton salt dome example.

## Practical Notes

- The current tracked scripts cover training and folder-based validation of `.vti` examples.
- Pre-generated example models and result images are already included, so you can inspect outputs without rerunning training.
- The repository contains more saved result artifacts than executable helper scripts; this README intentionally documents the files that are currently present and usable.

## Citation

If you use this repository in academic work, please cite the associated manuscript draft:

```text
Wenjin Chen, Shengwang Yu, Zhengfeng Jin, Lei Yi, and Xiao Gong,
"A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion".
```

If you need a formal bibliography entry later, update this section once the paper has a journal reference or DOI.
