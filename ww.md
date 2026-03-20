# A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion
## Network Architecture

<p align="center">
  <img src="Network%20architecture.jpg" width="88%" alt="The architecture of the proposed deep learning convolutional neural network.">
</p>
<p align="center"><em>The architecture of the proposed deep learning convolutional neural network.</em></p>

This network is the hybrid 2D-3D Transformer proposed in the manuscript. It first extracts 2D features from surface gravity responses, then lifts the 2D features into an initial 3D volume through the Channel-to-Depth Lifting module. A 3D Transformer is subsequently used to model global spatial correlations, and cross-dimensional attention is incorporated during decoding to progressively recover the 3D density distribution. By default, the input is composed of `Gz`, `Gzz`, and normalized depth encoding, so that gravity anomalies, gravity gradients, and depth prior information can all be utilized simultaneously.

## Repository Structure

```text
.
├── source code/
│   ├── train_code.py
│   ├── config.py
│   └── data_preparation.py
├── examples/
│   ├── example one/
│   ├── example two/
│   └── example three/
├── Field data example/
│   ├── Gzz.txt
│   └── Field data example/
│       ├── 2D slice/
│       ├── 3D slice/
│       ├── 3D view of the predicted density model/
│       └── gravity/
├── test_code/
├── best_model.pth
├── test_code.py
└── README.md
```
Here, source code/ contains the core training and data-generation code corresponding to the manuscript method, examples/ and Field data example/ contain real-data examples, and  test_code.py correspond to the scripted validation , respectively. ```
## Description

This repository is the source-code release corresponding to the manuscript `A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion(2).docx`. The method is designed for reconstructing three-dimensional density-contrast models from gravity anomaly (`Gz`) or vertical gravity-gradient (`Gzz`) observations.

The proposed framework is centered on the method presented in the manuscript. The model first constructs a three-channel surface-to-volume input tensor from observed gravity, observed gravity gradient, and normalized depth encoding. The default input size is `[B, 3, D, H, W]`, where the default grid size is `(D, H, W) = (16, 32, 32)`. Then, a 2D encoder extracts high-level response features from the surface observations, and the Channel-to-Depth Lifting module reorganizes these 2D features into a coarse 3D seed volume. A 3D Transformer bottleneck further models the volumetric representation, and cross-dimensional attention introduces high-resolution 2D details during the 3D decoding process.

The training objective is a composite loss consisting of physics-consistency, depth-weighted, focal, gradient-difference, and edge-enhancement terms. The code also includes several auxiliary regularizers to improve optimization stability and sharpen reconstructed boundaries. The overall goal is to improve deep sensitivity, anomaly focusing, structural smoothness, and boundary delineation, so that the geometry of subsurface anomalous bodies can be recovered more effectively.

This repository currently includes:

- the main training and model-definition code in [`source code/`](source%20code/)
- the best model checkpoint `best_model.pth` obtained from a 400-epoch training run
- three synthetic `.vti` examples and their result figures in `examples/`
- the validation script [`test_code.py`](test_code.py) for testing trained results
- the real data file `Field data example/Gzz.txt` and the corresponding inversion figures
- evaluation results stored in `test_code/`

## Installation

<code>pip install torch numpy scipy matplotlib vtk</code>

## Requirements

- torch
- numpy
- scipy
- matplotlib
- vtk

The first four packages are used for training and scripted validation. `vtk` is required for reading the bundled `.vti` models in the repository.

## Usage

The implementation is built around the network architecture described in the manuscript. The core files include the model and training workflow in [`source code/train_code.py`](source%20code/train_code.py), the configuration in [`source code/config.py`](source%20code/config.py), and the synthetic data generator in [`source code/data_preparation.py`](source%20code/data_preparation.py).

A representative set of default parameters in the current implementation is shown below:

```python
config = V4Config(
    data_mode='joint',
    grid_shape=(16, 32, 32),
    dx=100.0,
    dz=100.0,
    encoder_channels=(32, 64, 128, 256),
    decoder_channels=(128, 64, 32),
    lifting_channels=64,
    num_transformer_layers=4,
    num_heads=8,
    lr=5e-4,
    batch_size=8,
    epochs=400,
    steps_per_epoch=500,
)
```

The composite loss can be summarized as:

```text
L = lambda_phys * L_phys
  + lambda_depth * L_depth
  + lambda_focus * L_focus
  + lambda_gdl * L_gdl
  + lambda_edge * L_edge
```

The corresponding default weights in the current code are:

```python
w_depth = 1.0
w_focus = 1.0
w_gdl = 1.5
w_physics = 0.3
w_edge = 0.5
w_l1 = 0.05
w_morph = 0.05
w_boundary = 0.5
depth_beta = 2.0
focus_beta = 10.0
```

To train the network from the repository root, first modify `save_dir` in [`source code/config.py`](source%20code/config.py), and then run:

```bash
python "source code/train_code.py" --epochs 400 --batch_size 8 --lr 5e-4
```

The code also provides a physics-gradient verification mode:

```bash
python "source code/train_code.py" --verify-only
```

Both the manuscript and the code focus on three typical examples: a Synthetic example one-prism model, a Synthetic example one-two prisms model
, and a Synthetic example one-two staircase models. The corresponding figures are already bundled in the repository and can be inspected directly without retraining.

## Run Test Codes

To perform scripted validation on the bundled `.vti` examples, use [`test_code.py`](test_code.py). This script loads `best_model.pth`, performs forward modelling to construct the required network input, runs inference on all `.vti` models under `examples/`, and writes `metrics.json`, `summary.csv`, `summary.json`, NumPy arrays, and high-resolution figures to `test_code/`.

```bash
python test_code.py \
  --checkpoint best_model.pth \
  --models-dir examples \
  --output-dir test_code \
  --device auto
```

The following figures are the high-resolution outputs produced by the scripted validation workflow. These images are already included in the repository for direct inspection.

<p align="center">
  <img src="test_code/vis_highres/Synthetic%20example%20one-prism%20model.png" width="32%" alt="Scripted validation figure for the one-prism example.">
  <img src="test_code/vis_highres/Synthetic%20example%20one-two%20staircase%20models.png" width="32%" alt="Scripted validation figure for the two-staircase example.">
  <img src="test_code/vis_highres/Synthetic%20example%20one-two%20prisms%20model.png" width="32%" alt="Scripted validation figure for the two-prism example.">
</p>
<p align="center"><em>High-resolution scripted validation figures for the three bundled synthetic examples.</em></p>

### Synthetic example one-prism model

This section follows Section 3.3 of the manuscript. The benchmark contains a single rectangular prism at `x = 800-2400 m`, `y = 800-2400 m`, and `z = 500-1200 m`, with a density contrast of `0.5 kg/m^3` in an otherwise zero-density background. As discussed in the paper, the predicted model preserves the prism location and overall morphology with only minor boundary smoothing.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/isosurface/isosurface_true.png" width="100%" alt="Fig. 5a synthetic prism model.">
      <br><em>(a) Synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/isosurface/isosurface_pred.png" width="100%" alt="Fig. 5b predicted prism model.">
      <br><em>(b) Predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em> 3D view of the prism model: (a) synthetic model and (b) predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/slice_3d/true_3d_x_slice.png" width="100%" alt="Fig. 6a 3D view of the synthetic model along x = 2000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/pred_slice_3d/3d_x_slice.png" width="100%" alt="Fig. 6b 3D view of the predicted model along x = 2000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/true_2d/true_x_slice.png" width="100%" alt="Fig. 6c 2D view of the synthetic model along x = 2000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/pred_2d/x_slice.png" width="100%" alt="Fig. 6d 2D view of the predicted model along x = 2000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em> Slice of the synthetic and predicted density models along x = 2000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/slice_3d/true_3d_y_slice.png" width="100%" alt="Fig. 7a 3D view of the synthetic model along y = 2000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/pred_slice_3d/3d_y_slice.png" width="100%" alt="Fig. 7b 3D view of the predicted model along y = 2000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/true_2d/true_y_slice.png" width="100%" alt="Fig. 7c 2D view of the synthetic model along y = 2000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/pred_2d/y_slice.png" width="100%" alt="Fig. 7d 2D view of the predicted model along y = 2000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em> Slice of the synthetic and predicted density models along y = 2000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/slice_3d/true_3d_z_slice.png" width="100%" alt="Fig. 8a 3D view of the synthetic model along z = 1000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/pred_slice_3d/3d_z_slice.png" width="100%" alt="Fig. 8b 3D view of the predicted model along z = 1000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/true_2d/true_z_slice.png" width="100%" alt="Fig. 8c 2D view of the synthetic model along z = 1000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/pred_2d/z_slice.png" width="100%" alt="Fig. 8d 2D view of the predicted model along z = 1000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along z = 1000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

The gravity-gradient comparison also follows the manuscript: the forward-modelled `Gzz` field from the predicted density model closely matches the observed field, and the residuals remain small and approximately zero-centered.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/gravity/observed_gzz.png" width="100%" alt="Fig. 9a observed gravity gradient with 5 percent Gaussian noise added.">
      <br><em>(a) Observed gravity gradient with 5% Gaussian noise added</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/gravity/forward_gzz.png" width="100%" alt="Fig. 9b predicted gravity gradient.">
      <br><em>(b) Predicted gravity gradient</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/gravity/residual.png" width="100%" alt="Fig. 9c gravity gradient difference between the observed and predicted data.">
      <br><em>(c) Gravity gradient difference</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20one/Synthetic%20example%20one-prism%20model/gravity/histogram.png" width="100%" alt="Fig. 9d histogram of the gravity gradient differences.">
      <br><em>(d) Histogram of the gravity gradient differences</em>
    </td>
  </tr>
</table>
<p align="center"><em> Observed and predicted gravity gradient data. (a) Observed gravity gradient with 5% Gaussian noise added; (b) predicted gravity gradient; (c) gravity gradient difference between the observed and predicted data; and (d) histogram of the gravity gradient differences.</em></p>

### Synthetic example one-two prisms model

This section follows the two-prism experiment in the manuscript. The synthetic model contains a positive prism with density contrast `0.5 kg/m^3` at `x = 600-1200 m`, `y = 600-1200 m`, `z = 500-1200 m`, and a negative prism with density contrast `-0.5 kg/m^3` at `x = 1800-2300 m`, `y = 600-1200 m`, `z = 500-1200 m`. The paper uses this example to verify polarity recovery, spatial separation, and suppression of cross-talk between adjacent anomalies.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/isosurface/isosurface_true.png" width="100%" alt="Fig. 10a synthetic two-prism model.">
      <br><em>(a) Synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/isosurface/isosurface_pred.png" width="100%" alt="Fig. 10b predicted two-prism model.">
      <br><em>(b) Predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>3D view of the two prisms model: (a) synthetic model and (b) predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/true_slice_3d/true_3d_x_slice.png" width="100%" alt="Fig. 11a 3D view of the synthetic model along x = 1000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/pred_slice_3d/3d_x_slice.png" width="100%" alt="Fig. 11b 3D view of the predicted model along x = 1000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/true_slice_2d/true_x_slice.png" width="100%" alt="Fig. 11c 2D view of the synthetic model along x = 1000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/pred_slice_2d/x_slice.png" width="100%" alt="Fig. 11d 2D view of the predicted model along x = 1000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along x = 1000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/true_slice_3d/true_3d_y_slice.png" width="100%" alt="Fig. 12a 3D view of the synthetic model along y = 1500 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/pred_slice_3d/3d_y_slice.png" width="100%" alt="Fig. 12b 3D view of the predicted model along y = 1500 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/true_slice_2d/true_y_slice.png" width="100%" alt="Fig. 12c 2D view of the synthetic model along y = 1500 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/pred_slice_2d/y_slice.png" width="100%" alt="Fig. 12d 2D view of the predicted model along y = 1500 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along y = 1500 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/true_slice_3d/true_3d_z_slice.png" width="100%" alt="Fig. 13a 3D view of the synthetic model along z = 1000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/pred_slice_3d/3d_z_slice.png" width="100%" alt="Fig. 13b 3D view of the predicted model along z = 1000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/true_slice_2d/true_z_slice.png" width="100%" alt="Fig. 13c 2D view of the synthetic model along z = 1000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/pred_slice_2d/z_slice.png" width="100%" alt="Fig. 13d 2D view of the predicted model along z = 1000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along z = 1000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

The manuscript emphasizes that the recovered `Gzz` response reproduces both anomaly polarity and interference patterns without significant cross-talk, and the residual histogram remains centered near zero.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/gravity/observed_gzz.png" width="100%" alt="Fig. 14a observed gravity gradient with 5 percent Gaussian noise added.">
      <br><em>(a) Observed gravity gradient with 5% Gaussian noise added</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/gravity/forward_gzz.png" width="100%" alt="Fig. 14b predicted gravity gradient.">
      <br><em>(b) Predicted gravity gradient</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/gravity/residual.png" width="100%" alt="Fig. 14c gravity gradient difference between the observed and predicted data.">
      <br><em>(c) Gravity gradient difference</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20two/Synthetic%20example%20one-two%20prisms%20model/gravity/histogram.png" width="100%" alt="Fig. 14d histogram of the gravity gradient differences.">
      <br><em>(d) Histogram of the gravity gradient differences</em>
    </td>
  </tr>
</table>
<p align="center"><em> Observed and predicted gravity gradient data. (a) Observed gravity gradient with 5% Gaussian noise added; (b) predicted gravity gradient; (c) gravity gradient difference between the observed and predicted data; and (d) histogram of the gravity gradient differences.</em></p>

### Synthetic example one-two staircase models

This section follows the structurally more complex two-staircase case from the manuscript. The paper uses this model to test whether the network can recover step-like interfaces, sharp geometric breaks, and piecewise-constant density structures with good positional accuracy across 3D views and orthogonal slices.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/isosurface/isosurface_true.png" width="100%" alt="Fig. 15a synthetic two-staircase model.">
      <br><em>(a) Synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/isosurface/isosurface_pred.png" width="100%" alt="Fig. 15b predicted two-staircase model.">
      <br><em>(b) Predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>3D view of the two staircase model: (a) synthetic model and (b) predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/true_slice_3d/true_3d_x_slice.png" width="100%" alt="Fig. 16a 3D view of the synthetic model along x = 2000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/pred_slice_3d/3d_x_slice.png" width="100%" alt="Fig. 16b 3D view of the predicted model along x = 2000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/true_2d/true_x_slice.png" width="100%" alt="Fig. 16c 2D view of the synthetic model along x = 2000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/pred_2d/x_slice.png" width="100%" alt="Fig. 16d 2D view of the predicted model along x = 2000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along x = 2000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/true_slice_3d/true_3d_y_slice.png" width="100%" alt="Fig. 17a 3D view of the synthetic model along y = 2000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/pred_slice_3d/3d_y_slice.png" width="100%" alt="Fig. 17b 3D view of the predicted model along y = 2000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/true_2d/true_y_slice.png" width="100%" alt="Fig. 17c 2D view of the synthetic model along y = 2000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/pred_2d/y_slice.png" width="100%" alt="Fig. 17d 2D view of the predicted model along y = 2000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along y = 2000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/true_slice_3d/true_3d_z_slice.png" width="100%" alt="Fig. 18a 3D view of the synthetic model along z = 1000 m.">
      <br><em>(a) 3D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/pred_slice_3d/3d_z_slice.png" width="100%" alt="Fig. 18b 3D view of the predicted model along z = 1000 m.">
      <br><em>(b) 3D view of the predicted model</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/true_2d/true_z_slice.png" width="100%" alt="Fig. 18c 2D view of the synthetic model along z = 1000 m.">
      <br><em>(c) 2D view of the synthetic model</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/pred_2d/z_slice.png" width="100%" alt="Fig. 18d 2D view of the predicted model along z = 1000 m.">
      <br><em>(d) 2D view of the predicted model</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slice of the synthetic and predicted density models along z = 1000 m. (a) 3D view of the synthetic model; (b) 3D view of the predicted model; (c) 2D view of the synthetic model; (d) 2D view of the predicted model.</em></p>

The manuscript notes that the predicted `Gzz` field preserves the dominant staircase anomaly pattern, while the residual map lacks coherent staircase-aligned artifacts and the histogram stays nearly symmetric around zero.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/gravity/observed_gzz.png" width="100%" alt="Fig. 19a observed gravity gradient with 5 percent Gaussian noise added.">
      <br><em>(a) Observed gravity gradient with 5% Gaussian noise added</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/gravity/forward_gzz.png" width="100%" alt="Fig. 19b predicted gravity gradient.">
      <br><em>(b) Predicted gravity gradient</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/gravity/residual.png" width="100%" alt="Fig. 19c gravity gradient difference between the observed and predicted data.">
      <br><em>(c) Gravity gradient difference</em>
    </td>
    <td align="center" width="50%">
      <img src="examples/example%20three/Synthetic%20example%20one-two%20staircase%20models/gravity/histogram.png" width="100%" alt="Fig. 19d histogram of the gravity gradient differences.">
      <br><em>(d) Histogram of the gravity gradient differences</em>
    </td>
  </tr>
</table>
<p align="center"><em>Observed and predicted gravity gradient data. (a) Observed gravity gradient with 5% Gaussian noise added; (b) predicted gravity gradient; (c) gravity gradient difference between the observed and predicted data; and (d) histogram of the gravity gradient differences.</em></p>

### Field data example

This section follows Section 3.5 of the manuscript and keeps the same figure order as the paper. The field example uses airborne gravity-gradient data acquired in 2008 over the Vinton salt dome in Louisiana, USA, by Bell Geospace. Following the interpretation in the manuscript, the predicted density model reveals a shallow high-density cap-rock with an extent of about `1500 m` in the north-south direction, about `1600 m` in the east-west direction, and a depth range of roughly `260-700 m`.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/gravity/observed_gzz.png" width="100%" alt="Fig. 20a measured airborne gravity gradient data.">
      <br><em>(a) Measured airborne gravity gradient data</em>
    </td>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/gravity/forward_gzz.png" width="100%" alt="Fig. 20b predicted airborne gravity gradient data.">
      <br><em>(b) Predicted airborne gravity gradient data</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/gravity/residual.png" width="100%" alt="Fig. 20c airborne gravity gradient differences between the measured and predicted data.">
      <br><em>(c) Airborne gravity gradient differences</em>
    </td>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/gravity/histogram.png" width="100%" alt="Fig. 20d histogram of the gravity gradient differences.">
      <br><em>(d) Histogram of the gravity gradient differences</em>
    </td>
  </tr>
</table>
<p align="center"><em>Measured and predicted airborne gravity gradient fields. (a) Measured airborne gravity gradient data; (b) predicted airborne gravity gradient data; (c) airborne gravity gradient differences between the measured and predicted data; and (d) histogram of the gravity gradient differences.</em></p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_x_slice.png" width="100%" alt="Fig. 21a 3D slice along x = 2000 m.">
      <br><em>(a) 3D slice along x = 2000 m</em>
    </td>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/2D%20slice/x_slice.png" width="100%" alt="Fig. 21b 2D slice along x = 2000 m.">
      <br><em>(b) 2D slice along x = 2000 m</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_y_slice.png" width="100%" alt="Fig. 21c 3D slice along y = 2000 m.">
      <br><em>(c) 3D slice along y = 2000 m</em>
    </td>
    <td align="center" width="50%">
      <img src="Field%20data%20example/Field%20data%20example/2D%20slice/y_slice.png" width="100%" alt="Fig. 21d 2D slice along y = 2000 m.">
      <br><em>(d) 2D slice along y = 2000 m</em>
    </td>
  </tr>
</table>
<p align="center"><em>Slices of the predicted density model. (a) 3D slice along x = 2000 m; (b) 2D slice along x = 2000 m; (c) 3D slice along y = 2000 m; and (d) 2D slice along y = 2000 m.</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/3D%20view%20of%20the%20predicted%20density%20model/isosurface_pred.png.">
</p>
<p align="center"><em>3D view of the predicted density model.</em></p>
