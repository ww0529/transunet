# Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion

## Network Architecture

<p align="center">
  <img src="Network%20architecture.jpg" width="88%" alt="Network architecture of the proposed hybrid 2D-3D Transformer.">
</p>
<p align="center"><em>Schematic of the Hybrid 2D-3D Transformer proposed for 3D density gravity inversion.</em></p>

The proposed network follows the paper-centered hybrid 2D-3D design. Surface observations are first encoded in 2D, then lifted into a coarse 3D seed volume through the Channel-to-Depth Lifting module. A 3D Transformer bottleneck models global volumetric context, and cross-dimensional attention injects high-resolution 2D cues back into the 3D decoding pathway. By default, the input combines `Gz`, `Gzz`, and normalized depth encoding so that gravity anomaly, gravity-gradient response, and depth prior are all used jointly.

## Repository Structure

```text
.
|-- source code/
|   |-- train_code.py
|   |-- config.py
|   `-- data_preparation.py
|-- examples/
|   |-- example one/
|   |-- example two/
|   `-- example three/
|-- Field data example/
|   |-- Gzz.txt
|   `-- Field data example/
|       |-- 2D slice/
|       |-- 3D slice/
|       |-- 3D view of the predicted density model/
|       `-- gravity/
|-- folder_validation_results/
|-- best_model.pth
|-- folder_validation_test.py
|-- Network architecture.jpg
|-- README.md
`-- README_zh.md
```

The `source code/` directory contains the core implementation of the manuscript method, `examples/` and `Field data example/` contain synthetic and field-style demonstrations, and `folder_validation_test.py` provides the scripted validation workflow used to generate the reviewer-facing results.

## Description

This repository is the source-code release accompanying the manuscript `A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion(2).docx`. The method is designed for reconstructing three-dimensional density-contrast models from gravity (`Gz`) or vertical gravity-gradient (`Gzz`) observations, and the code in this repository follows the manuscript workflow as closely as possible.

The framework is centered on the paper methodology. A three-channel surface-to-volume input tensor is constructed from observed gravity, observed gravity gradient, and normalized depth encoding, producing an input of size `[B, 3, D, H, W]` with the default grid `(D, H, W) = (16, 32, 32)`. A 2D encoder first extracts high-level response features from the surface observations. These 2D features are then reorganized by the Channel-to-Depth Lifting module into a coarse 3D seed volume, refined by a 3D Transformer bottleneck, and fused with high-resolution 2D cues through cross-dimensional attention during 3D decoding.

The training objective is built around a composite loss including physics consistency, depth weighting, anomaly focusing, gradient-difference regularization, and edge enhancement. In the released implementation, these paper-centered terms remain the training core, while auxiliary regularizers are also included to stabilize optimization and sharpen reconstructed boundaries. The overall goal is to improve depth sensitivity, anomaly focusing, structural smoothness, and boundary delineation in both synthetic and field-style demonstrations.

This repository currently includes:

- the main training and model-definition files in [`source code/`](source%20code/)
- a pretrained checkpoint `best_model.pth`
- three bundled synthetic `.vti` examples under `examples/`
- the reviewer-oriented validation script [`folder_validation_test.py`](folder_validation_test.py)
- the field-style benchmark grid [`Field data example/Gzz.txt`](Field%20data%20example/Gzz.txt)
- exported result figures in `examples/`, `Field data example/`, and `folder_validation_results/`

## Installation

<code>pip install torch numpy scipy matplotlib vtk</code>

## Requirements

- torch
- numpy
- scipy
- matplotlib
- vtk

The first four packages are required for training and scripted validation. `vtk` is required for reading the bundled `.vti` models used in repository validation.

## Usage

The implementation is organized around the manuscript architecture in [`source code/train_code.py`](source%20code/train_code.py), the experiment configuration in [`source code/config.py`](source%20code/config.py), and the synthetic data generator in [`source code/data_preparation.py`](source%20code/data_preparation.py).

A representative set of default parameters from the current implementation is:

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

The manuscript-centered composite loss can be summarized as:

```text
L = lambda_phys * L_phys
  + lambda_depth * L_depth
  + lambda_focus * L_focus
  + lambda_gdl * L_gdl
  + lambda_edge * L_edge
```

In the tracked code, the corresponding default weights are:

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

To train the network from the repository root, update `save_dir` in [`source code/config.py`](source%20code/config.py) and run:

```bash
python "source code/train_code.py" --epochs 400 --batch_size 8 --lr 5e-4
```

The code also provides a physics-gradient verification mode:

```bash
python "source code/train_code.py" --verify-only
```

The paper and the released code focus on three reviewer-facing synthetic demonstrations: a simple compact anomaly inversion, a more complex positive-negative geological configuration, and a staircase-style structurally sharper case. The corresponding figures are already bundled in the repository and can be inspected directly without re-running training.

## Run Test Code

For scripted validation of the bundled `.vti` examples, use [`folder_validation_test.py`](folder_validation_test.py). This script loads `best_model.pth`, forward-models the network inputs, performs inference on all `.vti` models under `examples/`, and writes `metrics.json`, `summary.csv`, `summary.json`, NumPy arrays, and high-resolution figures to `folder_validation_results/`.

```bash
python folder_validation_test.py \
  --checkpoint best_model.pth \
  --models-dir examples \
  --output-dir folder_validation_results \
  --device auto
```

The current repository snapshot is centered on scripted validation and exported review figures. Field-style demonstration images are already included in the repository and can be inspected directly.

The following high-resolution validation figures are produced by the scripted validation pipeline and are included here for convenience.

<p align="center">
  <img src="folder_validation_results/vis_highres/epoch_0000_highres.png" width="32%" alt="High-resolution validation snapshot.">
  <img src="folder_validation_results/vis_highres/epoch_0001_highres.png" width="32%" alt="High-resolution validation snapshot.">
  <img src="folder_validation_results/vis_highres/epoch_0002_highres.png" width="32%" alt="High-resolution validation snapshot.">
</p>
<p align="center"><em>High-resolution validation snapshots exported by the validation workflow.</em></p>

### Synthetic Model Inversion

This example corresponds to the simple synthetic benchmark described in the manuscript, where the network is expected to reconstruct a compact anomaly with correct location, extent, and boundary shape.

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/isosurface/isosurface_true.png" width="49%" alt="True and predicted isosurfaces.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/isosurface/isosurface_pred.png" width="49%" alt="True and predicted isosurfaces.">
</p>
<p align="center"><em>True and predicted isosurfaces.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/voxel_model/voxel_true.png" width="49%" alt="True and predicted voxel models.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/voxel_model/voxel_pred.png" width="49%" alt="True and predicted voxel models.">
</p>
<p align="center"><em>True and predicted voxel models.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/true_2d/true_x_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/true_2d/true_y_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/true_2d/true_z_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
</p>
<p align="center"><em>True orthogonal slices along x, y, and z.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_2d/x_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_2d/y_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_2d/z_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
</p>
<p align="center"><em>Predicted orthogonal slices along x, y, and z.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/slice_3d/true_3d_x_slice.png" width="32%" alt="True 3D slice renderings.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/slice_3d/true_3d_y_slice.png" width="32%" alt="True 3D slice renderings.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/slice_3d/true_3d_z_slice.png" width="32%" alt="True 3D slice renderings.">
</p>
<p align="center"><em>True 3D slice renderings.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_slice_3d/3d_x_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_slice_3d/3d_y_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_slice_3d/3d_z_slice.png" width="32%" alt="Predicted 3D slice renderings.">
</p>
<p align="center"><em>Predicted 3D slice renderings.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/observed_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/forward_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
</p>
<p align="center"><em>Observed and forward-modelled Gzz maps.</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/residual.png" width="49%" alt="Residual map and residual histogram.">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/histogram.png" width="49%" alt="Residual map and residual histogram.">
</p>
<p align="center"><em>Residual map and residual histogram.</em></p>

### Complex Geological Structure

This example corresponds to the manuscript case with a more complex positive-negative anomaly configuration, where correct polarity recovery and limited cross-talk are especially important.

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/isosurface/isosurface_true.png" width="49%" alt="True and predicted isosurfaces.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/isosurface/isosurface_pred.png" width="49%" alt="True and predicted isosurfaces.">
</p>
<p align="center"><em>True and predicted isosurfaces.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/voxel_model/voxel_true.png" width="49%" alt="True and predicted voxel models.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/voxel_model/voxel_pred.png" width="49%" alt="True and predicted voxel models.">
</p>
<p align="center"><em>True and predicted voxel models.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_2d/true_x_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_2d/true_y_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_2d/true_z_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
</p>
<p align="center"><em>True orthogonal slices along x, y, and z.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_2d/x_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_2d/y_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_2d/z_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
</p>
<p align="center"><em>Predicted orthogonal slices along x, y, and z.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_3d/true_3d_x_slice.png" width="32%" alt="True 3D slice renderings.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_3d/true_3d_y_slice.png" width="32%" alt="True 3D slice renderings.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_3d/true_3d_z_slice.png" width="32%" alt="True 3D slice renderings.">
</p>
<p align="center"><em>True 3D slice renderings.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_3d/3d_x_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_3d/3d_y_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_3d/3d_z_slice.png" width="32%" alt="Predicted 3D slice renderings.">
</p>
<p align="center"><em>Predicted 3D slice renderings.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/observed_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/forward_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
</p>
<p align="center"><em>Observed and forward-modelled Gzz maps.</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/residual.png" width="49%" alt="Residual map and residual histogram.">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/histogram.png" width="49%" alt="Residual map and residual histogram.">
</p>
<p align="center"><em>Residual map and residual histogram.</em></p>

### Complex Terrain Model

This example corresponds to the staircase-style, more structurally intricate synthetic test in the manuscript, used to assess recovery of sharp offsets, layered boundaries, and more realistic piecewise-constant geometry.

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/isosurface/isosurface_true.png" width="49%" alt="True and predicted isosurfaces.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/isosurface/isosurface_pred.png" width="49%" alt="True and predicted isosurfaces.">
</p>
<p align="center"><em>True and predicted isosurfaces.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/voxel_model/voxel_true.png" width="49%" alt="True and predicted voxel models.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/voxel_model/voxel_pred.png" width="49%" alt="True and predicted voxel models.">
</p>
<p align="center"><em>True and predicted voxel models.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_2d/true_x_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_2d/true_y_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_2d/true_z_slice.png" width="32%" alt="True orthogonal slices along x, y, and z.">
</p>
<p align="center"><em>True orthogonal slices along x, y, and z.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_2d/x_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_2d/y_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_2d/z_slice.png" width="32%" alt="Predicted orthogonal slices along x, y, and z.">
</p>
<p align="center"><em>Predicted orthogonal slices along x, y, and z.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_slice_3d/true_3d_x_slice.png" width="32%" alt="True 3D slice renderings.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_slice_3d/true_3d_y_slice.png" width="32%" alt="True 3D slice renderings.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_slice_3d/true_3d_z_slice.png" width="32%" alt="True 3D slice renderings.">
</p>
<p align="center"><em>True 3D slice renderings.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_slice_3d/3d_x_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_slice_3d/3d_y_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_slice_3d/3d_z_slice.png" width="32%" alt="Predicted 3D slice renderings.">
</p>
<p align="center"><em>Predicted 3D slice renderings.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/observed_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/forward_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
</p>
<p align="center"><em>Observed and forward-modelled Gzz maps.</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/residual.png" width="49%" alt="Residual map and residual histogram.">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/histogram.png" width="49%" alt="Residual map and residual histogram.">
</p>
<p align="center"><em>Residual map and residual histogram.</em></p>

### Field Data Example

This section shows the field-style inversion figures included in `Field data example`. In contrast to the synthetic examples above, these figures illustrate the predicted density volume, slice views, and the agreement between observed and forward-modelled `Gzz` responses for the bundled field-style benchmark.

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/3D%20view%20of%20the%20predicted%20density%20model/isosurface_pred.png" width="49%" alt="Predicted density isosurface.">
  <img src="Field%20data%20example/Field%20data%20example/3D%20view%20of%20the%20predicted%20density%20model/voxel_pred.png" width="49%" alt="Predicted density voxel model.">
</p>
<p align="center"><em>Predicted density isosurface and voxel view.</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/2D%20slice/x_slice.png" width="32%" alt="Predicted orthogonal 2D slices.">
  <img src="Field%20data%20example/Field%20data%20example/2D%20slice/y_slice.png" width="32%" alt="Predicted orthogonal 2D slices.">
  <img src="Field%20data%20example/Field%20data%20example/2D%20slice/z_slice.png" width="32%" alt="Predicted orthogonal 2D slices.">
</p>
<p align="center"><em>Predicted 2D slices along x, y, and z.</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_x_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_y_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_z_slice.png" width="32%" alt="Predicted 3D slice renderings.">
</p>
<p align="center"><em>Predicted 3D slice renderings along x, y, and z.</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/gravity/observed_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
  <img src="Field%20data%20example/Field%20data%20example/gravity/forward_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
</p>
<p align="center"><em>Observed and forward-modelled Gzz maps.</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/gravity/residual.png" width="49%" alt="Residual map and residual histogram.">
  <img src="Field%20data%20example/Field%20data%20example/gravity/histogram.png" width="49%" alt="Residual map and residual histogram.">
</p>
<p align="center"><em>Residual distribution and residual histogram for the field-style Gzz fit.</em></p>

## Related Publications

If you use this code in academic work, please primarily cite the accompanying manuscript:

Chen, W., Yu, S., Jin, Z., Yi, L., and Gong, X., `A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion`.

The manuscript is also positioned in the context of classical and modern gravity-inversion studies, including:

Li, Y., and Oldenburg, D. W., 1998, 3-D inversion of gravity data: Geophysics, 63(1), 109-119.

Zhdanov, M. S., Robert, E., and Souvik, M., 2004, Three-dimensional regularized focusing inversion of gravity gradient tensor component data: Geophysics, 69(4), 925-937.

Chen, W., Tenzer, R., Tan, X., and Zhao, S., 2024, An Adaptive Conjugate Gradient Least-Squares Regularization (ACGLSR) Method for 3D Gravity Density Inversion: Pure and Applied Geophysics, 181(1), 203-218.

## Acknowledgements

According to the manuscript draft, this work was supported by the National Natural Science Foundation of China, the Strategic Priority Research Program of the Chinese Academy of Sciences, Jiangxi province key-laboratory opening funding, and the Jiangxi Province Key Research and Development Program.

## License

This repository snapshot is currently provided as the source-code package for academic review of the accompanying manuscript. Please contact the authors for reuse and redistribution details until a formal standalone license file is added to the repository.
