<img src="Field%20data%20example/predicted_density_model.png" width="520">

## Description

This repository is the source-code release accompanying the manuscript `A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion(2).docx`. The method is designed for reconstructing three-dimensional density-contrast models from gravity (`Gz`) or vertical gravity-gradient (`Gzz`) observations, with the code in this repository implementing the manuscript workflow as closely as possible.

The proposed framework follows the paper-centered design. A three-channel surface-to-volume input tensor is constructed from observed gravity, observed gravity gradient, and normalized depth encoding, resulting in an input of size `[B, 3, D, H, W]` with the default grid `(D, H, W) = (16, 32, 32)`. A 2D encoder first extracts high-level response features from the surface observations. These 2D features are then reorganized by the Channel-to-Depth Lifting module into a coarse 3D seed volume, refined by a 3D Transformer bottleneck, and fused with high-resolution 2D cues through cross-dimensional attention during 3D decoding.

The manuscript further emphasizes a composite loss function consisting of physics-consistency, depth-weighted, focal, gradient-difference, and edge-enhancement terms. In the released implementation, these paper-centered losses remain the core of training, while auxiliary regularizers are also present in the code to stabilize optimization and sharpen boundaries. The overall goal is to improve depth sensitivity, anomaly focusing, structural smoothness, and boundary delineation for reviewer-facing synthetic and field-style demonstrations.

This repository currently includes:

- the main training and model definition files in [`source code/`](source%20code/)
- a pretrained checkpoint `best_model.pth`
- three bundled synthetic `.vti` examples in `examples/`
- a reviewer-oriented validation script `folder_validation_test.py`
- an interactive testing and visualization program `test_code.py`
- a field-style benchmark grid in `Field data example/Gzz.txt`
- exported result figures in `examples/`, `Field data example/`, and `folder_validation_results/`

## Installation

<code>pip install torch numpy scipy matplotlib vtk PySide6 pyvista pyvistaqt</code>

## Requirements

- torch
- numpy
- scipy
- matplotlib
- vtk
- PySide6
- pyvista
- pyvistaqt

The first four packages are required for training and scripted validation. `vtk` is required for reading the bundled `.vti` models. `PySide6`, `pyvista`, and `pyvistaqt` are used by `test_code.py` for interactive visual inspection.

## Usage

The codebase is organized around the manuscript architecture implemented in [`source code/train_code.py`](source%20code/train_code.py), the configuration in [`source code/config.py`](source%20code/config.py), and the synthetic data generator in [`source code/data_preparation.py`](source%20code/data_preparation.py).

A representative set of default parameters from the current implementation is as follows:

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

The paper and the code both focus on three reviewer-facing synthetic demonstrations: a simple single rectangular-prism inversion, a more complicated positive-negative geological configuration, and a structurally sharper staircase-style terrain case. The corresponding figures are already bundled in the repository and can be inspected directly without re-running training.

## Run Test Codes

For scripted validation of the bundled `.vti` examples, use [`folder_validation_test.py`](folder_validation_test.py). This script loads `best_model.pth`, forward-models the input responses needed by the network, performs inference on all `.vti` models under `examples/`, and writes `metrics.json`, `summary.csv`, `summary.json`, NumPy arrays, and high-resolution review figures to `folder_validation_results/`.

```bash
python folder_validation_test.py \
  --checkpoint best_model.pth \
  --models-dir examples \
  --output-dir folder_validation_results \
  --device auto
```

For interactive testing, reviewer inspection, and field-style visualization, run [`test_code.py`](test_code.py):

```bash
python test_code.py
```

The interactive program can load either:

- `Field data example/Gzz.txt`
- the bundled `.vti` models under `examples/`
- a checkpoint such as `best_model.pth`

It then performs model loading, input construction, prediction, 2D slice display, and 3D visualization for qualitative review.

The following high-resolution validation figures are produced by the scripted validation pipeline and are included here for convenience.

<p align="center">
  <img src="folder_validation_results/vis_highres/epoch_0000_highres.png" width="32%" alt="High-resolution validation snapshots exported by HighResVisualizer.">
  <img src="folder_validation_results/vis_highres/epoch_0001_highres.png" width="32%" alt="High-resolution validation snapshots exported by HighResVisualizer.">
  <img src="folder_validation_results/vis_highres/epoch_0002_highres.png" width="32%" alt="High-resolution validation snapshots exported by HighResVisualizer.">
</p>
<p align="center"><em>High-resolution validation snapshots exported by HighResVisualizer.</em></p>

The field-style benchmark density result bundled with the repository is shown at the top of this README. In the manuscript, the field benchmark is associated with the 2008 airborne gravity-gradient data over the Vinton salt dome in Louisiana, USA.

### Synthetic Model Inversion

This example corresponds to the simple synthetic benchmark described in the manuscript, where the network is expected to reconstruct a compact anomaly with correct position, extent, and boundary shape.

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

## Related publications

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
