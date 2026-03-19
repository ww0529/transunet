## 简介

本仓库是论文稿件 `A Hybrid 2D-3D Transformer Network with Channel-to-Depth Lifting for 3D Density Gravity Inversion(2).docx` 对应的源代码发布版本。该方法面向重力异常 `Gz` 或垂向重力梯度 `Gzz` 观测的三维密度对比模型重建。
所提出的框架以论文方法为核心。模型首先将观测重力、观测重力梯度以及归一化深度编码构造成三通道的表面到体输入张量，默认输入尺寸为 `[B, 3, D, H, W]`，其中默认网格大小为 `(D, H, W) = (16, 32, 32)`。随后，2D 编码器先从表面观测中提取高层响应特征，再通过 Channel-to-Depth Lifting 模块将这些 2D 特征重组为粗略的 3D 初始体，经由 3D Transformer 瓶颈进一步建模，并在 3D 解码过程中结合跨维注意力融入高分辨率 2D 细节信息。

训练代码的损失函数由物理一致性项、深度加权项、聚焦项、梯度差分项以及边缘增强项组成的复合损失函数。同时代码中还加入了一些辅助正则项，以提高优化稳定性并增强边界清晰度。整体目标是提升深部敏感性、异常聚焦能力、结构平滑性以及边界刻画能力，从而更好地反演出地下异常体的形状。

当前仓库主要包含：

- 位于 [`source code/`](source%20code/) 中的主要训练与模型定义代码
- 训练一轮400 epoch 所生成的最好的模型权重 `best_model.pth`
- `examples/` 中的三个示例合成 `.vti` 模型，以及结果图片
- 测试训练结果的验证脚本 [`folder_validation_test.py`](folder_validation_test.py)
- 位于 `Field data example/Gzz.txt` 的真实数据，以及结果反演图片
- 分布在`folder_validation_results/` 中的测试结果

## 安装

<code>pip install torch numpy scipy matplotlib vtk</code>

## 依赖环境

- torch
- numpy
- scipy
- matplotlib
- vtk
前四个包用于训练和脚本化验证。`vtk` 用于读取仓库中自带的 `.vti` 模型。
## 使用说明

代码主体围绕论文中的网络结构展开实现，核心文件包括 [`source code/train_code.py`](source%20code/train_code.py) 中的模型与训练流程、[`source code/config.py`](source%20code/config.py) 中的配置，以及 [`source code/data_preparation.py`](source%20code/data_preparation.py) 中的合成数据生成器。

当前实现中的一组代表性默认参数如下：

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

复合损失函数可概括为：

```text
L = lambda_phys * L_phys
  + lambda_depth * L_depth
  + lambda_focus * L_focus
  + lambda_gdl * L_gdl
  + lambda_edge * L_edge
```

在当前代码实现中，对应的默认权重为：

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

若要在仓库根目录下训练网络，请先在 [`source code/config.py`](source%20code/config.py) 中修改 `save_dir`，然后运行：

```bash
python "source code/train_code.py" --epochs 400 --batch_size 8 --lr 5e-4
```

代码还提供了物理梯度验证模式：

```bash
python "source code/train_code.py" --verify-only
```

论文与代码都重点展示了三个典型案例：简单单长方体反演案例、更复杂的正负异常地质构型案例，以及结构边界更尖锐的阶梯式地形案例。对应图片已经随仓库一并提供，无需重新训练即可直接查看。

## 运行测试代码

若要对仓库自带的 `.vti` 示例进行脚本化验证，请使用 [`folder_validation_test.py`](folder_validation_test.py)。该脚本会加载 `best_model.pth`，根据网络所需输入执行正演，随后对 `examples/` 目录下的全部 `.vti` 模型进行推理，并将 `metrics.json`、`summary.csv`、`summary.json`、NumPy 数组以及高分辨率图输出到 `folder_validation_results/`。

```bash
python folder_validation_test.py \
  --checkpoint best_model.pth \
  --models-dir examples \
  --output-dir folder_validation_results \
  --device auto
```

下面展示的是脚本化验证流程输出的高分辨率验证图，这些图片已包含在仓库中，便于直接查看。

<p align="center">
  <img src="folder_validation_results/vis_highres/epoch_0000_highres.png" width="32%" alt="HighResVisualizer 导出的高分辨率验证快照。">
  <img src="folder_validation_results/vis_highres/epoch_0001_highres.png" width="32%" alt="HighResVisualizer 导出的高分辨率验证快照。">
  <img src="folder_validation_results/vis_highres/epoch_0002_highres.png" width="32%" alt="HighResVisualizer 导出的高分辨率验证快照。">
</p>
<p align="center"><em>由 HighResVisualizer 导出的高分辨率验证。</em></p>

### Synthetic Model Inversion

该案例对应论文中的简单合成基准，目标是让网络重建一个位置、范围和边界都尽可能准确的紧凑异常体。

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/isosurface/isosurface_true.png" width="49%" alt="真实与预测等值面。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/isosurface/isosurface_pred.png" width="49%" alt="真实与预测等值面。">
</p>
<p align="center"><em>真实与预测等值面。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/voxel_model/voxel_true.png" width="49%" alt="真实与预测体素模型。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/voxel_model/voxel_pred.png" width="49%" alt="真实与预测体素模型。">
</p>
<p align="center"><em>真实与预测体素模型。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/true_2d/true_x_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/true_2d/true_y_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/true_2d/true_z_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
</p>
<p align="center"><em>真实 x、y、z 三向正交切片。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_2d/x_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_2d/y_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_2d/z_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
</p>
<p align="center"><em>预测 x、y、z 三向正交切片。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/slice_3d/true_3d_x_slice.png" width="32%" alt="真实 3D 切片渲染图。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/slice_3d/true_3d_y_slice.png" width="32%" alt="真实 3D 切片渲染图。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/slice_3d/true_3d_z_slice.png" width="32%" alt="真实 3D 切片渲染图。">
</p>
<p align="center"><em>真实 3D 切片渲染图。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_slice_3d/3d_x_slice.png" width="32%" alt="预测 3D 切片渲染图。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_slice_3d/3d_y_slice.png" width="32%" alt="预测 3D 切片渲染图。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/pred_slice_3d/3d_z_slice.png" width="32%" alt="预测 3D 切片渲染图。">
</p>
<p align="center"><em>预测 3D 切片渲染图。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/observed_gzz.png" width="49%" alt="观测与正演 Gzz 图。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/forward_gzz.png" width="49%" alt="观测与正演 Gzz 图。">
</p>
<p align="center"><em>观测与正演 Gzz 图。</em></p>

<p align="center">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/residual.png" width="49%" alt="残差图与残差直方图。">
  <img src="examples/example%20one/Synthetic%20Model%20Inversion/gravity/histogram.png" width="49%" alt="残差图与残差直方图。">
</p>
<p align="center"><em>残差图与残差直方图。</em></p>

### Complex Geological Structure

该案例对应论文中更复杂的正负异常组合场景，重点考察模型对异常极性恢复以及不同异常体之间串扰抑制的能力。

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/isosurface/isosurface_true.png" width="49%" alt="真实与预测等值面。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/isosurface/isosurface_pred.png" width="49%" alt="真实与预测等值面。">
</p>
<p align="center"><em>真实与预测等值面。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/voxel_model/voxel_true.png" width="49%" alt="真实与预测体素模型。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/voxel_model/voxel_pred.png" width="49%" alt="真实与预测体素模型。">
</p>
<p align="center"><em>真实与预测体素模型。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_2d/true_x_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_2d/true_y_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_2d/true_z_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
</p>
<p align="center"><em>真实 x、y、z 三向正交切片。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_2d/x_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_2d/y_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_2d/z_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
</p>
<p align="center"><em>预测 x、y、z 三向正交切片。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_3d/true_3d_x_slice.png" width="32%" alt="真实 3D 切片渲染图。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_3d/true_3d_y_slice.png" width="32%" alt="真实 3D 切片渲染图。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/true_slice_3d/true_3d_z_slice.png" width="32%" alt="真实 3D 切片渲染图。">
</p>
<p align="center"><em>真实 3D 切片渲染图。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_3d/3d_x_slice.png" width="32%" alt="预测 3D 切片渲染图。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_3d/3d_y_slice.png" width="32%" alt="预测 3D 切片渲染图。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/pred_slice_3d/3d_z_slice.png" width="32%" alt="预测 3D 切片渲染图。">
</p>
<p align="center"><em>预测 3D 切片渲染图。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/observed_gzz.png" width="49%" alt="观测与正演 Gzz 图。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/forward_gzz.png" width="49%" alt="观测与正演 Gzz 图。">
</p>
<p align="center"><em>观测与正演 Gzz 图。</em></p>

<p align="center">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/residual.png" width="49%" alt="残差图与残差直方图。">
  <img src="examples/example%20two/Complex%20Geological%20Structure/gravity/histogram.png" width="49%" alt="残差图与残差直方图。">
</p>
<p align="center"><em>残差图与残差直方图。</em></p>

### Complex Terrain Model

该案例对应论文中更具结构复杂性的阶梯式地形合成测试，用于检验模型对尖锐错断、层状边界以及更贴近分段常数地质几何体的恢复能力。

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/isosurface/isosurface_true.png" width="49%" alt="真实与预测等值面。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/isosurface/isosurface_pred.png" width="49%" alt="真实与预测等值面。">
</p>
<p align="center"><em>真实与预测等值面。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/voxel_model/voxel_true.png" width="49%" alt="真实与预测体素模型。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/voxel_model/voxel_pred.png" width="49%" alt="真实与预测体素模型。">
</p>
<p align="center"><em>真实与预测体素模型。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_2d/true_x_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_2d/true_y_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_2d/true_z_slice.png" width="32%" alt="真实 x、y、z 三向正交切片。">
</p>
<p align="center"><em>真实 x、y、z 三向正交切片。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_2d/x_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_2d/y_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_2d/z_slice.png" width="32%" alt="预测 x、y、z 三向正交切片。">
</p>
<p align="center"><em>预测 x、y、z 三向正交切片。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_slice_3d/true_3d_x_slice.png" width="32%" alt="真实 3D 切片渲染图。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_slice_3d/true_3d_y_slice.png" width="32%" alt="真实 3D 切片渲染图。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/true_slice_3d/true_3d_z_slice.png" width="32%" alt="真实 3D 切片渲染图。">
</p>
<p align="center"><em>真实 3D 切片渲染图。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_slice_3d/3d_x_slice.png" width="32%" alt="预测 3D 切片渲染图。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_slice_3d/3d_y_slice.png" width="32%" alt="预测 3D 切片渲染图。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/pred_slice_3d/3d_z_slice.png" width="32%" alt="预测 3D 切片渲染图。">
</p>
<p align="center"><em>预测 3D 切片渲染图。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/observed_gzz.png" width="49%" alt="观测与正演 Gzz 图。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/forward_gzz.png" width="49%" alt="观测与正演 Gzz 图。">
</p>
<p align="center"><em>观测与正演 Gzz 图。</em></p>

<p align="center">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/residual.png" width="49%" alt="残差图与残差直方图。">
  <img src="examples/example%20three/Complex%20Terrain%20Model/gravity/histogram.png" width="49%" alt="残差图与残差直方图。">
</p>
<p align="center"><em>残差图与残差直方图。</em></p>

### Field Data Example

该部分展示仓库中 `Field data example` 所包含的真实数据反演结果图。与前面的合成案例不同，这里主要用于展示模型在实测数据 `Gzz` 输入条件下的预测密度体、切片结果以及观测与正演响应之间的拟合情况。

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/3D%20view%20of%20the%20predicted%20density%20model/isosurface_pred.png" width="49%" alt="Predicted density isosurface.">
  <img src="Field%20data%20example/Field%20data%20example/3D%20view%20of%20the%20predicted%20density%20model/voxel_pred.png" width="49%" alt="Predicted density voxel model.">
</p>
<p align="center"><em>预测密度模型的等值面与体素视图。</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/2D%20slice/x_slice.png" width="32%" alt="Predicted orthogonal 2D slices.">
  <img src="Field%20data%20example/Field%20data%20example/2D%20slice/y_slice.png" width="32%" alt="Predicted orthogonal 2D slices.">
  <img src="Field%20data%20example/Field%20data%20example/2D%20slice/z_slice.png" width="32%" alt="Predicted orthogonal 2D slices.">
</p>
<p align="center"><em>预测密度模型在 x、y、z 三个方向上的二维切片。</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_x_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_y_slice.png" width="32%" alt="Predicted 3D slice renderings.">
  <img src="Field%20data%20example/Field%20data%20example/3D%20slice/3d_z_slice.png" width="32%" alt="Predicted 3D slice renderings.">
</p>
<p align="center"><em>预测密度模型在 x、y、z 三个方向上的三维切片渲染图。</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/gravity/observed_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
  <img src="Field%20data%20example/Field%20data%20example/gravity/forward_gzz.png" width="49%" alt="Observed and forward-modelled Gzz maps.">
</p>
<p align="center"><em>观测 Gzz 与预测密度模型正演得到的 Gzz 对比图。</em></p>

<p align="center">
  <img src="Field%20data%20example/Field%20data%20example/gravity/residual.png" width="49%" alt="Residual map and residual histogram.">
  <img src="Field%20data%20example/Field%20data%20example/gravity/histogram.png" width="49%" alt="Residual map and residual histogram.">
</p>
<p align="center"><em>Gzz 残差分布图与残差直方图。</em></p>
