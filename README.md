# Hybrid 2D–3D Transformer Network with Channel-to-Depth Lifting for Physics-Guided 3D Density Gravity Inversion

<img src="https://raw.githubusercontent.com/your-repo/transunet/master/logo.png" width="300">

## 描述

TransUNet 是一个基于混合 2D-3D Transformer 网络的深度学习模块，用于物理约束的三维密度重力反演。该模型采用通道到深度提升（Channel-to-Depth Lifting, CDL）机制，将 2D 重力异常数据转换为 3D 密度模型。

核心创新包括：
- **混合 2D-3D 架构**：结合 2D 卷积编码器和 3D Transformer 解码器
- **通道到深度提升**：创新的特征提升机制，将 2D 特征转换为 3D 表示
- **物理约束**：集成重力正演算子，确保反演结果的物理一致性
- **深度加权**：考虑深度敏感性的加权机制
- **边界增强**：改进边界识别能力

## 安装

### 系统要求
- **操作系统**：Windows 10/11 或 Linux
- **Python**：3.8+
- **深度学习框架**：PyTorch（CUDA 支持推荐）

### 依赖安装

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib scikit-image
pip install vtk  # 用于 VTI 模型生成和可视化
```

## 文件格式要求

### 训练阶段
该软件采用内置生成学习模式，训练阶段无需外部输入文件。程序自动生成包含以下内容的训练样本：
- 3D 密度模型（支持多种地质模型类型）
- 对应的重力异常数据

### 推理阶段
输入文件仅需一个重力数据文件，格式要求：
- **格式**：`.npy` 格式或 Tensor 格式矩阵
- **数据类型**：2D 网格化重力异常数据（Gz）或重力梯度数据（Gzz）
- **维度**：与配置中的网格参数一致

## 代码结构

该项目是一个自包含的单文件程序，主要分为两部分：

### (1) 配置与参数设置

对应代码中的 `V4Config` 类，包括：

**网格参数**
```python
grid_shape: Tuple[int, int, int] = (16, 32, 32)  # 网格维度 (深度, 北向, 东向)
dx: float = 100.0                                  # 水平网格间距 (m)
dz: float = 100.0                                  # 深度网格间距 (m)
```

**深度学习超参数**
```python
lr: float = 5e-4                    # 学习率
batch_size: int = 8                # 批大小
epochs: int = 400                  # 最大训练轮数
steps_per_epoch: int = 500         # 每轮迭代步数
```

**物理约束参数**
```python
w_depth: float = 1.0               # 深度加权指数
w_physics: float = 0.3             # 物理一致性损失权重
w_edge: float = 0.3                # 边界增强系数
w_focus: float = 0.3               # 焦点损失权重
```

**模型架构参数**
```python
encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
decoder_channels: Tuple[int, ...] = (128, 64, 32)
lifting_channels: int = 64
num_transformer_layers: int = 6
num_heads: int = 8
```

### (2) 结果输出

训练过程中和训练后自动运行，包括：

- **最优模型检查点**：3D 密度体积反演结果
- **学习曲线**：
  - 损失函数曲线
  - 反演精度指标曲线（Deep Anomaly IoU）
- **可视化结果**：
  - 密度切片图像
  - 3D 体素渲染
  - 重力异常拟合对比（观测 vs 预测）

## 使用方法

### 基本训练流程

```python
from train_code import TransUNetTrainer
from config import V4Config

# 创建配置
config = V4Config()

# 初始化训练器
trainer = TransUNetTrainer(config)

# 开始训练
trainer.train()
```

### 自定义配置示例

```python
from config import V4Config

config = V4Config(
    grid_shape=(16, 32, 32),      # 网格维度
    dx=100.0,                      # 水平间距
    dz=100.0,                      # 深度间距
    batch_size=16,                 # 增加批大小
    epochs=500,                    # 增加训练轮数
    lr=1e-3,                       # 调整学习率
    w_physics=0.5,                 # 增加物理约束权重
)
```

### 生成测试模型

项目包含 VTI 模型生成工具，用于创建合成地质模型：

```bash
python examples/generate_vti_models.py
```

支持的模型类型：
- **Figure-8 Staircase**：八字形阶梯模型
- **Center Prism**：中心棱柱体模型
- **Dual Prism**：双棱柱体模型（正负密度对比）

## 示例结果

### 示例一：合成模型反演

<img src="examples/example one/synthetic model.png" width="400">
<img src="examples/example one/predicted model.png" width="400">

### 示例二：复杂地质结构

<img src="examples/example two/synthetic model.png" width="400">
<img src="examples/example two/predicted model.png" width="400">

### 示例三：八字形阶梯模型

<img src="examples/example three/Real the Eight-Character Step Pattern.png" width="400">
<img src="examples/example three/predicted model.png" width="400">

### 实际数据应用

<img src="Field data example/predicted_density_model.png" width="400">

## 核心特性

### 1. 混合 2D-3D 架构
- 2D 编码器：高效提取水平特征
- 3D Transformer 解码器：捕捉深度依赖关系
- 通道到深度提升：创新的特征维度转换机制

### 2. 物理约束
- 集成可微分重力正演算子
- 物理一致性损失函数
- 确保反演结果满足重力场方程

### 3. 深度敏感性处理
- 深度加权机制
- 焦点损失函数
- 改进深层异常识别能力

### 4. 边界增强
- 边界检测模块
- 边界增强损失
- 提高异常体边界清晰度

### 5. 数据增强
- 弹性变形增强
- 多种几何变换
- 提高模型泛化能力

## 技术细节

### 重力正演算子

采用频域方法实现可微分的重力正演：

```python
class DifferentiableForward(nn.Module):
    """可微分重力正演算子，用于物理一致性损失"""

    def __init__(self, shape, dx, dz):
        # 基于 FFT 的高效计算
        # 支持批处理和自动微分
```

### Transformer 编码器

采用多头自注意力机制：
- 层数：6 层
- 注意力头数：8
- 位置编码：支持
- 深度注意力：支持

### 损失函数

多任务学习框架：
- L1 损失：基础重建损失
- 物理一致性损失：重力场约束
- 深度加权损失：深度敏感性
- 焦点损失：难样本挖掘
- 边界增强损失：边界清晰度
- 形态学损失：结构约束

## 性能指标

### 评估指标
- **Deep Anomaly IoU**：异常体交并比
- **RMSE**：均方根误差
- **相关系数**：重力异常拟合度

### 典型性能
- 网格分辨率：16 × 32 × 32
- 推理时间：< 100ms（单样本）
- 内存占用：< 4GB（训练）

## 常见问题

### Q: 如何处理不同大小的网格？
A: 修改 `V4Config` 中的 `grid_shape` 参数，模型会自动适应。

### Q: 如何调整物理约束的强度？
A: 通过 `w_physics` 参数控制，范围通常为 0.1-1.0。

### Q: 支持 GPU 加速吗？
A: 是的，代码自动检测 CUDA 可用性，优先使用 GPU。

### Q: 如何使用自己的重力数据进行反演？
A: 准备 `.npy` 格式的 2D 重力异常数据，按照推理阶段的格式要求输入。

## 许可证

该项目采用 Apache License 2.0 许可证。详见 LICENSE 文件。

## 引用

如果你在研究中使用了本项目，请引用以下论文：

[待补充相关论文引用]

## 致谢

感谢所有贡献者和测试人员的支持。

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
