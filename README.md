# Hybrid 2D–3D Transformer Network with Channel-to-Depth Lifting for Physics-Guided 3D Density Gravity Inversion

<img src="https://raw.githubusercontent.com/your-repo/transunet/master/logo.png" width="300">

## Description

TransUNet is a deep learning module based on a hybrid 2D-3D Transformer network for physics-constrained 3D density gravity inversion. The model employs a Channel-to-Depth Lifting (CDL) mechanism to transform 2D gravity anomaly data into 3D density models.

Core innovations include:
- **Hybrid 2D-3D Architecture**: Combines 2D convolutional encoder with 3D Transformer decoder
- **Channel-to-Depth Lifting**: Novel feature lifting mechanism that transforms 2D features into 3D representations
- **Physics Constraints**: Integrates differentiable gravity forward operator ensuring physical consistency
- **Depth Weighting**: Weighted mechanism accounting for depth sensitivity
- **Edge Enhancement**: Improved boundary identification capability

## Installation

### System Requirements
- **Operating System**: Windows 10/11 or Linux
- **Python**: 3.8+
- **Deep Learning Framework**: PyTorch (CUDA support recommended)

### Dependencies Installation

```bash
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib scikit-image
pip install vtk  # For VTI model generation and visualization
```

## File Format Requirements

### Training Stage
The software adopts a built-in generative learning mode; no external input files are needed during training. The program automatically generates training samples containing:
- 3D density models (supporting multiple geological model types)
- Corresponding gravity anomaly data

### Inference Stage
Input files require only one gravity data file with the following specifications:
- **Format**: `.npy` format or Tensor format matrix
- **Data Type**: 2D gridded gravity anomaly data (Gz) or gravity gradient data (Gzz)
- **Dimensions**: Consistent with grid parameters in configuration

## Code Structure

The project is a self-contained single-file program, divided into two main parts:

### (1) Configuration and Parameter Setting

Corresponds to the `V4Config` class in the code, including:

**Grid Parameters**
```python
grid_shape: Tuple[int, int, int] = (16, 32, 32)  # Grid dimensions (depth, north, east)
dx: float = 100.0                                  # Horizontal grid spacing (m)
dz: float = 100.0                                  # Depth grid spacing (m)
```

**Deep Learning Hyperparameters**
```python
lr: float = 5e-4                    # Learning rate
batch_size: int = 8                # Batch size
epochs: int = 400                  # Maximum training epochs
steps_per_epoch: int = 500         # Iteration steps per epoch
```

**Physics Constraint Parameters**
```python
w_depth: float = 1.0               # Depth weighting index
w_physics: float = 0.3             # Physics consistency loss weight
w_edge: float = 0.3                # Edge enhancement coefficient
w_focus: float = 0.3               # Focus loss weight
```

**Model Architecture Parameters**
```python
encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
decoder_channels: Tuple[int, ...] = (128, 64, 32)
lifting_channels: int = 64
num_transformer_layers: int = 6
num_heads: int = 8
```

### (2) Result Output

Automatically runs during and after training, including:

- **Optimal Model Checkpoint**: 3D density volume inversion results
- **Learning Curves**:
  - Loss function curves
  - Inversion accuracy metric curves (Deep Anomaly IoU)
- **Visualization Results**:
  - Density slice images
  - 3D voxel rendering
  - Gravity anomaly fitting comparison (observed vs. predicted)

## Usage

### Basic Training Workflow

```python
from train_code import TransUNetTrainer
from config import V4Config

# Create configuration
config = V4Config()

# Initialize trainer
trainer = TransUNetTrainer(config)

# Start training
trainer.train()
```

### Custom Configuration Example

```python
from config import V4Config

config = V4Config(
    grid_shape=(16, 32, 32),      # Grid dimensions
    dx=100.0,                      # Horizontal spacing
    dz=100.0,                      # Depth spacing
    batch_size=16,                 # Increase batch size
    epochs=500,                    # Increase training epochs
    lr=1e-3,                       # Adjust learning rate
    w_physics=0.5,                 # Increase physics constraint weight
)
```

### Generate Test Models

The project includes a VTI model generation tool for creating synthetic geological models:

```bash
python examples/generate_vti_models.py
```

Supported model types:
- **Complex Terrain**: Complex terrain model
- **Center Prism**: Central prism model
- **Dual Prism**: Dual prism model (positive-negative density contrast)

## Example Results

### Example One: Synthetic Model Inversion

<img src="examples/example one/synthetic model.png" width="400">
<img src="examples/example one/predicted model.png" width="400">

### Example Two: Complex Geological Structure

<img src="examples/example two/synthetic model.png" width="400">
<img src="examples/example two/predicted model.png" width="400">

### Example Three: Complex Terrain Model

<img src="examples/example three/Real the Eight-Character Step Pattern.png" width="400">
<img src="examples/example three/predicted model.png" width="400">

### Real Data Application

<img src="Field data example/predicted_density_model.png" width="400">

## Core Features

### 1. Hybrid 2D-3D Architecture
- 2D Encoder: Efficiently extracts horizontal features
- 3D Transformer Decoder: Captures depth dependencies
- Channel-to-Depth Lifting: Novel feature dimension transformation mechanism

### 2. Physics Constraints
- Integrated differentiable gravity forward operator
- Physics consistency loss function
- Ensures inversion results satisfy gravity field equations

### 3. Depth Sensitivity Handling
- Depth weighting mechanism
- Focus loss function
- Improved deep anomaly identification capability

### 4. Edge Enhancement
- Boundary detection module
- Edge enhancement loss
- Improves boundary clarity of anomalous bodies

### 5. Data Augmentation
- Elastic deformation augmentation
- Multiple geometric transformations
- Improves model generalization capability

## Technical Details

### Gravity Forward Operator

Implements differentiable gravity forward computation using frequency domain methods:

```python
class DifferentiableForward(nn.Module):
    """Differentiable gravity forward operator for physics consistency loss"""

    def __init__(self, shape, dx, dz):
        # FFT-based efficient computation
        # Supports batch processing and automatic differentiation
```

### Transformer Encoder

Employs multi-head self-attention mechanism:
- Layers: 6
- Attention Heads: 8
- Position Encoding: Supported
- Depth Attention: Supported

### Loss Functions

Multi-task learning framework:
- L1 Loss: Basic reconstruction loss
- Physics Consistency Loss: Gravity field constraint
- Depth Weighted Loss: Depth sensitivity
- Focus Loss: Hard sample mining
- Edge Enhancement Loss: Boundary clarity
- Morphological Loss: Structure constraint

## Performance Metrics

### Evaluation Metrics
- **Deep Anomaly IoU**: Anomaly body intersection over union
- **RMSE**: Root mean square error
- **Correlation Coefficient**: Gravity anomaly fitting degree

### Typical Performance
- Grid Resolution: 16 × 32 × 32
- Inference Time: < 100ms (single sample)
- Memory Usage: < 4GB (training)

## FAQ

### Q: How to handle different grid sizes?
A: Modify the `grid_shape` parameter in `V4Config`, the model will automatically adapt.

### Q: How to adjust the strength of physics constraints?
A: Control via the `w_physics` parameter, typically ranging from 0.1-1.0.

### Q: Does it support GPU acceleration?
A: Yes, the code automatically detects CUDA availability and prioritizes GPU usage.

### Q: How to perform inversion with my own gravity data?
A: Prepare 2D gravity anomaly data in `.npy` format and input according to the inference stage format requirements.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Citation

If you use this project in your research, please consider citing the following papers:

[Related paper citations to be added]

## Acknowledgments

Thanks to all contributors and testers for their support.

## Contact

For questions or suggestions, feel free to submit an Issue or Pull Request.
