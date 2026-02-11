# Hybrid 2D–3D Transformer Network with Channel-to-Depth Lifting for Physics-Guided 3D Density Gravity Inversion

1. **Operating system**: Windows 10/11 or Linux (Python 3.8+ and PyTorch environment are required).

2. **File format requirements**: This software adopts a built-in generative learning mode; no external input files are needed during the training stage. The program automatically generates training samples containing 3D density models and corresponding gravity anomaly data. For inference with a trained model, the input file only requires one gravity data file (`.npy` format or Tensor format matrix), containing 2D gridded gravity anomaly (Gz) or gravity gradient data (Gzz).

3. **Code description**: `transunet.py` is a self-contained, single-file program. Simply run this file to start training. The code includes parameter configuration, model training monitoring, and result visualization. The code is mainly divided into two parts:
   - (1) The first part is **Configuration and Parameter Setting** (corresponding to the `V4Config` class in the code). Here, you can set the grid parameters of gravity inversion, including grid dimensions (default 16 × 32 × 32) and grid spacing (dx, dz). You can also set deep learning hyperparameters, including learning rate, batch size, and maximum number of training epochs. In addition, physics constraint parameters are configured here, such as depth weighting index, physics-consistency loss weight, edge enhancement coefficient, and geological model types (prism, sphere, fault, etc.) for automatic training data generation.
   - (2) The second part is the **Result Output**, which runs automatically during and after training. It includes the 3D density volume obtained by inversion (saved as the best model checkpoint), the learning curves of loss function and inversion accuracy metric (Deep Anomaly IoU) during training, as well as density slice images, 3D voxel rendering, and gravity anomaly fitting comparison (observed vs. predicted) for monitoring training progress.

4. **Quick start**:
   ```bash
   # Train the model with default settings
   python transunet.py

   # Train with custom parameters
   python transunet.py --epochs 200 --batch_size 8 --lr 5e-6

   # Resume training from a checkpoint
   python transunet.py --resume path/to/checkpoint.pth

   # Run gradient verification only
   python transunet.py --verify-only
   ```
