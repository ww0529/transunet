# V4 Gravity Inversion GUI

## Overview
This is a professional GUI application for testing V4 gravity inversion models. It provides an intuitive interface for loading data, running inversions, and visualizing results.

## Features
- Load gravity data (Gzz.txt, VTI format)
- Load pre-trained V4 models (.pth files)
- Run gravity inversion
- 2D/3D visualization of results
- Density scaling optimization
- Real coordinate display
- Export results as images

## Requirements
- Python 3.8+
- PyTorch 2.0+
- PySide6
- NumPy, SciPy
- Matplotlib
- VTK (for VTI file support)

## Installation

### 1. Install dependencies
```bash
pip install torch torchvision torchaudio
pip install PySide6
pip install numpy scipy matplotlib
pip install vtk
```

### 2. Prepare model file
Place your trained model file (best_model.pth) in one of these locations:
- `../best_model.pth` (parent directory)
- `./best_model.pth` (current directory)
- `../source code/Train code/best_model.pth`

## Usage

### Run the GUI
```bash
python run_gui.py
```

Or directly:
```bash
python jgui.py
```

### Basic Workflow
1. **Load Model**: Click "Load Model" button and select your trained .pth file
2. **Load Data**: Click "Load Data" and select:
   - Gzz.txt: Text file with X, Y, Gzz columns
   - VTI: VTK ImageData format with 3D density model
3. **Run Inversion**: Click "Run Inversion" button
4. **View Results**:
   - 2D Slices: View horizontal and vertical cross-sections
   - 3D Slices: View orthogonal slices in 3D space
   - 3D Voxels: View density distribution as voxel body
5. **Save Results**: Click "Save Image" to export visualizations

### Data Format

#### Gzz.txt Format
```
X1 Y1 Gzz1
X2 Y2 Gzz2
...
```
- X, Y: Coordinates in meters
- Gzz: Gravity anomaly in Eotvos

#### VTI Format
- 3D density model in VTK ImageData format
- Automatically computes synthetic gravity from density

## GUI Controls

### Left Panel
- **Model Section**: Load trained model
- **Data Section**: Load gravity data or 3D model
- **Parameters**: Set density range and scaling options
- **Run Inversion**: Execute the inversion
- **Slice Control**: Adjust visualization slice positions
- **Threshold**: Control voxel display threshold

### Right Panel
- **2D Slices Tab**: 2D cross-sections
- **3D Slices Tab**: 3D orthogonal slices
- **3D Voxels Tab**: 3D density visualization

## Model Parameters

### Density Range
- Min: Minimum density (default: -150 kg/m³)
- Max: Maximum density (default: 350 kg/m³)

### Options
- **Sign Inversion**: Flip density sign if needed
- **Gzz Residual Scaling**: Auto-scale density based on gravity fit

## Output

### Visualization Tabs
1. **2D Slices**:
   - Input Gzz or true density
   - Predicted Z-slice
   - Predicted Y-slice
   - Predicted X-slice

2. **3D Slices**:
   - Orthogonal slices (X, Y, Z)
   - Adjustable slice positions
   - Visibility toggle for each slice

3. **3D Voxels**:
   - Density distribution as 3D voxel body
   - Adjustable threshold
   - Color-coded by density value

### Export
- Save current visualization as PNG (300 dpi)
- Supports all three visualization modes

## Troubleshooting

### Model not loading
- Check that model file exists and is valid PyTorch format
- Ensure train_code.py and config.py are in the same directory
- Check console output for detailed error messages

### Data loading issues
- For Gzz.txt: Ensure 3 columns (X, Y, Gzz)
- For VTI: Install VTK with `pip install vtk`
- Check file path and permissions

### GPU issues
- If CUDA not available, will automatically use CPU
- Check PyTorch CUDA compatibility with your GPU

## File Structure
```
Train code/
├── jgui.py              # Main GUI application
├── run_gui.py           # Launch script
├── train_code.py        # Model definition
├── config.py            # Configuration
├── data_preparation.py  # Data utilities
└── best_model.pth       # Trained model (optional)
```

## Author
Developed for V4 Gravity Inversion Project

## License
MIT License
