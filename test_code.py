
import sys
import os
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import zoom, uniform_filter, gaussian_filter
from typing import Optional, Tuple, Dict, Any


def _configure_module_paths() -> None:
    """Make imports work across both current and legacy project layouts."""
    base_candidates = []

    env_root = os.environ.get("TRANSUNET_ROOT")
    if env_root:
        base_candidates.append(Path(env_root).expanduser())

    script_dir = Path(__file__).resolve().parent
    base_candidates.extend([script_dir, *script_dir.parents])

    cwd = Path.cwd().resolve()
    base_candidates.extend([cwd, *cwd.parents])

    added = set()
    for base in base_candidates:
        try:
            base = base.resolve()
        except OSError:
            continue

        module_dirs = (
            base / "source code",
            base / "source code" / "Train code",
            base / "source code" / "Dataset preparation",
        )
        for module_dir in module_dirs:
            if not module_dir.is_dir():
                continue
            module_path = str(module_dir)
            if module_path in added:
                continue
            added.add(module_path)
            if module_path not in sys.path:
                sys.path.insert(0, module_path)


_configure_module_paths()

# ============== 1. 导入依赖 ==============
# PySide6 GUI
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QCheckBox, QSlider, QFileDialog, QMessageBox, QTabWidget,
    QSplitter, QFrame, QStatusBar, QProgressBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator, QIntValidator

# Matplotlib 2D 可视化
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyVista 3D visualization
try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
    print("[OK] PyVista imported successfully")
except ImportError:
    HAS_PYVISTA = False
    print("[WARNING] PyVista not available, using matplotlib for 3D")
from mpl_toolkits.mplot3d import Axes3D

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== 2. V4 Model Import ==============
HAS_V4 = False
HAS_FAST_FORWARD = False
PhysicsInformedUNet = None
V4Config = None
HighResVisualizer = None
FastGravityForward = None

try:
    from train_code import PhysicsInformedUNet, HighResVisualizer
    print("[OK] V4 core model imported from train_code successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import V4 core model: {e}")
    import traceback
    traceback.print_exc()

try:
    from config import V4Config
    print("[OK] V4 config imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import V4 config: {e}")
    import traceback
    traceback.print_exc()

HAS_V4 = PhysicsInformedUNet is not None and HighResVisualizer is not None and V4Config is not None

try:
    from data_preparation import FastGravityForward
    HAS_FAST_FORWARD = True
    print("[OK] FastGravityForward imported from data_preparation")
except ImportError as e:
    print(f"[WARNING] FastGravityForward import failed: {e}")


# ============== 3. 数据管理器 ==============
class DataManager:
    """管理输入数据的加载和预处理"""

    def __init__(self):
        self.gzz_raw: Optional[np.ndarray] = None  # 原始 Gzz 数据
        self.gzz_32: Optional[np.ndarray] = None   # 插值到 32x32
        self.x_range: Tuple[float, float] = (0, 4000)
        self.y_range: Tuple[float, float] = (0, 4000)
        self.dx: float = 100.0
        self.dy: float = 100.0
        self.original_shape: Tuple[int, int] = (32, 32)

    def load_gzz_txt(self, filepath: str) -> bool:
        """加载 Gzz.txt 格式数据"""
        try:
            # 尝试不同的分隔符
            try:
                data = np.loadtxt(filepath, delimiter=',')
            except:
                data = np.loadtxt(filepath)

            if data.shape[1] < 3:
                raise ValueError("数据至少需要 3 列 (X, Y, Gzz)")

            x, y, gzz = data[:, 0], data[:, 1], data[:, 2]

            # 计算网格参数
            unique_x = np.unique(x)
            unique_y = np.unique(y)
            nx, ny = len(unique_x), len(unique_y)

            self.x_range = (float(unique_x.min()), float(unique_x.max()))
            self.y_range = (float(unique_y.min()), float(unique_y.max()))
            self.dx = float(unique_x[1] - unique_x[0]) if nx > 1 else 100.0
            self.dy = float(unique_y[1] - unique_y[0]) if ny > 1 else 100.0
            self.original_shape = (ny, nx)

            # 转换为 Eötvös
            gzz_values = gzz.copy()
            if np.max(np.abs(gzz_values)) < 1e-4:
                gzz_values *= 1e9

            # 重塑为网格
            self.gzz_raw = gzz_values.reshape(ny, nx)

            # 插值到 32x32
            if (ny, nx) != (32, 32):
                self.gzz_32 = zoom(self.gzz_raw, (32/ny, 32/nx), order=3)
            else:
                self.gzz_32 = self.gzz_raw.copy()

            print(f"✓ Loaded: {filepath}")
            print(f"  Original: {nx}x{ny}, Range: [{gzz.min():.2e}, {gzz.max():.2e}]")
            print(f"  Coords: X={self.x_range}, Y={self.y_range}")

            return True

        except Exception as e:
            print(f"✗ Load failed: {e}")
            return False

    def load_vti(self, filepath: str) -> bool:
        """加载 VTI (VTK ImageData) 格式文件"""
        try:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy

            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(filepath)
            reader.Update()

            image_data = reader.GetOutput()
            dims = image_data.GetDimensions()  # (nx, ny, nz)
            spacing = image_data.GetSpacing()
            origin = image_data.GetOrigin()

            nx, ny, nz = dims

            # 尝试多种方式获取数据数组
            data_flat = None
            array_name = "Unknown"

            # 方法1: PointData.GetScalars
            point_data = image_data.GetPointData()
            if point_data.GetScalars() is not None:
                array = point_data.GetScalars()
                data_flat = vtk_to_numpy(array)
                array_name = "PointData.Scalars"
            # 方法2: PointData.GetArray(0)
            elif point_data.GetNumberOfArrays() > 0:
                array = point_data.GetArray(0)
                data_flat = vtk_to_numpy(array)
                array_name = f"PointData.Array[0]: {array.GetName()}"
            # 方法3: CellData.GetScalars
            else:
                cell_data = image_data.GetCellData()
                if cell_data.GetScalars() is not None:
                    array = cell_data.GetScalars()
                    data_flat = vtk_to_numpy(array)
                    array_name = "CellData.Scalars"
                    # CellData 维度比 PointData 少1
                    nx, ny, nz = max(1, nx-1), max(1, ny-1), max(1, nz-1)
                elif cell_data.GetNumberOfArrays() > 0:
                    array = cell_data.GetArray(0)
                    data_flat = vtk_to_numpy(array)
                    array_name = f"CellData.Array[0]: {array.GetName()}"
                    nx, ny, nz = max(1, nx-1), max(1, ny-1), max(1, nz-1)

            if data_flat is None:
                raise ValueError("VTI 文件中没有找到数据数组（PointData/CellData均为空）")

            print(f"  Found data in: {array_name}")
            print(f"  Data shape: {data_flat.shape}, Target: ({nz}, {ny}, {nx}) = {nz*ny*nx}")

            # 重塑数据
            try:
                # VTK 使用 Fortran 顺序 (x变化最快)
                data_3d = data_flat.reshape((nz, ny, nx), order='F')
            except ValueError:
                # 尝试 C 顺序
                try:
                    data_3d = data_flat.reshape((nz, ny, nx), order='C')
                except ValueError:
                    # 如果维度不匹配，尝试推断
                    total = data_flat.size
                    if total == nx * ny * nz:
                        data_3d = data_flat.reshape((nz, ny, nx))
                    elif total == (nx-1) * (ny-1) * (nz-1):
                        data_3d = data_flat.reshape((nz-1, ny-1, nx-1))
                    else:
                        raise ValueError(f"无法重塑数据: {data_flat.size} -> ({nz}, {ny}, {nx})")

            # 计算坐标范围
            self.x_range = (origin[0], origin[0] + spacing[0] * (nx - 1))
            self.y_range = (origin[1], origin[1] + spacing[1] * (ny - 1))
            self.dx = spacing[0]
            self.dy = spacing[1]
            self.dz = spacing[2] if len(spacing) > 2 else 100.0

            # 如果是3D数据，使用正演计算重力
            if nz > 1:
                # 存储完整3D数据用于可视化
                self.density_3d = data_3d
                self.original_shape = (ny, nx)
                print(f"  3D Data: {nz}x{ny}x{nx}")

                # 使用正演计算重力作为模型输入
                try:
                    from data_preparation import FastGravityForward
                    fwd = FastGravityForward((nz, ny, nx), dx=self.dx, dz=self.dz, mode='gzz')
                    # 假设密度值是归一化的 [-1, 1]，转换为 g/cm³
                    density_gcm3 = data_3d * 0.3  # 假设最大密度差 0.3 g/cm³
                    gzz_computed = fwd.forward(density_gcm3)
                    self.gzz_raw = gzz_computed
                    print(f"  Computed Gzz from forward: range=[{gzz_computed.min():.4e}, {gzz_computed.max():.4e}]")
                except Exception as e:
                    print(f"  Forward failed: {e}, using column sum approximation")
                    # 备用：使用深度加权的列积分近似重力
                    depth_weights = np.exp(-np.arange(nz) * 0.1)[:, None, None]  # 浅层权重大
                    self.gzz_raw = (data_3d * depth_weights).sum(axis=0)
            else:
                self.gzz_raw = data_3d[0, :, :]
                self.original_shape = (ny, nx)

            # 插值到 32x32
            if self.gzz_raw.shape != (32, 32):
                self.gzz_32 = zoom(self.gzz_raw, (32/self.gzz_raw.shape[0], 32/self.gzz_raw.shape[1]), order=3)
            else:
                self.gzz_32 = self.gzz_raw.copy()

            print(f"✓ Loaded VTI: {filepath}")
            print(f"  Dims: {nx}x{ny}x{nz}, Spacing: {spacing}")
            print(f"  Value Range: [{data_3d.min():.4f}, {data_3d.max():.4f}]")
            print(f"  Coords: X={self.x_range}, Y={self.y_range}")

            return True

        except ImportError:
            print("✗ VTK 未安装。请运行: pip install vtk")
            return False
        except Exception as e:
            print(f"✗ Load VTI failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_file(self, filepath: str) -> bool:
        """根据文件扩展名自动选择加载方法"""
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.vti':
            return self.load_vti(filepath)
        elif ext in ['.txt', '.csv', '.dat']:
            return self.load_gzz_txt(filepath)
        elif ext == '.mat':
            return self.load_mat(filepath) if hasattr(self, 'load_mat') else False
        else:
            print(f"✗ 不支持的文件格式: {ext}")
            return False

    def get_input_tensor(
        self,
        nz: int = 16,
        data_mode: str = 'joint',
        target_hw: Tuple[int, int] = (32, 32)
    ) -> Optional[torch.Tensor]:
        """Build model input tensor [1, C, D, H, W] for train_code model."""
        if self.gzz_32 is None:
            return None

        target_ny, target_nx = int(target_hw[0]), int(target_hw[1])
        if target_ny <= 0 or target_nx <= 0:
            target_ny, target_nx = 32, 32

        gzz_map = self.gzz_32
        if gzz_map.shape != (target_ny, target_nx):
            sy = target_ny / gzz_map.shape[0]
            sx = target_nx / gzz_map.shape[1]
            gzz_map = zoom(gzz_map, (sy, sx), order=3)

        if hasattr(self, 'density_3d') and self.density_3d is not None and HAS_FAST_FORWARD and FastGravityForward is not None:
            try:
                m_true = self.density_3d
                tnz, tny, tnx = m_true.shape
                fwd = FastGravityForward((tnz, tny, tnx), dx=self.dx, dz=getattr(self, 'dz', self.dx), mode='joint')
                field_data = fwd.forward(m_true)
                gz_raw, gzz_raw = field_data[0], field_data[1]

                sy = target_ny / gz_raw.shape[0]
                sx = target_nx / gz_raw.shape[1]
                gz_map = zoom(gz_raw, (sy, sx), order=1)
                gzz_map = zoom(gzz_raw, (sy, sx), order=1)
            except Exception as e:
                print(f"[VTI Mode] FastGravityForward failed: {e}, fallback to filtered estimate")
                gz_map = uniform_filter(gzz_map, size=5)
        else:
            gz_map = uniform_filter(gzz_map, size=5)

        gzz_max = np.abs(gzz_map).max() + 1e-8
        gz_max = np.abs(gz_map).max() + 1e-8
        gzz_norm = gzz_map / gzz_max
        gz_norm = gz_map / gz_max

        z_indices = np.linspace(0, 1, nz)
        z_map = np.tile(z_indices[:, None, None], (1, target_ny, target_nx))
        z_tensor = torch.from_numpy(z_map).float()

        mode = str(data_mode).lower()
        if mode == 'joint':
            gz_tensor = torch.from_numpy(gz_norm).float().unsqueeze(0).expand(nz, -1, -1)
            gzz_tensor = torch.from_numpy(gzz_norm).float().unsqueeze(0).expand(nz, -1, -1)
            input_vol = torch.stack([gz_tensor, gzz_tensor, z_tensor], dim=0)
        else:
            primary = gz_norm if mode == 'gz' else gzz_norm
            primary_tensor = torch.from_numpy(primary).float().unsqueeze(0).expand(nz, -1, -1)
            input_vol = torch.stack([primary_tensor, z_tensor], dim=0)

        return input_vol.unsqueeze(0)



# ============== 4. 模型管理器 ==============
class ModelManager:
    """Manage V4 model loading and inference."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.model_path: str = ""
        self.is_loaded: bool = False
        self.last_error: str = ""
        self.config: Optional[Any] = None
        self.data_mode: str = 'joint'
        self.grid_shape: Tuple[int, int, int] = (16, 32, 32)
        self.input_channels: int = 3

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Any:
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict']
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
        return checkpoint

    @staticmethod
    def _build_config_from_checkpoint(raw_cfg: Any) -> Any:
        cfg = V4Config()
        if raw_cfg is None:
            return cfg

        if isinstance(raw_cfg, V4Config):
            return raw_cfg

        if isinstance(raw_cfg, dict):
            src = raw_cfg
        elif hasattr(raw_cfg, '__dict__'):
            src = {k: v for k, v in vars(raw_cfg).items() if not k.startswith('_')}
        else:
            return cfg

        valid = {k: v for k, v in src.items() if hasattr(cfg, k)}
        try:
            return V4Config(**valid)
        except Exception:
            for key, value in valid.items():
                setattr(cfg, key, value)
            return cfg

    def load_model(self, filepath: str) -> bool:
        """Load model weights and infer runtime config from checkpoint."""
        if not HAS_V4:
            self.last_error = "V4 module not available"
            print(f"[ERROR] {self.last_error}")
            return False

        try:
            self.last_error = ""
            checkpoint = torch.load(filepath, map_location=DEVICE, weights_only=False)
            raw_cfg = checkpoint.get('config') if isinstance(checkpoint, dict) else None
            cfg = self._build_config_from_checkpoint(raw_cfg)

            self.model = PhysicsInformedUNet(cfg).to(DEVICE)
            state_dict = self._extract_state_dict(checkpoint)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            self.config = cfg
            self.data_mode = str(getattr(cfg, 'data_mode', 'joint')).lower()
            grid_shape = getattr(cfg, 'grid_shape', (16, 32, 32))
            if isinstance(grid_shape, (list, tuple)) and len(grid_shape) == 3:
                self.grid_shape = tuple(int(v) for v in grid_shape)
            else:
                self.grid_shape = (16, 32, 32)
            self.input_channels = 3 if self.data_mode == 'joint' else 2

            self.model_path = filepath
            self.is_loaded = True

            print(f"[OK] Model loaded: {os.path.basename(filepath)}")
            print(f"  data_mode={self.data_mode}, grid_shape={self.grid_shape}, in_channels={self.input_channels}")
            print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            return True

        except Exception as e:
            self.last_error = str(e)
            print(f"[ERROR] Model load failed: {self.last_error}")
            self.is_loaded = False
            self.model = None
            return False

    def predict(self, input_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """Run model inference."""
        if not self.is_loaded or self.model is None:
            return None

        try:
            if input_tensor.ndim != 5:
                raise ValueError(f"Input tensor must be 5D [B,C,D,H,W], got shape={tuple(input_tensor.shape)}")
            if input_tensor.shape[1] != self.input_channels:
                raise ValueError(
                    f"Input channel mismatch: expected {self.input_channels}, got {input_tensor.shape[1]}"
                )

            with torch.no_grad():
                pred = self.model(input_tensor.to(DEVICE)).squeeze().cpu().numpy()
            return pred
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None



# ============== 5. 密度处理器 ==============
class DensityProcessor:
    """处理密度转换和缩放"""

    @staticmethod
    def normalize_to_real(pred_norm: np.ndarray,
                          rho_min: float = -150,
                          rho_max: float = 350,
                          invert: bool = True) -> np.ndarray:
        """归一化 [-1,1] -> 真实密度 [kg/m³]"""
        pred = -pred_norm if invert else pred_norm.copy()
        scale = (rho_max - rho_min) / 2
        offset = (rho_max + rho_min) / 2
        return pred * scale + offset

    @staticmethod
    def forward_gzz(density: np.ndarray, dx: float = 100.0, dz: float = 100.0) -> np.ndarray:
        """
        使用 FFT 加速的精确 Gzz 正演计算
        
        基于卷积定理：Gzz = density * kernel (在频域中是乘法)
        kernel 是 Gzz 的 Green 函数
        
        对于点质量在 (x', y', z') 处，观测点在 (x, y, 0) 处：
        Gzz = G * rho * dV * (2*z'^2 - dx^2 - dy^2) / r^5
        其中 dx = x - x', dy = y - y', r = sqrt(dx^2 + dy^2 + z'^2)
        """
        nz, ny, nx = density.shape
        G = 6.674e-11  # 万有引力常数 m³/(kg·s²)
        
        # 体素体积
        dV = dx * dx * dz
        
        # 创建扩展网格用于 FFT (避免周期性边界效应)
        ny_ext, nx_ext = 2 * ny, 2 * nx
        
        # 初始化 Gzz
        gzz_calc = np.zeros((ny, nx))
        
        # 创建水平坐标网格 (相对于网格中心)
        x = (np.arange(nx_ext) - nx) * dx
        y = (np.arange(ny_ext) - ny) * dx
        XX, YY = np.meshgrid(x, y)
        
        # 对每个深度层计算贡献并叠加
        for iz in range(nz):
            z = (iz + 0.5) * dz  # 层中心深度
            
            # 计算 Gzz Green 函数 (卷积核)
            r2 = XX**2 + YY**2 + z**2
            r = np.sqrt(r2)
            r5 = r**5 + 1e-30  # 避免除零
            
            # Gzz kernel: (2*z^2 - x^2 - y^2) / r^5
            kernel = (2.0 * z**2 - XX**2 - YY**2) / r5
            
            # 避免中心点的奇异值
            kernel[ny, nx] = 0.0
            
            # 将密度层扩展到双倍尺寸 (零填充)
            density_ext = np.zeros((ny_ext, nx_ext))
            density_ext[:ny, :nx] = density[iz, :, :]
            
            # FFT 卷积
            kernel_fft = np.fft.fft2(kernel)
            density_fft = np.fft.fft2(density_ext)
            conv_fft = kernel_fft * density_fft
            conv = np.real(np.fft.ifft2(conv_fft))
            
            # 提取有效区域
            gzz_layer = conv[:ny, :nx]
            
            # 累加贡献
            gzz_calc += gzz_layer * G * dV
        
        # 转换单位: 1/s² -> Eötvös (1 E = 1e-9 /s²)
        gzz_calc = gzz_calc * 1e9
        
        # 安全检查
        gzz_calc = np.nan_to_num(gzz_calc, nan=0.0, posinf=0.0, neginf=0.0)
        return gzz_calc

    @staticmethod
    def scale_by_residual(pred_density: np.ndarray,
                          gzz_obs: np.ndarray,
                          dx: float = 100.0,
                          dz: float = 100.0,
                          background: float = 100.0) -> Tuple[np.ndarray, float]:
        """基于 Gzz 残差缩放密度"""
        # 验证输入
        if pred_density is None or gzz_obs is None:
            return pred_density, 1.0
        if pred_density.ndim != 3 or gzz_obs.ndim != 2:
            return pred_density, 1.0

        # 正演计算
        gzz_calc = DensityProcessor.forward_gzz(pred_density, dx, dz)

        # Resample observed field to prediction grid if needed
        if gzz_obs.shape != gzz_calc.shape:
            sy = gzz_calc.shape[0] / gzz_obs.shape[0]
            sx = gzz_calc.shape[1] / gzz_obs.shape[1]
            gzz_obs = zoom(gzz_obs, (sy, sx), order=3)

        # 线性最小二乘
        obs_max = np.abs(gzz_obs).max()
        mask = np.abs(gzz_obs) > obs_max * 0.3

        if mask.sum() > 0:
            numerator = np.sum(gzz_calc[mask] * gzz_obs[mask])
            denominator = np.sum(gzz_calc[mask] * gzz_calc[mask]) + 1e-10
        else:
            numerator = np.sum(gzz_calc * gzz_obs)
            denominator = np.sum(gzz_calc * gzz_calc) + 1e-10

        alpha = np.clip(numerator / denominator, 0.5, 5.0)  # 限制在合理范围

        # 缩放异常
        density_anomaly = pred_density - background
        scaled = background + density_anomaly * alpha

        # 安全钳位 - 确保密度在物理合理范围内
        min_density = background - 300  # e.g., -200 kg/m³
        max_density = background + 400  # e.g., 500 kg/m³
        scaled = np.clip(scaled, min_density, max_density)

        print(f"  Scale factor: {alpha:.3f}")
        print(f"  Density: [{pred_density.min():.1f}, {pred_density.max():.1f}] -> [{scaled.min():.1f}, {scaled.max():.1f}] kg/m³")

        return scaled, alpha


# ============== 6. 可视化器 ==============
class Visualizer:
    """2D/3D 可视化"""

    @staticmethod
    def clear_colorbar(ax):
        """清除ax关联的colorbar（如果存在）"""
        # 查找并移除与该ax关联的colorbar
        if hasattr(ax, '_colorbar'):
            try:
                ax._colorbar.remove()
            except:
                pass
            ax._colorbar = None

    @staticmethod
    def create_figure_2x2() -> Tuple[Figure, np.ndarray]:
        """创建 2x2 布局的 Figure"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        return fig, axes

    @staticmethod
    def plot_gzz_2d(ax, gzz: np.ndarray, x_range: Tuple, y_range: Tuple, title: str = "Gzz"):
        """绘制 2D Gzz 热图"""
        Visualizer.clear_colorbar(ax)
        ax.clear()
        # Y轴从下往上：extent=[left, right, bottom, top], origin='lower'
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        im = ax.imshow(gzz, cmap='jet', aspect='auto', extent=extent, origin='lower')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(direction='in')  # 刻度线在色条内部
        cbar.ax.set_title('E', fontsize=9)  # 单位在色条顶部
        ax._colorbar = cbar  # 保存引用以便后续清除
        return im

    @staticmethod
    def plot_slice_z(ax, volume: np.ndarray, z_idx: int, x_range: Tuple, y_range: Tuple,
                     vmin: float, vmax: float, title: str = ""):
        """绘制 Z 切片 (水平切片)"""
        Visualizer.clear_colorbar(ax)
        ax.clear()
        if z_idx >= volume.shape[0]:
            z_idx = volume.shape[0] - 1

        slice_data = volume[z_idx, :, :]
        # Y轴从下往上：extent=[left, right, bottom, top], origin='lower'
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        im = ax.imshow(slice_data, cmap='jet', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(direction='in')  # 刻度线在色条内部
        cbar.ax.set_title('kg/m³', fontsize=9)  # 单位在色条顶部
        ax._colorbar = cbar
        return im

    @staticmethod
    def plot_slice_y(ax, volume: np.ndarray, y_idx: int, x_range: Tuple, z_extent: float,
                     vmin: float, vmax: float, title: str = ""):
        """绘制 Y 切片 (垂直切片, XZ平面)"""
        Visualizer.clear_colorbar(ax)
        ax.clear()
        nz, ny, nx = volume.shape
        if y_idx >= ny:
            y_idx = ny - 1

        slice_data = volume[:, y_idx, :]  # [Z, X]
        # Z轴从下往上：extent=[left, right, bottom, top], origin='lower'
        extent = [x_range[0], x_range[1], 0, z_extent]
        im = ax.imshow(slice_data, cmap='jet', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Depth (m)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(direction='in')  # 刻度线在色条内部
        cbar.ax.set_title('kg/m³', fontsize=9)  # 单位在色条顶部
        ax._colorbar = cbar
        return im

    @staticmethod
    def plot_3d_slices(ax, volume: np.ndarray, x_range: Tuple, y_range: Tuple, z_extent: float,
                       x_pos: float, y_pos: float, z_pos: float, vmin: float, vmax: float,
                       show_x: bool = True, show_y: bool = True, show_z: bool = True,
                       title: str = ""):
        """绘制 3D 正交切片（支持精确坐标位置和可见性控制）
        
        Args:
            x_pos, y_pos, z_pos: 精确的切片位置（米）
        """
        from scipy.interpolate import RegularGridInterpolator
        
        ax.clear()

        # 设置3D背景透明
        ax.set_facecolor('none')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))  # 淡灰色半透明
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        # 网格线也设置为淡色
        ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)
        ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)
        ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)

        nz, ny, nx = volume.shape

        # 计算网格坐标
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(0, z_extent, nz)
        
        # 限制切片位置在有效范围内
        x_pos = np.clip(x_pos, x_range[0], x_range[1])
        y_pos = np.clip(y_pos, y_range[0], y_range[1])
        z_pos = np.clip(z_pos, 0, z_extent)
        
        # 创建插值器 (z, y, x 顺序)
        interp = RegularGridInterpolator((z, y, x), volume, method='linear', bounds_error=False, fill_value=None)

        # 绘制三个正交切片
        from matplotlib import cm
        # 确保 vmin < vmax 以避免 ValueError
        if vmin >= vmax:
            vmin = vmax - 1
            vmax = vmin + 2
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        slices_drawn = 0

        # Z 切片 (水平) - 精确位置插值
        if show_z:
            XX, YY = np.meshgrid(x, y)
            ZZ = np.full_like(XX, z_pos)
            # 构建插值点 (z, y, x)
            points = np.stack([ZZ.flatten(), YY.flatten(), XX.flatten()], axis=-1)
            slice_data = interp(points).reshape(ny, nx)
            colors_z = cm.jet(norm(slice_data))
            ax.plot_surface(XX, YY, ZZ, facecolors=colors_z, shade=False, alpha=0.7)
            slices_drawn += 1

        # Y 切片 (垂直) - 精确位置插值
        if show_y:
            XX_y, ZZ_y = np.meshgrid(x, z)
            YY_y = np.full_like(XX_y, y_pos)
            # 构建插值点 (z, y, x)
            points = np.stack([ZZ_y.flatten(), YY_y.flatten(), XX_y.flatten()], axis=-1)
            slice_data = interp(points).reshape(nz, nx)
            colors_y = cm.jet(norm(slice_data))
            ax.plot_surface(XX_y, YY_y, ZZ_y, facecolors=colors_y, shade=False, alpha=0.7)
            slices_drawn += 1

        # X 切片 (垂直) - 精确位置插值
        if show_x:
            YY_x, ZZ_x = np.meshgrid(y, z)
            XX_x = np.full_like(YY_x, x_pos)
            # 构建插值点 (z, y, x)
            points = np.stack([ZZ_x.flatten(), YY_x.flatten(), XX_x.flatten()], axis=-1)
            slice_data = interp(points).reshape(nz, ny)
            colors_x = cm.jet(norm(slice_data))
            ax.plot_surface(XX_x, YY_x, ZZ_x, facecolors=colors_x, shade=False, alpha=0.7)
            slices_drawn += 1

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Depth (m)')
        ax.set_title(title)  # 只显示传入的标题，不加slices数量
        
        # 设置完整的坐标轴范围
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(0, z_extent)  # Z轴：深度0在下方，最大深度在上方
        
        # 调整 3D 轴比例，使 Z 轴在视觉上更长
        x_len = x_range[1] - x_range[0]
        y_len = y_range[1] - y_range[0]
        z_len = z_extent
        max_len = max(x_len, y_len, z_len)
        # Z 轴放大因子：让 Z 轴看起来至少是最大轴的 0.6 倍（可调）
        z_scale = max(1.0, max_len / z_len * 0.6) if z_len > 0 else 1.0
        ax.set_box_aspect([x_len / max_len, y_len / max_len, z_len / max_len * z_scale])
        
        # 视角设置：azim=225 让 Z 轴在左边
        ax.view_init(elev=25, azim=225)

    @staticmethod
    def plot_3d_voxels(ax, volume: np.ndarray, x_range: Tuple, y_range: Tuple, z_extent: float,
                       vmin: float, vmax: float, threshold: float = 0.3, title: str = "",
                       reverse_color: bool = False, show_negative: bool = False):
        """
        绘制 3D 密度体素图（真实体素立方体）

        Args:
            volume: 3D 密度数据 (nz, ny, nx) [kg/m³]
            threshold: 显示阈值 (相对于异常的百分比)
            title: 标题
            reverse_color: 是否反转颜色
        """
        ax.clear()
        from matplotlib import cm

        # 设置3D背景透明
        ax.set_facecolor('none')  # 坐标轴面板透明
        ax.xaxis.pane.fill = False  # X轴面板透明
        ax.yaxis.pane.fill = False  # Y轴面板透明
        ax.zaxis.pane.fill = False  # Z轴面板透明
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))  # 淡灰色半透明
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        # 网格线也设置为淡色
        ax.xaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)
        ax.yaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)
        ax.zaxis._axinfo['grid']['color'] = (0.8, 0.8, 0.8, 0.3)

        nz, ny, nx = volume.shape

        # 降采样以提高性能（体素太多会很慢）
        max_size = 16  # 高分辨率使体素更细腻精细

        # ... (中间省略代码) ...
        # (需要重新读取中间代码以确认上下文，这里我假设中间代码没变)
        # 不，replace_file_content 必须匹配 TargetContent。

        # 既然比较长，我分两步。第一步修改 def 行。 第二步修改 colors 计算。
        pass
        if nx > max_size or ny > max_size or nz > max_size:
            from scipy.ndimage import zoom as nd_zoom
            scale = min(max_size/nx, max_size/ny, max_size/nz)
            vol_ds = nd_zoom(volume, scale, order=1)
        else:
            vol_ds = volume

        nz_ds, ny_ds, nx_ds = vol_ds.shape

        # 使用更鲁棒的阈值计算方法
        flat_vol = vol_ds.flatten()

        # 统计信息
        data_min, data_max = flat_vol.min(), flat_vol.max()
        data_range = data_max - data_min

        # 调试输出
        print(f"[Voxel Debug] {title}")
        print(f"  Data range: [{data_min:.2f}, {data_max:.2f}]")

        if data_range < 1e-6:
            ax.set_title(f"{title}\n(No significant anomaly)")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Depth (m)')
            return

        # 阈值逻辑：显示高于正阈值或低于负阈值的区域
        # 支持正负密度同时显示
        value_range = vmax - vmin
        if value_range < 1e-6:
            value_range = 1.0
        
        # 计算正负阈值
        center = (vmax + vmin) / 2
        thresh_range = (vmax - vmin) * threshold / 2
        pos_thresh = center + thresh_range
        neg_thresh = center - thresh_range
        
        # 根据show_negative决定是否显示负异常
        if show_negative:
            filled = (vol_ds > pos_thresh) | (vol_ds < neg_thresh)
            print(f"  Threshold: pos>{pos_thresh:.2f} or neg<{neg_thresh:.2f} (threshold={threshold:.0%})")
        else:
            # 只显示正异常
            filled = vol_ds > pos_thresh
            print(f"  Threshold: >{pos_thresh:.2f} (threshold={threshold:.0%}, no negative)")

        print(f"  Filled voxels: {filled.sum()} / {filled.size} ({100*filled.sum()/filled.size:.1f}%)")

        if filled.sum() == 0:
            ax.set_title(f"{title}\n(No voxels above threshold)")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Depth (m)')
            return

        # 创建颜色数组（4D: nz, ny, nx, 4）
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = np.zeros((*vol_ds.shape, 4))  # RGBA

        for iz in range(nz_ds):
            for iy in range(ny_ds):
                for ix in range(nx_ds):
                    if filled[iz, iy, ix]:
                        val = vol_ds[iz, iy, ix]
                        norm_val = norm(val)
                        if reverse_color:
                            norm_val = 1.0 - norm_val
                        colors[iz, iy, ix] = cm.jet(norm_val)

        # 创建坐标网格 - 从0开始
        x_extent = x_range[1] - x_range[0]
        y_extent = y_range[1] - y_range[0]
        
        x = np.linspace(0, x_extent, nx_ds + 1)
        y = np.linspace(0, y_extent, ny_ds + 1)
        z = np.linspace(0, z_extent, nz_ds + 1)

        # 创建 3D 网格
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 转置以匹配 voxels 的预期格式 (需要 nx, ny, nz)
        # ax.voxels 期望的 filled 形状是 (nx, ny, nz)
        filled_t = np.transpose(filled, (2, 1, 0))  # (nz, ny, nx) -> (nx, ny, nz)
        colors_t = np.transpose(colors, (2, 1, 0, 3))  # (nz, ny, nx, 4) -> (nx, ny, nz, 4)

        # 绘制体素
        ax.voxels(X, Y, Z, filled_t, facecolors=colors_t, edgecolor='none', alpha=0.8)

        # 添加 colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array(vol_ds)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
        cbar.ax.tick_params(direction='in')  # 刻度线在色条内部
        cbar.ax.set_title('kg/m³', fontsize=9)  # 单位在色条顶部

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Depth (m)')
        
        # 强制坐标从0开始
        ax.set_xlim(0, x_extent)
        ax.set_ylim(0, y_extent)
        ax.set_zlim(0, z_extent)
        
        # 不反转Z轴，Z=0在底部，深度向上增大
        ax.view_init(elev=25, azim=225)  # Z轴在左边

    @staticmethod
    def plot_3d_isosurface(ax, volume: np.ndarray, x_range: Tuple, y_range: Tuple, z_extent: float,
                           vmin: float, vmax: float, threshold: float = 0.3, title: str = "",
                           show_negative: bool = False):
        """
        绘制 3D 等值面图（Isosurface）- 比体素更平滑专业

        Args:
            volume: 3D 密度数据 (nz, ny, nx) [kg/m³]
            threshold: 显示阈值 (相对于异常的百分比)
            title: 标题
        """
        ax.clear()
        from matplotlib import cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        try:
            from skimage import measure
        except ImportError:
            print("[Isosurface] skimage not available, falling back to voxels")
            Visualizer.plot_3d_voxels(ax, volume, x_range, y_range, z_extent, vmin, vmax, threshold, title)
            return

        # 设置3D背景透明
        ax.set_facecolor('none')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))

        nz, ny, nx = volume.shape

        # 计算阈值
        data_min, data_max = volume.min(), volume.max()
        data_range = data_max - data_min

        print(f"[Isosurface Debug] {title}")
        print(f"  Data range: [{data_min:.2f}, {data_max:.2f}]")

        if data_range < 1e-6:
            ax.set_title(f"{title}\n(No significant anomaly)")
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Depth (m)')
            return

        # 计算等值面阈值 - 对称处理正负异常
        value_range = vmax - vmin
        if value_range < 1e-6:
            value_range = 1.0  # 避免除以零
        
        # 计算正负阈值（对称）
        center = (vmax + vmin) / 2
        thresh_range = value_range * threshold / 2
        # 正异常等值面：显示密度 > center + thresh_range 的区域
        pos_level = center + thresh_range
        # 负异常等值面：显示密度 < center - thresh_range 的区域
        neg_level = center - thresh_range

        print(f"  vmin={vmin:.2f}, vmax={vmax:.2f}, center={center:.2f}")
        print(f"  pos_level={pos_level:.2f}, neg_level={neg_level:.2f}")

        # 计算网格间距 - volume 是 (nz, ny, nx)
        # marching_cubes spacing 对应 (z, y, x)
        spacing_z = z_extent / nz
        spacing_y = (y_range[1] - y_range[0]) / ny
        spacing_x = (x_range[1] - x_range[0]) / nx
        spacing = (spacing_z, spacing_y, spacing_x)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        surfaces_drawn = 0

        # 绘制正异常等值面
        if pos_level < data_max:
            try:
                verts, faces, normals, values = measure.marching_cubes(volume, level=pos_level, spacing=spacing)
                # verts 是 (z, y, x) 格式，转换为 matplotlib 3D坐标
                verts_plot = np.zeros_like(verts)
                verts_plot[:, 0] = verts[:, 2] + x_range[0]  # X = x + offset
                verts_plot[:, 1] = verts[:, 1] + y_range[0]  # Y = y + offset
                verts_plot[:, 2] = verts[:, 0]               # Z = depth

                # 使用顶点值为每个面着色
                face_values = values[faces].mean(axis=1)  # 每个面的平均值
                face_colors = cm.jet(norm(face_values))

                mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
                mesh.set_facecolor(face_colors)
                mesh.set_edgecolor('none')
                ax.add_collection3d(mesh)
                surfaces_drawn += 1
                print(f"  Positive isosurface: {len(faces)} faces at level {pos_level:.1f}")
            except Exception as e:
                print(f"  Positive isosurface failed: {e}")

        # 绘制负异常等值面（仅当show_negative=True时）
        if show_negative and neg_level > data_min:
            try:
                verts, faces, normals, values = measure.marching_cubes(volume, level=neg_level, spacing=spacing)
                # verts 是 (z, y, x) 格式，转换为 matplotlib 3D坐标
                verts_plot = np.zeros_like(verts)
                verts_plot[:, 0] = verts[:, 2] + x_range[0]  # X = x + offset
                verts_plot[:, 1] = verts[:, 1] + y_range[0]  # Y = y + offset
                verts_plot[:, 2] = verts[:, 0]               # Z = depth

                # 负异常使用vmin值着色（显示为蓝色）
                face_colors = cm.jet(norm(vmin))

                mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
                mesh.set_facecolor(face_colors)
                mesh.set_edgecolor('none')
                ax.add_collection3d(mesh)
                surfaces_drawn += 1
                print(f"  Negative isosurface: {len(faces)} faces at level {neg_level:.1f}")
            except Exception as e:
                print(f"  Negative isosurface failed: {e}")

        if surfaces_drawn == 0:
            ax.text((x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2, z_extent/2,
                    "No isosurface\n(adjust threshold)", ha='center', va='center', fontsize=10)

        # 设置坐标轴范围
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(0, z_extent)

        # 添加 colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array([vmin, vmax])
        try:
            fig = ax.get_figure()
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(direction='in')
            cbar.ax.set_title('kg/m³', fontsize=9)
        except Exception as e:
            print(f"  Colorbar failed: {e}")

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Depth (m)')
        # 不反转Z轴：深度0在下方，最大深度在上方
        ax.set_title("") # 移除标题
        
        # 设置密网格效果 - 使背景网格可见
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        
        ax.xaxis._axinfo['grid']['linewidth'] = 0.5
        ax.yaxis._axinfo['grid']['linewidth'] = 0.5
        ax.zaxis._axinfo['grid']['linewidth'] = 0.5
        ax.xaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.3, 0.6)
        ax.yaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.3, 0.6)
        ax.zaxis._axinfo['grid']['color'] = (0.3, 0.3, 0.3, 0.6)
        
        # 设置更多刻度以获得密网格
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax.zaxis.set_major_locator(plt.MaxNLocator(8))
        
        # 绘制XYZ三个背景平面的密网格线
        grid_num = 15  # 网格线数量
        grid_color = (0.5, 0.5, 0.5, 0.3)  # 灰色半透明
        
        # 底部网格 (z=0 平面)
        x_grid = np.linspace(x_range[0], x_range[1], grid_num)
        y_grid = np.linspace(y_range[0], y_range[1], grid_num)
        X_floor, Y_floor = np.meshgrid(x_grid, y_grid)
        Z_floor = np.zeros_like(X_floor)
        ax.plot_wireframe(X_floor, Y_floor, Z_floor, color=grid_color, linewidth=0.3)
        
        # 后墙网格 (y=y_min 平面)
        z_grid = np.linspace(0, z_extent, grid_num)
        X_back, Z_back = np.meshgrid(x_grid, z_grid)
        Y_back = np.full_like(X_back, y_range[0])
        ax.plot_wireframe(X_back, Y_back, Z_back, color=grid_color, linewidth=0.3)
        
        # 左墙网格 (x=x_min 平面)
        Y_left, Z_left = np.meshgrid(y_grid, z_grid)
        X_left = np.full_like(Y_left, x_range[0])
        ax.plot_wireframe(X_left, Y_left, Z_left, color=grid_color, linewidth=0.3)

        # 调整 3D 轴比例，使 Z 轴在视觉上更长
        x_len = x_range[1] - x_range[0]
        y_len = y_range[1] - y_range[0]
        z_len = z_extent
        max_len = max(x_len, y_len, z_len)
        # Z 轴放大因子
        z_scale = max(1.0, max_len / z_len * 0.6) if z_len > 0 else 1.0
        ax.set_box_aspect([x_len / max_len, y_len / max_len, z_len / max_len * z_scale])

        ax.view_init(elev=25, azim=225)



# ============== 7. 主窗口 ==============
class V4GUIProWindow(QMainWindow):
    """V4 反演专业版主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("V4 Gravity Inversion Pro")
        self.resize(1400, 900)

        # 管理器
        self.data_mgr = DataManager()
        self.model_mgr = ModelManager()

        # 状态
        self.pred_norm: Optional[np.ndarray] = None
        self.pred_real: Optional[np.ndarray] = None
        self.scale_factor: float = 1.0

        # 参数
        self.rho_min = -150.0
        self.rho_max = 350.0
        self.invert_sign = True
        self.dz = 100.0

        self._init_ui()
        self._setup_status_bar()

    def _init_ui(self):
        """初始化 UI"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧控制面板
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        # 右侧可视化区域
        viz_widget = self._create_viz_area()
        main_layout.addWidget(viz_widget, stretch=1)

    def _create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # === 1. 模型加载 ===
        gb_model = QGroupBox("1. 模型")
        vl = QVBoxLayout()

        self.btn_load_model = QPushButton("📂 加载模型 (.pth)")
        self.btn_load_model.clicked.connect(self._on_load_model)
        vl.addWidget(self.btn_load_model)

        self.lbl_model = QLabel("未加载")
        self.lbl_model.setStyleSheet("color: gray;")
        vl.addWidget(self.lbl_model)

        gb_model.setLayout(vl)
        layout.addWidget(gb_model)

        # === 2. 数据加载 ===
        gb_data = QGroupBox("2. 数据")
        vl = QVBoxLayout()

        self.btn_load_data = QPushButton("📂 加载数据 (txt/vti)")
        self.btn_load_data.clicked.connect(self._on_load_data)
        vl.addWidget(self.btn_load_data)

        self.lbl_data = QLabel("未加载")
        self.lbl_data.setStyleSheet("color: gray;")
        vl.addWidget(self.lbl_data)

        gb_data.setLayout(vl)
        layout.addWidget(gb_data)

        # === 3. 参数设置 ===
        gb_params = QGroupBox("3. 参数")
        gl = QGridLayout()

        gl.addWidget(QLabel("密度范围:"), 0, 0)
        self.ed_rho_min = QLineEdit("-150")
        self.ed_rho_min.setValidator(QDoubleValidator())
        self.ed_rho_max = QLineEdit("350")
        self.ed_rho_max.setValidator(QDoubleValidator())
        gl.addWidget(self.ed_rho_min, 0, 1)
        gl.addWidget(QLabel("~"), 0, 2)
        gl.addWidget(self.ed_rho_max, 0, 3)
        gl.addWidget(QLabel("kg/m³"), 0, 4)

        self.chk_invert = QCheckBox("符号反转")
        self.chk_invert.setChecked(True)
        self.chk_invert.stateChanged.connect(self._on_param_changed)
        gl.addWidget(self.chk_invert, 1, 0, 1, 5)

        self.chk_scale = QCheckBox("Gzz 残差缩放")
        self.chk_scale.setChecked(True)
        gl.addWidget(self.chk_scale, 2, 0, 1, 2)

        # 自定义缩放因子输入框
        self.ed_scale_factor = QLineEdit()
        self.ed_scale_factor.setPlaceholderText("自动")
        self.ed_scale_factor.setValidator(QDoubleValidator(0.1, 10.0, 3))
        self.ed_scale_factor.setToolTip("留空自动计算，或输入0.5~5.0的自定义缩放因子")
        gl.addWidget(self.ed_scale_factor, 2, 2, 1, 2)
        gl.addWidget(QLabel("倍"), 2, 4)

        # === 添加 Gzz 噪声控制 ===
        self.chk_add_noise = QCheckBox("Gzz 加噪声")
        self.chk_add_noise.setChecked(False)
        self.chk_add_noise.setToolTip("为 Gzz 重力梯度场添加高斯噪声，模拟观测误差")
        self.chk_add_noise.stateChanged.connect(self._on_refresh)
        gl.addWidget(self.chk_add_noise, 3, 0, 1, 2)

        self.ed_noise_level = QLineEdit("3")
        self.ed_noise_level.setValidator(QDoubleValidator(0.1, 50.0, 1))
        self.ed_noise_level.setToolTip("噪声水平百分比 (1-50%)")
        self.ed_noise_level.editingFinished.connect(self._on_refresh)
        gl.addWidget(self.ed_noise_level, 3, 2, 1, 1)
        gl.addWidget(QLabel("%"), 3, 3)

        gb_params.setLayout(gl)
        layout.addWidget(gb_params)

        # === 4. 运行 ===
        self.btn_run = QPushButton("🚀 运行反演")
        self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                font-size: 14px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #388E3C; }
        """)
        self.btn_run.clicked.connect(self._on_run_inversion)
        layout.addWidget(self.btn_run)

        # === 5. 切片控制 ===
        gb_slice = QGroupBox("5. 切片控制 (米)")
        gl_slice = QGridLayout()

        # X 切片
        self.chk_show_x = QCheckBox("X:")
        self.chk_show_x.setChecked(True)
        self.chk_show_x.stateChanged.connect(self._on_slice_changed)
        self.ed_slice_x = QLineEdit("2000")
        self.ed_slice_x.setValidator(QDoubleValidator())
        self.ed_slice_x.editingFinished.connect(self._on_slice_changed)
        gl_slice.addWidget(self.chk_show_x, 0, 0)
        gl_slice.addWidget(self.ed_slice_x, 0, 1)
        gl_slice.addWidget(QLabel("m"), 0, 2)

        # Y 切片
        self.chk_show_y = QCheckBox("Y:")
        self.chk_show_y.setChecked(True)
        self.chk_show_y.stateChanged.connect(self._on_slice_changed)
        self.ed_slice_y = QLineEdit("2000")
        self.ed_slice_y.setValidator(QDoubleValidator())
        self.ed_slice_y.editingFinished.connect(self._on_slice_changed)
        gl_slice.addWidget(self.chk_show_y, 1, 0)
        gl_slice.addWidget(self.ed_slice_y, 1, 1)
        gl_slice.addWidget(QLabel("m"), 1, 2)

        # Z 切片 (深度)
        self.chk_show_z = QCheckBox("Z (深度):")
        self.chk_show_z.setChecked(True)
        self.chk_show_z.stateChanged.connect(self._on_slice_changed)
        self.ed_slice_z = QLineEdit("300")
        self.ed_slice_z.setValidator(QDoubleValidator())
        self.ed_slice_z.editingFinished.connect(self._on_slice_changed)
        gl_slice.addWidget(self.chk_show_z, 2, 0)
        gl_slice.addWidget(self.ed_slice_z, 2, 1)
        gl_slice.addWidget(QLabel("m"), 2, 2)

        # 体素阈值
        gl_slice.addWidget(QLabel("阈值:"), 3, 0)
        self.sl_thresh = QSlider(Qt.Horizontal)
        self.sl_thresh.setRange(10, 90)  # 10% ~ 90%
        self.sl_thresh.setValue(30)
        self.sl_thresh.valueChanged.connect(self._on_thresh_changed)
        self.lbl_thresh = QLabel("30%")
        gl_slice.addWidget(self.sl_thresh, 3, 1)
        gl_slice.addWidget(self.lbl_thresh, 3, 2)

        # 颜色反转
        self.chk_reverse_color = QCheckBox("反转颜色")
        self.chk_reverse_color.setChecked(False)
        self.chk_reverse_color.stateChanged.connect(self._on_refresh)
        gl_slice.addWidget(self.chk_reverse_color, 4, 0, 1, 3)

        # 强制真实模型最大值
        self.chk_force_true_max = QCheckBox("真实模型强制最大值")
        self.chk_force_true_max.setChecked(False)
        self.chk_force_true_max.setToolTip("将VTI真实密度的高值区域强制显示为colorbar最大值颜色")
        self.chk_force_true_max.stateChanged.connect(self._on_refresh)
        gl_slice.addWidget(self.chk_force_true_max, 5, 0, 1, 3)

        # 显示负异常
        self.chk_show_negative = QCheckBox("显示负异常")
        self.chk_show_negative.setChecked(False)
        self.chk_show_negative.setToolTip("勾选后同时显示正负密度异常（适用于正负密度模型）")
        self.chk_show_negative.stateChanged.connect(self._on_refresh)
        gl_slice.addWidget(self.chk_show_negative, 6, 0, 1, 3)

        gb_slice.setLayout(gl_slice)
        layout.addWidget(gb_slice)

        # === 6. 刷新视图 ===
        self.btn_refresh = QPushButton("🔄 刷新视图")
        self.btn_refresh.clicked.connect(self._on_refresh)
        layout.addWidget(self.btn_refresh)

        # === 7. 导出 ===
        self.btn_save = QPushButton("💾 保存图像")
        self.btn_save.clicked.connect(self._on_save)
        layout.addWidget(self.btn_save)

        layout.addStretch()
        return panel

    def _add_slider(self, layout, name: str, max_val: int, default: int) -> Tuple[QSlider, QLabel]:
        """添加滑块控件"""
        h = QHBoxLayout()
        lbl_name = QLabel(f"{name}:")
        lbl_name.setFixedWidth(60)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(self._on_slider_changed)

        lbl_val = QLabel(str(default))
        lbl_val.setFixedWidth(40)

        h.addWidget(lbl_name)
        h.addWidget(slider)
        h.addWidget(lbl_val)
        layout.addLayout(h)

        return slider, lbl_val

    def _create_viz_area(self) -> QWidget:
        """创建可视化区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Tab 切换 2D/3D
        self.tabs = QTabWidget()

        # 2D 标签页
        self.fig_2d, self.axes_2d = Visualizer.create_figure_2x2()
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.tabs.addTab(self.canvas_2d, "2D 切片")

        # 真实2D切片标签页 (新增 - 显示VTI真实密度)
        self.fig_true_2d, self.axes_true_2d = Visualizer.create_figure_2x2()
        self.canvas_true_2d = FigureCanvas(self.fig_true_2d)
        self.tabs.addTab(self.canvas_true_2d, "真实2D切片")

        # Gzz 残差标签页 (新增)
        self.fig_gzz_residual = Figure(figsize=(14, 5))
        self.fig_gzz_residual.patch.set_facecolor('white')
        self.canvas_gzz_residual = FigureCanvas(self.fig_gzz_residual)
        self.tabs.addTab(self.canvas_gzz_residual, "Gzz 残差")

        # 3D 切片标签页
        self.fig_3d = Figure(figsize=(12, 10))
        self.fig_3d.patch.set_facecolor('white')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.tabs.addTab(self.canvas_3d, "3D 切片")

        # 3D 真实切片标签页 (新增)
        self.fig_true_3d = Figure(figsize=(12, 10))
        self.fig_true_3d.patch.set_facecolor('white')
        self.canvas_true_3d = FigureCanvas(self.fig_true_3d)
        self.tabs.addTab(self.canvas_true_3d, "3D 真实切片")

        # 3D 体素标签页 - 使用 Matplotlib（避免PyVista数据顺序问题）
        self.fig_voxel = Figure(figsize=(12, 10))
        self.fig_voxel.patch.set_facecolor('white')
        self.canvas_voxel = FigureCanvas(self.fig_voxel)
        self.tabs.addTab(self.canvas_voxel, "3D 体素")
        self.plotter = None  # 不使用PyVista

        # 3D 等值面标签页
        self.fig_isosurface = Figure(figsize=(12, 10))
        self.fig_isosurface.patch.set_facecolor('white')
        self.canvas_isosurface = FigureCanvas(self.fig_isosurface)
        self.tabs.addTab(self.canvas_isosurface, "3D 等值面")

        layout.addWidget(self.tabs)
        return widget

    def _setup_status_bar(self):
        """设置状态栏"""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪 | 请加载模型和数据")

    # ============== 事件处理 ==============
    def _on_load_model(self):
        """加载模型"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch (*.pth)")
        if path:
            if self.model_mgr.load_model(path):
                model_name = os.path.basename(path)
                mode = self.model_mgr.data_mode
                depth = self.model_mgr.grid_shape[0]
                self.lbl_model.setText(f"OK {model_name}\nmode={mode}, D={depth}")
                self.lbl_model.setStyleSheet("color: green; font-weight: bold;")
                self.status.showMessage(f"Model loaded: {path}")
            else:
                detail = self.model_mgr.last_error or "Unknown error"
                if "pytorchstreamreader" in detail.lower() or "corrupted" in detail.lower() or "byteorder" in detail.lower():
                    model_dir = os.path.dirname(path)
                    candidates = [
                        os.path.join(model_dir, "latest.pth"),
                        os.path.join(os.path.dirname(model_dir), "best_model.pth"),
                    ]
                    for cand in candidates:
                        if os.path.isfile(cand) and os.path.abspath(cand) != os.path.abspath(path):
                            if self.model_mgr.load_model(cand):
                                model_name = os.path.basename(cand)
                                mode = self.model_mgr.data_mode
                                depth = self.model_mgr.grid_shape[0]
                                self.lbl_model.setText(f"OK {model_name}\nmode={mode}, D={depth}")
                                self.lbl_model.setStyleSheet("color: green; font-weight: bold;")
                                self.status.showMessage(f"Model loaded (fallback): {cand}")
                                QMessageBox.information(self, "Recovered", f"Selected checkpoint is corrupted.\nLoaded fallback: {cand}")
                                return
                QMessageBox.critical(self, "Error", f"Model load failed\n{detail}")

    def _on_load_data(self):
        """加载数据"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "",
            "All Supported (*.txt *.csv *.mat *.vti);;Text (*.txt *.csv);;VTI (*.vti);;MAT (*.mat)"
        )
        if path:
            # 使用统一的 load_file 方法自动选择加载器
            if self.data_mgr.load_file(path):
                shape_str = str(self.data_mgr.original_shape)
                if hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None:
                    shape_str = f"3D: {self.data_mgr.density_3d.shape}"

                    # 自动调节范围
                    d_min, d_max = self.data_mgr.density_3d.min(), self.data_mgr.density_3d.max()
                    # 如果范围太小，稍微扩大一点避免除以零
                    if d_max - d_min < 1e-6:
                        d_min -= 0.1
                        d_max += 0.1

                    self.ed_rho_min.setText(f"{d_min:.1f}")
                    self.ed_rho_max.setText(f"{d_max:.1f}")
                    self.status.showMessage(f"已自动更新显示范围: [{d_min:.1f}, {d_max:.1f}]")

                self.lbl_data.setText(f"✓ {os.path.basename(path)}\n{shape_str}")
                self.lbl_data.setStyleSheet("color: blue; font-weight: bold;")
                self.status.showMessage(f"数据已加载: {path}")
                self._update_gzz_display()
                
                # 如果加载的是VTI文件（包含3D密度），立即更新相关视图
                if hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None:
                    # 获取切片参数
                    try:
                        x_m = float(self.ed_slice_x.text())
                        y_m = float(self.ed_slice_y.text())
                        z_m = float(self.ed_slice_z.text())
                        vmin = float(self.ed_rho_min.text())
                        vmax = float(self.ed_rho_max.text())
                    except:
                        x_m, y_m, z_m = 2000, 2000, 300
                        vmin, vmax = 0, 0.5
                    
                    x_range = self.data_mgr.x_range
                    y_range = self.data_mgr.y_range
                    
                    # 更新真实2D切片视图
                    self._update_true_2d_slice_view(x_m, y_m, z_m, vmin, vmax, x_range, y_range)
                    self._update_isosurface_view()
                    self._update_voxel_view()
            else:
                QMessageBox.critical(self, "错误", "数据加载失败")

    def _on_param_changed(self):
        """参数变化"""
        self.invert_sign = self.chk_invert.isChecked()
        if self.pred_norm is not None:
            self._process_prediction()
            self._on_refresh()

    def _on_slice_changed(self):
        """切片输入或可见性变化"""
        if self.pred_real is not None:
            self._update_display_only()

    def _get_slice_indices(self) -> Tuple[float, float, float, bool, bool, bool]:
        """
        从米输入获取精确切片位置和可见性
        Returns: (x_m, y_m, z_m, show_x, show_y, show_z) - 单位为米
        """
        if self.pred_real is None:
            return 2000.0, 2000.0, 300.0, True, True, True

        nz, ny, nx = self.pred_real.shape
        x_range = self.data_mgr.x_range
        y_range = self.data_mgr.y_range
        z_extent = nz * self.dz

        # 获取精确的米值
        try:
            x_m = float(self.ed_slice_x.text())
            x_m = max(x_range[0], min(x_range[1], x_m))
        except:
            x_m = (x_range[0] + x_range[1]) / 2

        try:
            y_m = float(self.ed_slice_y.text())
            y_m = max(y_range[0], min(y_range[1], y_m))
        except:
            y_m = (y_range[0] + y_range[1]) / 2

        try:
            z_m = float(self.ed_slice_z.text())
            z_m = max(0, min(z_extent, z_m))
        except:
            z_m = z_extent / 4

        show_x = self.chk_show_x.isChecked()
        show_y = self.chk_show_y.isChecked()
        show_z = self.chk_show_z.isChecked()

        return x_m, y_m, z_m, show_x, show_y, show_z

    def _on_thresh_changed(self):
        """阈值滑块变化"""
        val = self.sl_thresh.value()
        self.lbl_thresh.setText(f"{val}%")

        # 只更新体素视图
        if self.pred_real is not None:
            self._update_voxel_view()

    def _on_run_inversion(self):
        """运行反演"""
        # 检查前置条件
        if not self.model_mgr.is_loaded:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        if self.data_mgr.gzz_32 is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        try:
            self.status.showMessage("正在运行反演...")
            QApplication.processEvents()

            # 构建输入
            input_tensor = self.data_mgr.get_input_tensor(
                nz=self.model_mgr.grid_shape[0],
                data_mode=self.model_mgr.data_mode,
                target_hw=(self.model_mgr.grid_shape[1], self.model_mgr.grid_shape[2])
            )
            if input_tensor is None:
                raise ValueError("输入张量构建失败")

            # 推理
            self.pred_norm = self.model_mgr.predict(input_tensor)
            if self.pred_norm is None:
                raise ValueError("模型推理失败")

            print(f"✓ Prediction: shape={self.pred_norm.shape}, range=[{self.pred_norm.min():.3f}, {self.pred_norm.max():.3f}]")

            # 处理预测结果
            self._process_prediction()

            # 更新显示
            self._on_refresh()

            # 如果是 VTI 文件，生成高分辨率可视化输出
            if hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None and HighResVisualizer is not None:
                self._save_highres_vti_result()

            self.status.showMessage(f"反演完成 | 缩放因子: {self.scale_factor:.3f}")
            QMessageBox.information(self, "完成", f"反演完成!\n缩放因子: {self.scale_factor:.3f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"反演失败: {e}")
            self.status.showMessage("反演失败")

    def _save_highres_vti_result(self):
        """保存 VTI 文件的高分辨率可视化结果"""
        try:
            from matplotlib.gridspec import GridSpec
            from scipy.ndimage import zoom as nd_zoom

            # 获取真实密度（VTI中加载的）和预测密度
            true_density_raw = self.data_mgr.density_3d  # 归一化的 [-1, 1] 或实际密度
            pred_density = self.pred_real  # 真实密度 kg/m³

            if true_density_raw is None or pred_density is None:
                print("⚠ 无法生成高分辨率可视化：缺少数据")
                return

            # 获取网格参数
            dx = self.data_mgr.dx
            dz = getattr(self.data_mgr, 'dz', 100.0)

            print(f"[HighRes] True density shape: {true_density_raw.shape}, range: [{true_density_raw.min():.3f}, {true_density_raw.max():.3f}]")
            print(f"[HighRes] Pred density shape: {pred_density.shape}, range: [{pred_density.min():.1f}, {pred_density.max():.1f}]")

            # 调整尺寸匹配
            if true_density_raw.shape != pred_density.shape:
                scale_factors = [pred_density.shape[i] / true_density_raw.shape[i] for i in range(3)]
                true_density_resized = nd_zoom(true_density_raw, scale_factors, order=1)
                print(f"[HighRes] Resized true density: {true_density_resized.shape}")
            else:
                true_density_resized = true_density_raw.copy()

            # 自动处理密度映射（支持负密度）
            # 使用预测密度的实际范围，不假设正值
            pred_min, pred_max = pred_density.min(), pred_density.max()
            true_min, true_max = true_density_resized.min(), true_density_resized.max()

            # 检测归一化值（|max| <= 1.5）
            if np.abs(true_density_resized).max() <= 1.5:
                if true_max - true_min > 0.01:
                    # 线性映射到预测范围（保持负值）
                    true_density_kg = (true_density_resized - true_min) / (true_max - true_min) * (pred_max - pred_min) + pred_min
                else:
                    # 平坦数据，使用中间值
                    true_density_kg = true_density_resized * (pred_max - pred_min) / 2 + (pred_min + pred_max) / 2
                print(f"[HighRes] Scaled true density: [{true_density_kg.min():.1f}, {true_density_kg.max():.1f}] kg/m³")
            else:
                true_density_kg = true_density_resized
                print(f"[HighRes] True density (no scaling): [{true_density_kg.min():.1f}, {true_density_kg.max():.1f}] kg/m³")

            # 使用简化正演计算重力（从密度计算）
            try:
                from data_preparation import FastGravityForward
                nz, ny, nx = pred_density.shape
                fwd = FastGravityForward((nz, ny, nx), dx=dx, dz=dz, mode='gz')

                # 先归一化密度到相同尺度 [-1, 1]
                true_norm_for_fwd = (true_density_kg - true_density_kg.mean()) / (np.abs(true_density_kg).max() + 1e-8)
                pred_norm_for_fwd = (pred_density - pred_density.mean()) / (np.abs(pred_density).max() + 1e-8)

                # 计算归一化重力
                obs_gravity = fwd.forward(true_norm_for_fwd)
                pred_gravity = fwd.forward(pred_norm_for_fwd)

                # 动态归一化重力到相同范围
                obs_max = max(abs(obs_gravity.min()), abs(obs_gravity.max()), 1e-8)
                pred_max_g = max(abs(pred_gravity.min()), abs(pred_gravity.max()), 1e-8)

                obs_gravity = obs_gravity / obs_max
                pred_gravity = pred_gravity / pred_max_g

                # 符号校正：如果预测与观测相关性为负，则反转预测
                corr = np.corrcoef(obs_gravity.flatten(), pred_gravity.flatten())[0, 1]
                if corr < 0:
                    pred_gravity = -pred_gravity
                    print(f"[HighRes] Sign corrected (correlation was {corr:.3f})")

                print(f"[HighRes] Obs gravity (normalized): [{obs_gravity.min():.3f}, {obs_gravity.max():.3f}]")
                print(f"[HighRes] Pred gravity (normalized): [{pred_gravity.min():.3f}, {pred_gravity.max():.3f}]")

            except Exception as e:
                print(f"[HighRes] Forward failed: {e}, using gradient approximation")
                # 使用梯度近似
                from scipy.ndimage import gaussian_filter
                obs_gravity = gaussian_filter(true_density_kg.sum(axis=0), sigma=2)
                pred_gravity = gaussian_filter(pred_density.sum(axis=0), sigma=2)

            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vti_results")
            os.makedirs(save_dir, exist_ok=True)

            # 归一化密度用于可视化
            # 重要：使用相同的归一化因子，并确保符号一致

            # 先检查符号是否需要反转
            corr = np.corrcoef(true_density_kg.flatten(), pred_density.flatten())[0, 1]
            pred_aligned = pred_density if corr >= 0 else -pred_density

            # 使用共同的最大值进行归一化
            combined_max = max(
                abs(true_density_kg.min()), abs(true_density_kg.max()),
                abs(pred_aligned.min()), abs(pred_aligned.max())
            )
            if combined_max > 0:
                true_norm = true_density_kg / combined_max
                pred_norm = pred_aligned / combined_max
            else:
                true_norm = true_density_kg
                pred_norm = pred_aligned

            # ===== 使用 HighResVisualizer 生成高分辨率可视化 =====
            visualizer = HighResVisualizer(save_dir, dpi=200, upsampling=2)
            save_path = visualizer.save_epoch_result(
                epoch=0,
                obs_gravity=obs_gravity,
                pred_gravity=pred_gravity,
                true_density=true_norm,
                pred_density=pred_norm,
                dx=dx,
                dz=dz
            )
            print(f"✓ 高分辨率可视化已保存: {save_path}")

            # ===== 额外生成 XYZ 切片对比图（使用归一化密度确保统一色标）=====
            # 将两者归一化到相同范围 [0, 1] 用于可视化对比
            true_for_vis = (true_density_kg - true_density_kg.min()) / (true_density_kg.max() - true_density_kg.min() + 1e-8)

            # 检测预测与真实的相关性，如果为负则反转预测
            pred_corr = np.corrcoef(true_density_kg.flatten(), pred_density.flatten())[0, 1]
            pred_adjusted = pred_density if pred_corr >= 0 else -pred_density
            if pred_corr < 0:
                print(f"[Visualization] Inverted pred density (correlation was {pred_corr:.3f})")

            pred_for_vis = (pred_adjusted - pred_adjusted.min()) / (pred_adjusted.max() - pred_adjusted.min() + 1e-8)

            # 传递真实密度范围用于colorbar显示
            density_range = (true_density_kg.min(), true_density_kg.max())
            self._save_xyz_slice_comparison(true_for_vis, pred_for_vis, dx, dz, save_dir, density_range)

            # ===== 分别保存体素图 (带XYZ密网格) =====
            threshold = self.sl_thresh.value() / 100.0
            
            # 保存真实模型体素图
            fig_true = plt.figure(figsize=(10, 8))
            fig_true.patch.set_facecolor('white')
            ax_true = fig_true.add_subplot(111, projection='3d')
            self._plot_voxel_body_local(ax_true, true_density_kg, "", threshold)
            fig_true.tight_layout()
            path_true = os.path.join(save_dir, "voxel_true.png")
            fig_true.savefig(path_true, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig_true)
            print(f"✓ 体素图已保存: {path_true}")
            
            # 保存预测模型体素图
            fig_pred = plt.figure(figsize=(10, 8))
            fig_pred.patch.set_facecolor('white')
            ax_pred = fig_pred.add_subplot(111, projection='3d')
            self._plot_voxel_body_local(ax_pred, pred_density, "", threshold)
            fig_pred.tight_layout()
            path_pred = os.path.join(save_dir, "voxel_pred.png")
            fig_pred.savefig(path_pred, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig_pred)
            print(f"✓ 体素图已保存: {path_pred}")

            self.status.showMessage(f"反演完成 | 高分辨率图像已保存")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ 高分辨率可视化生成失败: {e}")

    def _save_voxel_comparison(self, true_density, pred_density, dx, dz, save_dir):
        """保存完整的高分辨率9格可视化图"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            from matplotlib import cm
            from scipy.ndimage import zoom as nd_zoom

            nz, ny, nx = pred_density.shape

            # 创建 3x3 布局
            fig = plt.figure(figsize=(16, 14))
            gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.2, 1.5], hspace=0.25, wspace=0.3)
            fig.suptitle("High-Res 3D Inversion Result", fontsize=14, fontweight='bold')

            # 坐标范围
            x_range = self.data_mgr.x_range
            y_range = self.data_mgr.y_range
            z_extent = nz * dz
            depth_km = nz * dz / 1000
            extent_xy = [0, nx * dx / 1000, ny * dx / 1000, 0]

            # ========== 第一行: 重力场对比 ==========
            # 使用正演算子计算重力
            try:
                from data_preparation import FastGravityForward
                fwd = FastGravityForward((nz, ny, nx), dx=dx, dz=dz, mode='gz')
                obs_gravity = fwd.forward(true_density)
                pred_gravity = fwd.forward(pred_density)
            except Exception as e:
                print(f"[Warning] FastGravityForward failed: {e}, using approximation")
                # 使用加权求和近似（考虑深度衰减）
                depth_weights = np.exp(-np.arange(nz)[:, None, None] * dz / 1000 * 0.5)
                obs_gravity = (true_density * depth_weights).sum(axis=0)
                pred_gravity = (pred_density * depth_weights).sum(axis=0)

            # 归一化到相同范围
            combined_max = max(
                abs(obs_gravity.min()), abs(obs_gravity.max()),
                abs(pred_gravity.min()), abs(pred_gravity.max()), 1e-8
            )
            obs_gravity = obs_gravity / combined_max
            pred_gravity = pred_gravity / combined_max

            # 上采样
            obs_up = nd_zoom(obs_gravity, 2, order=1)
            pred_up = nd_zoom(pred_gravity, 2, order=1)
            residual = obs_gravity - pred_gravity
            res_up = nd_zoom(residual, 2, order=1)

            vmax_g = max(abs(obs_up.max()), abs(obs_up.min()))

            # 观测重力
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(obs_up, cmap='jet', extent=extent_xy, vmin=-vmax_g, vmax=vmax_g)
            ax1.set_title("Observed Gravity (Normalized)", fontsize=10)
            ax1.set_xlabel("X (km)")
            ax1.set_ylabel("Y (km)")
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(direction='in')

            # 预测重力
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(pred_up, cmap='jet', extent=extent_xy, vmin=-vmax_g, vmax=vmax_g)
            ax2.set_title("Predicted Gravity (Normalized)", fontsize=10)
            ax2.set_xlabel("X (km)")
            ax2.set_ylabel("Y (km)")
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(direction='in')

            # 残差
            ax3 = fig.add_subplot(gs[0, 2])
            res_max = max(abs(res_up.max()), abs(res_up.min()), 1e-8)
            im3 = ax3.imshow(res_up, cmap='RdBu_r', extent=extent_xy, vmin=-res_max, vmax=res_max)
            ax3.set_title("Residuals", fontsize=10)
            ax3.set_xlabel("X (km)")
            ax3.set_ylabel("Y (km)")
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(direction='in')

            # ========== 第二行: 3D 平滑切片 ==========
            ax4 = fig.add_subplot(gs[1, 0], projection='3d')
            self._plot_3d_smooth_slices(ax4, true_density, dx, dz, "True Model: Smooth Slices")

            ax5 = fig.add_subplot(gs[1, 1], projection='3d')
            self._plot_3d_smooth_slices(ax5, pred_density, dx, dz, "Pred Model: Smooth Slices")

            # 信息面板
            ax_info = fig.add_subplot(gs[1, 2])
            ax_info.axis('off')
            info_text = (
                f"Info:\n"
                f"Grid: {nz}×{ny}×{nx}\n"
                f"dx: {dx}m, dz: {dz}m\n"
                f"Depth: {depth_km:.2f}km\n\n"
                f"Density Range:\n"
                f"  True: [{true_density.min():.2f}, {true_density.max():.2f}]\n"
                f"  Pred: [{pred_density.min():.2f}, {pred_density.max():.2f}]\n"
                f"  RMSE: {np.sqrt(np.mean((pred_density - true_density)**2)):.4f}"
            )
            ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # ========== 第三行: 3D 体素体 ==========
            threshold = self.sl_thresh.value() / 100.0

            ax6 = fig.add_subplot(gs[2, 0], projection='3d')
            self._plot_voxel_body_local(ax6, true_density, f"True Model: Voxel Body\n(Solid > {int(threshold*100)}%)", threshold)

            ax7 = fig.add_subplot(gs[2, 1], projection='3d')
            self._plot_voxel_body_local(ax7, pred_density, f"Pred Model: Voxel Body\n(Solid > {int(threshold*100)}%)", threshold)

            # 误差分布
            ax8 = fig.add_subplot(gs[2, 2])
            error = np.abs(pred_density - true_density)
            depth_error = error.mean(axis=(1, 2))
            depths = np.arange(nz) * dz / 1000
            ax8.barh(depths, depth_error, height=dz/1000*0.8, color='coral', edgecolor='darkred')
            ax8.set_xlabel("Mean Absolute Error", fontsize=10)
            ax8.set_ylabel("Depth (km)", fontsize=10)
            ax8.set_title("Error by Depth", fontsize=10)
            ax8.invert_yaxis()
            ax8.grid(True, alpha=0.3)

            # 保存
            save_path = os.path.join(save_dir, "highres_result.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"✓ 高分辨率可视化图已保存: {save_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ 高分辨率可视化生成失败: {e}")

    def _plot_3d_smooth_slices(self, ax, density, dx, dz, title):
        """绘制3D平滑切片"""
        nz, ny, nx = density.shape

        # 创建网格
        x = np.linspace(0, nx * dx / 1000, nx)
        y = np.linspace(0, ny * dx / 1000, ny)
        X, Y = np.meshgrid(x, y)

        # 中间层Z切片
        mid_z = nz // 2
        Z_slice = np.ones_like(X) * mid_z * dz / 1000
        slice_data = density[mid_z, :, :]

        # 归一化颜色
        norm = plt.Normalize(vmin=-1, vmax=1)
        colors = plt.cm.viridis(norm(slice_data))

        ax.plot_surface(X, Y, Z_slice, facecolors=colors, alpha=0.9, shade=True)

        # 侧面投影
        z_proj = np.linspace(0, nz * dz / 1000, nz)
        Y_proj, Z_proj = np.meshgrid(y, z_proj)
        X_proj = np.zeros_like(Y_proj)
        yz_slice = density[:, :, nx//2]
        colors_yz = plt.cm.viridis(norm(yz_slice))
        ax.plot_surface(X_proj, Y_proj, Z_proj, facecolors=colors_yz, alpha=0.6)

        ax.set_xlabel("X (km)", fontsize=8)
        ax.set_ylabel("Y (km)", fontsize=8)
        ax.set_zlabel("Depth (km)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.invert_zaxis()
        ax.view_init(elev=25, azim=-60)

    def _plot_voxel_body_local(self, ax, density, title, threshold=0.3):
        """绘制3D体素体 - 使用中位数阈值，带XYZ密网格"""
        nz, ny, nx = density.shape

        # 使用中位数作为背景
        center = np.median(density)
        anomaly_range = max(abs(density.max() - center), abs(density.min() - center))

        if anomaly_range < 0.01:
            ax.text(0.5, 0.5, 0.5, "No significant anomaly",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            return

        # 阈值
        pos_thresh = center + threshold * anomaly_range
        neg_thresh = center - threshold * anomaly_range
        voxels = (density > pos_thresh) | (density < neg_thresh)

        if not voxels.any():
            ax.text(0.5, 0.5, 0.5, "No voxels above threshold",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            return

        # 使用 jet colormap 创建颜色
        from matplotlib import cm
        vmin, vmax = density.min(), density.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = np.empty(voxels.shape + (4,))
        
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    if voxels[iz, iy, ix]:
                        val = density[iz, iy, ix]
                        colors[iz, iy, ix] = cm.jet(norm(val))
                    else:
                        colors[iz, iy, ix] = (0, 0, 0, 0)

        ax.voxels(voxels, facecolors=colors, edgecolor='white', linewidth=0.2)
        
        # 绘制XYZ三面密网格
        grid_num = 12
        grid_color = (0.5, 0.5, 0.5, 0.3)
        
        # 底部网格 (z=0)
        x_grid = np.linspace(0, nx, grid_num)
        y_grid = np.linspace(0, ny, grid_num)
        X_floor, Y_floor = np.meshgrid(x_grid, y_grid)
        Z_floor = np.zeros_like(X_floor)
        ax.plot_wireframe(X_floor, Y_floor, Z_floor, color=grid_color, linewidth=0.3)
        
        # 后墙网格 (y=0)
        z_grid = np.linspace(0, nz, grid_num)
        X_back, Z_back = np.meshgrid(x_grid, z_grid)
        Y_back = np.zeros_like(X_back)
        ax.plot_wireframe(X_back, Y_back, Z_back, color=grid_color, linewidth=0.3)
        
        # 左墙网格 (x=0)
        Y_left, Z_left = np.meshgrid(y_grid, z_grid)
        X_left = np.zeros_like(Y_left)
        ax.plot_wireframe(X_left, Y_left, Z_left, color=grid_color, linewidth=0.3)
        
        # 设置坐标轴
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.set_title("", fontsize=9)  # 移除标题
        ax.view_init(elev=25, azim=-60)
        
        # 设置轴范围
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

    def _save_xyz_slice_comparison(self, true_density, pred_density, dx, dz, save_dir, density_range=None):
        """生成 XYZ 三向切片对比图"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            nz, ny, nx = pred_density.shape

            # 使用界面控件中的切片位置（而不是自动计算异常体中心）
            cx, cy, cz, _, _, _ = self._get_slice_indices()
            # 确保索引在有效范围内
            cx = max(0, min(nx - 1, cx))
            cy = max(0, min(ny - 1, cy))
            cz = max(0, min(nz - 1, cz))

            print(f"[XYZ Slices] Using UI slice positions: z={cz}, y={cy}, x={cx}")

            # 创建 2x3 布局：第一行真实模型，第二行预测模型
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
            fig.suptitle(f"XYZ Slice Comparison (Z={cz*dz:.0f}m, Y={cy*dx:.0f}m, X={cx*dx:.0f}m)", fontsize=14)

            # 使用归一化范围[0,1]用于imshow
            vmin = 0.0
            vmax = 1.0

            # 真实密度范围用于colorbar标签
            if density_range is not None:
                real_vmin, real_vmax = density_range
                cbar_label = f'Density (kg/m³)\n[{real_vmin:.1f} - {real_vmax:.1f}]'
            else:
                cbar_label = 'Normalized'

            # 坐标范围
            x_extent = [0, nx * dx / 1000, ny * dx / 1000, 0]
            y_extent = [0, nx * dx / 1000, nz * dz / 1000, 0]
            z_extent = [0, ny * dx / 1000, nz * dz / 1000, 0]

            # ===== 第一行: 真实模型 =====
            # Z 切片
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(true_density[cz, :, :], cmap='jet', extent=x_extent, vmin=vmin, vmax=vmax)
            ax1.set_title(f"True: Z-Slice (D={cz*dz:.0f}m)")
            ax1.set_xlabel("X (km)")
            ax1.set_ylabel("Y (km)")
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, label=cbar_label)
            cbar1.ax.tick_params(direction='in')

            # Y 切片
            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(true_density[:, cy, :], cmap='jet', extent=y_extent, vmin=vmin, vmax=vmax)
            ax2.set_title(f"True: Y-Slice (Y={cy*dx:.0f}m)")
            ax2.set_xlabel("X (km)")
            ax2.set_ylabel("Depth (km)")
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, label=cbar_label)
            cbar2.ax.tick_params(direction='in')

            # X 切片
            ax3 = fig.add_subplot(gs[0, 2])
            im3 = ax3.imshow(true_density[:, :, cx], cmap='jet', extent=z_extent, vmin=vmin, vmax=vmax)
            ax3.set_title(f"True: X-Slice (X={cx*dx:.0f}m)")
            ax3.set_xlabel("Y (km)")
            ax3.set_ylabel("Depth (km)")
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, label=cbar_label)
            cbar3.ax.tick_params(direction='in')

            # ===== 第二行: 预测模型 =====
            # Z 切片
            ax4 = fig.add_subplot(gs[1, 0])
            im4 = ax4.imshow(pred_density[cz, :, :], cmap='jet', extent=x_extent, vmin=vmin, vmax=vmax)
            ax4.set_title(f"Pred: Z-Slice (D={cz*dz:.0f}m)")
            ax4.set_xlabel("X (km)")
            ax4.set_ylabel("Y (km)")
            cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8, label=cbar_label)
            cbar4.ax.tick_params(direction='in')

            # Y 切片
            ax5 = fig.add_subplot(gs[1, 1])
            im5 = ax5.imshow(pred_density[:, cy, :], cmap='jet', extent=y_extent, vmin=vmin, vmax=vmax)
            ax5.set_title(f"Pred: Y-Slice (Y={cy*dx:.0f}m)")
            ax5.set_xlabel("X (km)")
            ax5.set_ylabel("Depth (km)")
            cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.8, label=cbar_label)
            cbar5.ax.tick_params(direction='in')

            # X 切片
            ax6 = fig.add_subplot(gs[1, 2])
            im6 = ax6.imshow(pred_density[:, :, cx], cmap='jet', extent=z_extent, vmin=vmin, vmax=vmax)
            ax6.set_title(f"Pred: X-Slice (X={cx*dx:.0f}m)")
            ax6.set_xlabel("Y (km)")
            ax6.set_ylabel("Depth (km)")
            cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8, label=cbar_label)
            cbar6.ax.tick_params(direction='in')

            # 保存
            slice_path = os.path.join(save_dir, "xyz_slice_comparison.png")
            plt.savefig(slice_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"✓ XYZ切片对比图已保存: {slice_path}")

            # 生成3D空间切片图
            self._save_3d_slice_view(true_density, pred_density, dx, dz, save_dir, cz, cy, cx, density_range)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ XYZ切片对比图生成失败: {e}")

    def _save_3d_slice_view(self, true_density, pred_density, dx, dz, save_dir, cz, cy, cx, density_range=None):
        """生成 3D 空间切片对比图 - 每个方向单独显示"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from mpl_toolkits.mplot3d import Axes3D

            nz, ny, nx = pred_density.shape

            # 使用归一化范围[0,1]用于颜色映射
            vmin = 0.0
            vmax = 1.0
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            # 真实密度范围用于colorbar显示
            if density_range is not None:
                real_vmin, real_vmax = density_range

            # 创建网格坐标 (km)
            x = np.linspace(0, nx * dx / 1000, nx)
            y = np.linspace(0, ny * dx / 1000, ny)
            z = np.linspace(0, nz * dz / 1000, nz)

            # 增大图幅
            fig = plt.figure(figsize=(22, 14))

            # 视角参数：azim=60使Depth轴在图的左边
            AZIM_LEFT = 60  # Depth在左边
            ELEV = 25

            def plot_z_slice(ax, volume):
                """绘制单个Z切片在3D空间"""
                XX, YY = np.meshgrid(x, y)
                ZZ = np.full_like(XX, z[cz])
                colors = cm.jet(norm(volume[cz, :, :]))
                ax.plot_surface(XX, YY, ZZ, facecolors=colors, shade=False, alpha=0.9)
                ax.set_xlabel('X (km)', fontsize=9, labelpad=8)
                ax.set_ylabel('Y (km)', fontsize=9, labelpad=8)
                ax.set_zlabel('')  # 清空默认标签
                # 统一设置轴范围
                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(y[0], y[-1])
                ax.set_zlim(z[-1], z[0])
                # 在Z轴顶部手动添加Depth标签（水平放置）
                ax.text2D(0.02, 0.95, 'Depth (km)', transform=ax.transAxes,
                         fontsize=9, ha='left', va='top')
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.zaxis.set_major_locator(plt.MaxNLocator(4))
                ax.tick_params(axis='both', labelsize=7, pad=2)
                ax.view_init(elev=ELEV, azim=AZIM_LEFT)
                ax.set_box_aspect([1, 1, 1])  # 等比例轴

            def plot_y_slice(ax, volume):
                """绘制单个Y切片在3D空间"""
                XX_y, ZZ_y = np.meshgrid(x, z)
                YY_y = np.full_like(XX_y, y[cy])
                colors = cm.jet(norm(volume[:, cy, :]))
                ax.plot_surface(XX_y, YY_y, ZZ_y, facecolors=colors, shade=False, alpha=0.9)
                ax.set_xlabel('X (km)', fontsize=9, labelpad=8)
                ax.set_ylabel('Y (km)', fontsize=9, labelpad=8)
                ax.set_zlabel('')  # 清空默认标签
                # 统一设置轴范围
                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(y[0], y[-1])
                ax.set_zlim(z[-1], z[0])
                # 在Z轴顶部手动添加Depth标签（水平放置）
                ax.text2D(0.02, 0.95, 'Depth (km)', transform=ax.transAxes,
                         fontsize=9, ha='left', va='top')
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.zaxis.set_major_locator(plt.MaxNLocator(4))
                ax.tick_params(axis='both', labelsize=7, pad=2)
                ax.view_init(elev=ELEV, azim=AZIM_LEFT)
                ax.set_box_aspect([1, 1, 1])  # 等比例轴

            def plot_x_slice(ax, volume):
                """绘制单个X切片在3D空间 - 保持与其他切片相同的坐标系"""
                # X切片：沿X方向切，显示YZ平面
                YY_x, ZZ_x = np.meshgrid(y, z)
                XX_x = np.full_like(YY_x, x[cx])
                colors = cm.jet(norm(volume[:, :, cx]))
                # 保持与其他切片相同的坐标系统
                ax.plot_surface(XX_x, YY_x, ZZ_x, facecolors=colors, shade=False, alpha=0.9)
                ax.set_xlabel('X (km)', fontsize=9, labelpad=8)
                ax.set_ylabel('Y (km)', fontsize=9, labelpad=8)
                ax.set_zlabel('')  # 清空默认标签
                # 统一设置轴范围 - 与其他切片完全相同
                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(y[0], y[-1])
                ax.set_zlim(z[-1], z[0])
                # 在Z轴顶部手动添加Depth标签（水平放置）
                ax.text2D(0.02, 0.95, 'Depth (km)', transform=ax.transAxes,
                         fontsize=9, ha='left', va='top')
                ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                ax.zaxis.set_major_locator(plt.MaxNLocator(4))
                ax.tick_params(axis='both', labelsize=7, pad=2)
                ax.view_init(elev=ELEV, azim=AZIM_LEFT)
                ax.set_box_aspect([1, 1, 1])  # 等比例轴

            # 第一行: 真实模型
            ax1 = fig.add_subplot(2, 3, 1, projection='3d')
            plot_z_slice(ax1, true_density)

            ax2 = fig.add_subplot(2, 3, 2, projection='3d')
            plot_y_slice(ax2, true_density)

            ax3 = fig.add_subplot(2, 3, 3, projection='3d')
            plot_x_slice(ax3, true_density)

            # 第二行: 预测模型
            ax4 = fig.add_subplot(2, 3, 4, projection='3d')
            plot_z_slice(ax4, pred_density)

            ax5 = fig.add_subplot(2, 3, 5, projection='3d')
            plot_y_slice(ax5, pred_density)

            ax6 = fig.add_subplot(2, 3, 6, projection='3d')
            plot_x_slice(ax6, pred_density)

            # 为每个子图添加colorbar（显示真实密度范围）
            if density_range is not None:
                # 使用真实密度范围创建colorbar刻度
                real_vmin, real_vmax = density_range
                norm_real = plt.Normalize(vmin=real_vmin, vmax=real_vmax)
                mappable = cm.ScalarMappable(norm=norm_real, cmap='jet')
            else:
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
            mappable.set_array([])

            for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
                cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02, aspect=15)
                cbar.ax.tick_params(labelsize=7, direction='in')  # 刻度线在色条内部
                if density_range is not None:
                    cbar.ax.set_title('kg/m³', fontsize=8)

            plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.08, wspace=0.10, hspace=-0.05)  # 上移并紧凑子图

            # 保存
            slice_3d_path = os.path.join(save_dir, "3d_slice_comparison.png")
            plt.savefig(slice_3d_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"✓ 3D切片对比图已保存: {slice_3d_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠ 3D切片对比图生成失败: {e}")

    def _process_prediction(self):
        """处理预测结果"""
        if self.pred_norm is None:
            return

        # 获取参数
        try:
            self.rho_min = float(self.ed_rho_min.text())
            self.rho_max = float(self.ed_rho_max.text())
        except:
            self.rho_min, self.rho_max = -150, 350

        self.invert_sign = self.chk_invert.isChecked()
        background = (self.rho_min + self.rho_max) / 2

        # 归一化 -> 真实密度
        pred_real = DensityProcessor.normalize_to_real(
            self.pred_norm,
            self.rho_min, self.rho_max,
            self.invert_sign
        )

        # Gzz 残差缩放
        if self.chk_scale.isChecked() and self.data_mgr.gzz_32 is not None:
            print("\n--- Applying Gzz Scaling ---")

            # 检查是否有自定义缩放因子
            custom_scale_text = self.ed_scale_factor.text().strip()
            if custom_scale_text:
                try:
                    custom_alpha = float(custom_scale_text)
                    custom_alpha = np.clip(custom_alpha, 0.1, 10.0)  # 限制范围
                    print(f"  Using custom scale factor: {custom_alpha:.3f}")

                    # 手动应用缩放
                    density_anomaly = pred_real - background
                    pred_real = background + density_anomaly * custom_alpha

                    # 安全钳位
                    min_density = background - 300
                    max_density = background + 400
                    pred_real = np.clip(pred_real, min_density, max_density)

                    self.scale_factor = custom_alpha
                    print(f"  Density range: [{pred_real.min():.1f}, {pred_real.max():.1f}] kg/m³")
                except ValueError:
                    # 输入无效，使用自动计算
                    pred_real, self.scale_factor = DensityProcessor.scale_by_residual(
                        pred_real, self.data_mgr.gzz_32,
                        dx=self.data_mgr.dx, dz=self.dz,
                        background=background
                    )
            else:
                # 自动计算缩放因子
                pred_real, self.scale_factor = DensityProcessor.scale_by_residual(
                    pred_real, self.data_mgr.gzz_32,
                    dx=self.data_mgr.dx, dz=self.dz,
                    background=background
                )
        else:
            self.scale_factor = 1.0

        self.pred_real = pred_real

    def _update_gzz_display(self):
        """更新 Gzz 显示"""
        if self.data_mgr.gzz_32 is None:
            return

        ax = self.axes_2d[0, 0]
        Visualizer.plot_gzz_2d(
            ax, self.data_mgr.gzz_32,
            self.data_mgr.x_range, self.data_mgr.y_range,
            "Input Gzz (Eötvös)"
        )
        self.canvas_2d.draw()

    def _on_refresh(self):
        """刷新视图（包含处理）"""
        if self.pred_real is None:
            return

        self._update_display_only()

    def _update_display_only(self):
        """仅更新显示（不重新处理预测结果）"""
        if self.pred_real is None:
            return

        # 获取切片位置（米）和可见性
        x_m, y_m, z_m, show_x, show_y, show_z = self._get_slice_indices()

        try:
            vmin = float(self.ed_rho_min.text())
            vmax = float(self.ed_rho_max.text())
        except:
            vmin, vmax = -150, 350

        x_range = self.data_mgr.x_range
        y_range = self.data_mgr.y_range
        z_extent = self.pred_real.shape[0] * self.dz

        # 2D 视图
        self._update_2d_view(x_m, y_m, z_m, vmin, vmax, x_range, y_range, z_extent)

        # 真实2D切片视图 (新增 - VTI真实密度)
        self._update_true_2d_slice_view(x_m, y_m, z_m, vmin, vmax, x_range, y_range)

        # Gzz 残差视图 (新增)
        self._update_gzz_residual_view(x_range, y_range)

        # 3D 切片视图 (传递可见性)
        self._update_3d_view(x_m, y_m, z_m, show_x, show_y, show_z, vmin, vmax, x_range, y_range, z_extent)

        # 3D 真实切片视图 (新增)
        self._update_true_3d_view(x_m, y_m, z_m, show_x, show_y, show_z, vmin, vmax, x_range, y_range, z_extent)

        # 3D 体素视图
        self._update_voxel_view()

        # 3D 等值面视图
        self._update_isosurface_view()

    def _get_noisy_gzz(self) -> np.ndarray:
        """获取 Gzz 数据，如果启用噪声则添加高斯噪声"""
        if self.data_mgr.gzz_32 is None:
            return None
        
        gzz = self.data_mgr.gzz_32.copy()
        
        # 检查是否启用噪声
        if hasattr(self, 'chk_add_noise') and self.chk_add_noise.isChecked():
            try:
                noise_percent = float(self.ed_noise_level.text())
            except:
                noise_percent = 3.0
            
            # 计算噪声标准差：噪声百分比 × Gzz的最大绝对值
            noise_level = noise_percent / 100.0
            gzz_max = np.max(np.abs(gzz)) + 1e-8
            noise_std = noise_level * gzz_max
            
            # 添加高斯噪声
            noise = np.random.normal(0, noise_std, gzz.shape)
            gzz = gzz + noise.astype(np.float32)
            
            print(f"[Gzz Noise] Added {noise_percent:.1f}% Gaussian noise (std={noise_std:.3f})")
        
        return gzz

    def _update_2d_view(self, x_m, y_m, z_m, vmin, vmax, x_range, y_range, z_extent):
        """更新 2D 视图 - 使用精确米坐标插值"""
        from scipy.interpolate import RegularGridInterpolator

        nz, ny, nx = self.pred_real.shape

        # 创建网格坐标
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(0, z_extent, nz)

        # 创建插值器
        interp = RegularGridInterpolator((z, y, x), self.pred_real, method='linear', bounds_error=False, fill_value=None)

        # (0,0) 如果有VTI真实密度，显示真实密度切片；否则显示Gzz
        ax00 = self.axes_2d[0, 0]
        Visualizer.clear_colorbar(ax00)
        ax00.clear()

        if hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None:
            # VTI模式：显示真实密度的Z切片（使用插值）
            true_density = self.data_mgr.density_3d
            # 可能需要调整尺寸
            if true_density.shape != self.pred_real.shape:
                from scipy.ndimage import zoom as nd_zoom
                scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                true_density_resized = nd_zoom(true_density, scale_factors, order=1)
            else:
                true_density_resized = true_density

            # 获取真实密度的范围
            t_min, t_max = true_density_resized.min(), true_density_resized.max()

            # 归一化到 [0, 1]，确保与高分辨率保存一致（最小值蓝色，最大值红色）
            if t_max - t_min > 1e-8:
                true_normalized = (true_density_resized - t_min) / (t_max - t_min)
            else:
                true_normalized = np.zeros_like(true_density_resized) + 0.5

            # 手动反转颜色
            if self.chk_reverse_color.isChecked():
                true_normalized = 1.0 - true_normalized

            # 使用归一化数据，vmin=0, vmax=1
            display_vmin, display_vmax = 0.0, 1.0

            # 创建真实密度插值器（使用归一化数据）
            interp_true = RegularGridInterpolator((z, y, x), true_normalized, method='linear', bounds_error=False, fill_value=None)
            XX, YY = np.meshgrid(x, y)
            ZZ = np.full_like(XX, z_m)
            points = np.stack([ZZ.flatten(), YY.flatten(), XX.flatten()], axis=-1)
            slice_data = interp_true(points).reshape(ny, nx)

            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            im = ax00.imshow(slice_data, cmap='jet', aspect='auto', extent=extent, vmin=display_vmin, vmax=display_vmax, origin='lower')
            ax00.set_xlabel('X (m)')
            ax00.set_ylabel('Y (m)')
            cbar = plt.colorbar(im, ax=ax00)
            cbar.ax.tick_params(direction='in')
            # 显示实际密度范围
            cbar.ax.set_title(f'{t_min:.2f}~{t_max:.2f}', fontsize=8)
            ax00._colorbar = cbar
        else:
            # 普通模式：显示Gzz
            # 获取 Gzz 数据（可能添加噪声）
            gzz_display = self._get_noisy_gzz()
            Visualizer.plot_gzz_2d(
                ax00, gzz_display,
                self.data_mgr.x_range, self.data_mgr.y_range,
                "Input Gzz (Eötvös)"
            )

        # (0,1) 预测 Z 切片 - 精确插值
        ax01 = self.axes_2d[0, 1]
        Visualizer.clear_colorbar(ax01)
        ax01.clear()
        XX, YY = np.meshgrid(x, y)
        ZZ = np.full_like(XX, z_m)
        points = np.stack([ZZ.flatten(), YY.flatten(), XX.flatten()], axis=-1)
        slice_z = interp(points).reshape(ny, nx)
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        im = ax01.imshow(slice_z, cmap='jet', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax01.set_xlabel('X (m)')
        ax01.set_ylabel('Y (m)')
        cbar = plt.colorbar(im, ax=ax01)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)
        ax01._colorbar = cbar

        # (1,0) Y 切片 - 精确插值
        ax10 = self.axes_2d[1, 0]
        Visualizer.clear_colorbar(ax10)
        ax10.clear()
        XX_y, ZZ_y = np.meshgrid(x, z)
        YY_y = np.full_like(XX_y, y_m)
        points = np.stack([ZZ_y.flatten(), YY_y.flatten(), XX_y.flatten()], axis=-1)
        slice_y = interp(points).reshape(nz, nx)
        extent = [x_range[0], x_range[1], 0, z_extent]
        im = ax10.imshow(slice_y, cmap='jet', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax10.set_xlabel('X (m)')
        ax10.set_ylabel('Depth (m)')
        cbar = plt.colorbar(im, ax=ax10)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)
        ax10._colorbar = cbar

        # (1,1) X 切片 - 精确插值
        ax11 = self.axes_2d[1, 1]
        Visualizer.clear_colorbar(ax11)
        ax11.clear()
        YY_x, ZZ_x = np.meshgrid(y, z)
        XX_x = np.full_like(YY_x, x_m)
        points = np.stack([ZZ_x.flatten(), YY_x.flatten(), XX_x.flatten()], axis=-1)
        slice_x = interp(points).reshape(nz, ny)
        extent = [y_range[0], y_range[1], 0, z_extent]
        im = ax11.imshow(slice_x, cmap='jet', aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax11.set_xlabel('Y (m)')
        ax11.set_ylabel('Depth (m)')
        cbar = plt.colorbar(im, ax=ax11)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)
        ax11._colorbar = cbar

        self.fig_2d.tight_layout()
        self.canvas_2d.draw()

    def _update_true_2d_slice_view(self, x_m, y_m, z_m, vmin, vmax, x_range, y_range):
        """更新真实2D切片视图 - 显示VTI加载的真实密度"""
        if not hasattr(self.data_mgr, 'density_3d') or self.data_mgr.density_3d is None:
            return
            
        from scipy.interpolate import RegularGridInterpolator
        
        true_density = self.data_mgr.density_3d
        nz, ny, nx = true_density.shape
        z_extent = nz * self.dz
        
        # 创建网格坐标
        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        z = np.linspace(0, z_extent, nz)
        
        # 将真实密度的背景值(0)替换为中间值，使背景显示为绿色（与预测模型一致）
        background_value = (vmin + vmax) / 2
        true_density_display = true_density.copy()
        true_density_display[true_density_display == 0] = background_value
        
        # 创建插值器
        interp = RegularGridInterpolator((z, y, x), true_density_display, method='linear', bounds_error=False, fill_value=None)
        
        # 检查颜色反转
        cmap = 'jet_r' if self.chk_reverse_color.isChecked() else 'jet'
        
        # (0,0) Gzz 或密度信息
        ax00 = self.axes_true_2d[0, 0]
        Visualizer.clear_colorbar(ax00)
        ax00.clear()
        if self.data_mgr.gzz_32 is not None:
            Visualizer.plot_gzz_2d(ax00, self.data_mgr.gzz_32, x_range, y_range, "Input Gzz (Eötvös)")
        else:
            ax00.text(0.5, 0.5, "True 2D Slices\n(VTI Data)", transform=ax00.transAxes,
                      ha='center', va='center', fontsize=14)
            ax00.set_title("")
        
        # (0,1) Z 切片 (水平面)
        ax01 = self.axes_true_2d[0, 1]
        Visualizer.clear_colorbar(ax01)
        ax01.clear()
        XX_z, YY_z = np.meshgrid(x, y)
        ZZ_z = np.full_like(XX_z, z_m)
        points = np.stack([ZZ_z.flatten(), YY_z.flatten(), XX_z.flatten()], axis=-1)
        slice_z = interp(points).reshape(ny, nx)
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        im = ax01.imshow(slice_z, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax01.set_xlabel('X (m)')
        ax01.set_ylabel('Y (m)')
        cbar = plt.colorbar(im, ax=ax01)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)
        ax01._colorbar = cbar
        
        # (1,0) Y 切片
        ax10 = self.axes_true_2d[1, 0]
        Visualizer.clear_colorbar(ax10)
        ax10.clear()
        XX_y, ZZ_y = np.meshgrid(x, z)
        YY_y = np.full_like(XX_y, y_m)
        points = np.stack([ZZ_y.flatten(), YY_y.flatten(), XX_y.flatten()], axis=-1)
        slice_y = interp(points).reshape(nz, nx)
        extent = [x_range[0], x_range[1], 0, z_extent]
        im = ax10.imshow(slice_y, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax10.set_xlabel('X (m)')
        ax10.set_ylabel('Depth (m)')
        cbar = plt.colorbar(im, ax=ax10)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)
        ax10._colorbar = cbar
        
        # (1,1) X 切片
        ax11 = self.axes_true_2d[1, 1]
        Visualizer.clear_colorbar(ax11)
        ax11.clear()
        YY_x, ZZ_x = np.meshgrid(y, z)
        XX_x = np.full_like(YY_x, x_m)
        points = np.stack([ZZ_x.flatten(), YY_x.flatten(), XX_x.flatten()], axis=-1)
        slice_x = interp(points).reshape(nz, ny)
        extent = [y_range[0], y_range[1], 0, z_extent]
        im = ax11.imshow(slice_x, cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax, origin='lower')
        ax11.set_xlabel('Y (m)')
        ax11.set_ylabel('Depth (m)')
        cbar = plt.colorbar(im, ax=ax11)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)
        ax11._colorbar = cbar
        
        self.fig_true_2d.tight_layout()
        self.canvas_true_2d.draw()

    def _update_gzz_residual_view(self, x_range, y_range):
        """更新 Gzz 残差视图 - 显示观测Gzz、正演Gzz和残差"""
        if self.pred_real is None or self.data_mgr.gzz_32 is None:
            return

        self.fig_gzz_residual.clear()

        # 观测的 Gzz（可能添加噪声）
        gzz_obs = self._get_noisy_gzz()

        # 计算背景密度（密度范围的中点）
        try:
            rho_min = float(self.ed_rho_min.text())
            rho_max = float(self.ed_rho_max.text())
        except:
            rho_min, rho_max = -150, 350
        background = (rho_min + rho_max) / 2

        # 使用异常密度（减去背景值）进行正演
        density_anomaly = self.pred_real - background

        # 尝试使用精确的 FastGravityForward
        try:
            from data_preparation import FastGravityForward
            nz, ny, nx = density_anomaly.shape
            fwd = FastGravityForward((nz, ny, nx), dx=self.data_mgr.dx, dz=self.dz, mode='gzz')
            # FastGravityForward 需要归一化的密度输入
            density_max = np.abs(density_anomaly).max() + 1e-8
            density_norm = density_anomaly / density_max
            gzz_forward_raw = fwd.forward(density_norm)
            # 恢复量级（假设模型训练时的密度范围）
            gzz_forward_raw = gzz_forward_raw * (density_max / 300.0) * 50.0  # 经验缩放
            print(f"[Gzz Residual] Using FastGravityForward (mode=gzz)")
        except Exception as e:
            print(f"[Gzz Residual] FastGravityForward failed: {e}, using simplified forward")
            gzz_forward_raw = DensityProcessor.forward_gzz(
                density_anomaly,
                dx=self.data_mgr.dx,
                dz=self.dz
            )

        # 调整正演结果尺寸与观测一致
        if gzz_forward_raw.shape != gzz_obs.shape:
            from scipy.ndimage import zoom as nd_zoom
            scale_factors = [gzz_obs.shape[i] / gzz_forward_raw.shape[i] for i in range(2)]
            gzz_forward_raw = nd_zoom(gzz_forward_raw, scale_factors, order=1)

        # 检查相关性，如果为负则反转正演结果的符号
        corr_check = np.corrcoef(gzz_obs.flatten(), gzz_forward_raw.flatten())[0, 1]
        if corr_check < 0:
            gzz_forward_raw = -gzz_forward_raw
            print(f"[Gzz Residual] Sign corrected (original corr={corr_check:.3f})")

        # 使用最小二乘法将正演结果缩放到与观测相同的幅值
        # alpha = sum(obs * fwd) / sum(fwd * fwd)
        obs_flat = gzz_obs.flatten()
        fwd_flat = gzz_forward_raw.flatten()
        denominator = np.sum(fwd_flat * fwd_flat)
        if denominator > 1e-10:
            scale_alpha = np.sum(obs_flat * fwd_flat) / denominator
        else:
            scale_alpha = 1.0
        gzz_forward_scaled = gzz_forward_raw * scale_alpha

        # 计算实际残差（带单位）
        residual_real = gzz_obs - gzz_forward_scaled
        rmse_real = np.sqrt(np.mean(residual_real**2))

        # 归一化到 [0, 1] 范围进行形态对比
        obs_min, obs_max = gzz_obs.min(), gzz_obs.max()
        fwd_min, fwd_max = gzz_forward_scaled.min(), gzz_forward_scaled.max()

        obs_range = obs_max - obs_min if obs_max - obs_min > 1e-8 else 1.0
        fwd_range = fwd_max - fwd_min if fwd_max - fwd_min > 1e-8 else 1.0

        gzz_obs_norm = (gzz_obs - obs_min) / obs_range
        gzz_fwd_norm = (gzz_forward_scaled - fwd_min) / fwd_range

        # 归一化残差
        residual_norm = gzz_obs_norm - gzz_fwd_norm

        # 计算相关系数（校正后）
        corr = np.corrcoef(gzz_obs.flatten(), gzz_forward_scaled.flatten())[0, 1]
        rmse_norm = np.sqrt(np.mean(residual_norm**2))

        # Y轴从下往上
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

        # 创建 2x2 子图
        # 使用统一的颜色范围（观测Gzz的范围）
        obs_vmax = max(abs(obs_min), abs(obs_max))
        common_vmin = -obs_vmax if obs_min < 0 else 0
        common_vmax = obs_vmax

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy import stats
        from scipy.ndimage import zoom as nd_zoom

        # 对数据进行插值上采样使显示更平滑
        interp_factor = 4  # 插值放大倍数
        gzz_obs_interp = nd_zoom(gzz_obs, interp_factor, order=3)  # 三次样条插值
        gzz_fwd_interp = nd_zoom(gzz_forward_scaled, interp_factor, order=3)
        residual_interp = nd_zoom(residual_real, interp_factor, order=3)

        # 手动反转所有 Gzz 数据符号（用于颜色反转）
        if self.chk_reverse_color.isChecked():
            gzz_obs_interp = -gzz_obs_interp
            gzz_fwd_interp = -gzz_fwd_interp
            residual_interp = -residual_interp

        # 重新计算显示范围（因为符号可能变了）
        current_max = max(abs(gzz_obs_interp.min()), abs(gzz_obs_interp.max()))
        # 如果数据主要是正的（或者反转后变成了正的），vmin设为0或微负；如果是双极性的，设为对称
        # 为简单起见，保持与之前类似的逻辑：如果原本包含负值，就用对称range；否则0~max
        # 但既然是 Gzz，通常是对称分布或偏负/偏正。使用 min~max 也许最准确，但为了美观我们常用对称或 0起点的 jet

        # 这里使用 min/max 直接映射，确保颜色利用率
        common_vmin = gzz_obs_interp.min()
        common_vmax = gzz_obs_interp.max()
        # 稍微扩展一点
        if common_vmax - common_vmin < 1e-6:
            common_vmin -= 0.1
            common_vmax += 0.1

        # 图1: 观测 Gzz
        ax1 = self.fig_gzz_residual.add_subplot(2, 2, 1)
        im1 = ax1.imshow(gzz_obs_interp, cmap='jet', aspect='equal', extent=extent,
                         vmin=common_vmin, vmax=common_vmax, origin='lower')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = self.fig_gzz_residual.colorbar(im1, cax=cax1)
        cbar1.ax.tick_params(direction='in')
        cbar1.ax.set_title('E', fontsize=9)

        # 图2: 正演 Gzz（缩放后，使用与观测相同的颜色范围）
        ax2 = self.fig_gzz_residual.add_subplot(2, 2, 2)
        im2 = ax2.imshow(gzz_fwd_interp, cmap='jet', aspect='equal', extent=extent,
                         vmin=common_vmin, vmax=common_vmax, origin='lower')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = self.fig_gzz_residual.colorbar(im2, cax=cax2)
        cbar2.ax.tick_params(direction='in')
        cbar2.ax.set_title('E', fontsize=9)

        # 图3: 实际残差（带单位）
        ax3 = self.fig_gzz_residual.add_subplot(2, 2, 3)
        res_max = max(abs(residual_real.min()), abs(residual_real.max()), 1e-8)
        im3 = ax3.imshow(residual_interp, cmap='RdBu_r', aspect='equal', extent=extent,
                         vmin=-res_max, vmax=res_max, origin='lower')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')

        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = self.fig_gzz_residual.colorbar(im3, cax=cax3)
        cbar3.ax.tick_params(direction='in')
        cbar3.ax.set_title('E', fontsize=9)

        # 图4: 残差直方图（带正态分布拟合线）
        ax4 = self.fig_gzz_residual.add_subplot(2, 2, 4)
        residual_flat = residual_real.flatten()

        # 绘制直方图
        n, bins, patches = ax4.hist(residual_flat, bins=30, density=True, alpha=0.7,
                                     color='steelblue', edgecolor='white', label='Residual')

        # 正态分布拟合
        mu, std = stats.norm.fit(residual_flat)
        x_fit = np.linspace(bins[0], bins[-1], 100)
        y_fit = stats.norm.pdf(x_fit, mu, std)
        ax4.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')

        # 添加零线
        ax4.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        ax4.set_xlabel('Residual (E)')
        ax4.set_ylabel('Probability Density')

        ax4.legend(loc='upper right', fontsize=8, frameon=False)
        ax4.grid(True, alpha=0.3)

        self.fig_gzz_residual.tight_layout()
        self.canvas_gzz_residual.draw()

    def _update_3d_view(self, x_m, y_m, z_m, show_x, show_y, show_z, vmin, vmax, x_range, y_range, z_extent):
        """更新 3D 切片视图 - 使用精确米坐标"""
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')

        Visualizer.plot_3d_slices(
            ax, self.pred_real, x_range, y_range, z_extent,
            x_m, y_m, z_m, vmin, vmax,
            show_x=show_x, show_y=show_y, show_z=show_z,
            title="3D Orthogonal Slices"
        )

        self.fig_3d.tight_layout()
        self.canvas_3d.draw()

    def _update_true_3d_view(self, x_m, y_m, z_m, show_x, show_y, show_z, vmin, vmax, x_range, y_range, z_extent):
        """更新 3D 真实切片视图"""
        self.fig_true_3d.clear()

        # 检查是否有真实密度数据
        if not hasattr(self.data_mgr, 'density_3d') or self.data_mgr.density_3d is None:
            ax = self.fig_true_3d.add_subplot(111)
            ax.text(0.5, 0.5, "No True Density Data Available\n(Load .vti file to verify)",
                    ha='center', va='center', fontsize=14)
            self.canvas_true_3d.draw()
            return

        true_density = self.data_mgr.density_3d

        # 调整尺寸以匹配预测结果（如果需要）
        if self.pred_real is not None and true_density.shape != self.pred_real.shape:
            from scipy.ndimage import zoom as nd_zoom
            scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
            true_density = nd_zoom(true_density, scale_factors, order=1)
        
        # 将真实密度的背景值(0)替换为中间值，使背景显示为绿色（与预测模型一致）
        background_value = (vmin + vmax) / 2
        true_density = true_density.copy()
        true_density[true_density == 0] = background_value

        ax = self.fig_true_3d.add_subplot(111, projection='3d')

        # 不进行归一化，直接使用用户指定的vmin/vmax范围
        # 这样背景值(中间值)会显示为绿色
        
        # 手动反转颜色时交换vmin和vmax
        if self.chk_reverse_color.isChecked():
            display_vmin, display_vmax = vmax, vmin
        else:
            display_vmin, display_vmax = vmin, vmax

        title_str = f"True 3D Slices (Range: {self.data_mgr.density_3d.min():.2f} ~ {self.data_mgr.density_3d.max():.2f})"

        Visualizer.plot_3d_slices(
            ax, true_density, x_range, y_range, z_extent,
            x_m, y_m, z_m, display_vmin, display_vmax,
            show_x=show_x, show_y=show_y, show_z=show_z,
            title=title_str
        )

        # 添加 Colorbar
        from matplotlib import cm
        norm = plt.Normalize(vmin=display_vmin, vmax=display_vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap='jet')
        mappable.set_array([display_vmin, display_vmax])
        cbar = self.fig_true_3d.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(direction='in')
        cbar.ax.set_title('kg/m³', fontsize=9)

        self.fig_true_3d.tight_layout()
        self.canvas_true_3d.draw()

    def _update_voxel_view(self):
        """更新 3D 体素视图 - 使用 PyVista 专业渲染"""
        has_true_density = hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None
        
        # 如果既没有真实密度也没有预测密度，直接返回
        if self.pred_real is None and not has_true_density:
            return

        try:
            vmin = float(self.ed_rho_min.text())
            vmax = float(self.ed_rho_max.text())
        except:
            vmin, vmax = -150, 350

        threshold = self.sl_thresh.value() / 100.0

        # 强制使用 Matplotlib 渲染（PyVista有数据顺序问题）
        self._update_voxel_matplotlib(vmin, vmax, threshold)

    def _update_voxel_pyvista(self, vmin, vmax, threshold):
        """使用 PyVista 渲染体素 (与 gui.py 一致)"""
        self.plotter.clear()

        x_range = self.data_mgr.x_range
        y_range = self.data_mgr.y_range
        
        has_true_density = hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None

        if has_true_density:
            from scipy.ndimage import zoom as nd_zoom
            true_density = self.data_mgr.density_3d
            z_extent = true_density.shape[0] * self.dz
            
            if self.pred_real is not None:
                # 有预测结果：显示双模型对比
                if true_density.shape != self.pred_real.shape:
                    scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                    true_density_resized = nd_zoom(true_density, scale_factors, order=1)
                else:
                    true_density_resized = true_density.copy()
                z_extent = self.pred_real.shape[0] * self.dz

                # 真实模型使用自己的数据范围
                true_vmin = true_density_resized.min()
                true_vmax = true_density_resized.max()
                if true_vmax - true_vmin < 1e-6:
                    true_vmax = true_vmin + 1.0  # 避免范围太小
                
                # 检查是否强制最大值
                if self.chk_force_true_max.isChecked():
                    # 将高于中值的区域强制设置为一个超过vmax的值，确保显示为红色
                    true_center = (true_vmin + true_vmax) / 2
                    true_density_forced = np.zeros_like(true_density_resized)
                    # 高值区域设置为vmax（会显示为红色）
                    mask = true_density_resized > true_center
                    true_density_forced[mask] = vmax
                    print(f"[Voxel] Force True Max: {mask.sum()} points set to {vmax}")
                    # 使用阈值0确保所有设置的点都显示
                    self._add_pyvista_volume(true_density_forced, vmin, vmax, 0.01, "True Model",
                                              x_range, y_range, z_extent, offset_x=0)
                else:
                    print(f"[Voxel] True density clim: [{true_vmin:.3f}, {true_vmax:.3f}]")
                    # 渲染真实模型（左边）- 使用真实数据的范围
                    self._add_pyvista_volume(true_density_resized, true_vmin, true_vmax, threshold, "True Model",
                                              x_range, y_range, z_extent, offset_x=0)
                # 渲染预测模型（右边，偏移）- 使用用户设置的范围
                x_offset = x_range[1] - x_range[0] + 500
                self._add_pyvista_volume(self.pred_real, vmin, vmax, threshold, "Pred Model",
                                          x_range, y_range, z_extent, offset_x=x_offset)
            else:
                # 只有真实密度（VTI加载后未运行反演）：单模型显示
                self._add_pyvista_volume(true_density, vmin, vmax, threshold, "VTI Model",
                                          x_range, y_range, z_extent)
        else:
            # 只有预测结果
            z_extent = self.pred_real.shape[0] * self.dz
            self._add_pyvista_volume(self.pred_real, vmin, vmax, threshold, "Prediction",
                                      x_range, y_range, z_extent)

        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.camera.zoom(0.85)

    def _add_pyvista_volume(self, vol, vmin, vmax, thresh_pct, title, x_range, y_range, z_extent, offset_x=0):
        """添加一个 PyVista 体积到场景 (使用真实坐标)"""
        nz, ny, nx = vol.shape

        # 计算真实间距
        dx = (x_range[1] - x_range[0]) / (nx - 1) if nx > 1 else 100
        dy = (y_range[1] - y_range[0]) / (ny - 1) if ny > 1 else 100
        dz_spacing = z_extent / (nz - 1) if nz > 1 else 100

        # 创建 ImageData 网格
        grid = pv.ImageData()
        grid.dimensions = [nx, ny, nz]
        grid.spacing = (dx, dy, dz_spacing)
        grid.origin = (x_range[0] + offset_x, y_range[0], 0)
        
        # VTK ImageData 点顺序：x变化最快，z变化最慢
        # NumPy 数组 (nz, ny, nx) 需要转置为 (nx, ny, nz)，然后用Fortran顺序展平
        # 或者直接用ravel('F')按Fortran顺序展平
        vol_vtk = np.asfortranarray(vol.transpose(2, 1, 0)).ravel()
        grid.point_data["v"] = vol_vtk

        # 不反转Z轴，使用正值深度

        # 阈值过滤 - 基于 vmin/vmax 范围计算（与等值面一致）
        value_range = vmax - vmin
        if value_range < 1e-6:
            value_range = 1.0  # 避免除以零
        
        # 阈值：显示密度 > vmin + thresh_pct * range 的区域
        thresh = vmin + thresh_pct * value_range
        
        # 如果数据最大值低于阈值，使用数据自身范围的50%作为阈值（fallback）
        data_max = vol.max()
        data_min = vol.min()
        if data_max < thresh:
            fallback_thresh = data_min + 0.5 * (data_max - data_min)
            print(f"  Fallback: data_max={data_max:.3f} < thresh={thresh:.3f}, using {fallback_thresh:.3f}")
            thresh = fallback_thresh
        
        # 调试输出
        print(f"[PyVista Voxel] {title}: shape={vol.shape}, range=[{data_min:.3f}, {data_max:.3f}]")
        print(f"  vmin={vmin:.2f}, vmax={vmax:.2f}, thresh={thresh:.2f}, thresh_pct={thresh_pct:.0%}")

        try:
            # 过滤出密度高于阈值的部分
            thresh_grid = grid.threshold(value=thresh, scalars="v")
            print(f"  After threshold: {thresh_grid.n_points} points")
            if thresh_grid.n_points > 0:
                # 调试：显示threshold后的实际值范围
                thresh_values = thresh_grid.point_data["v"]
                print(f"  Thresholded data range: [{thresh_values.min():.3f}, {thresh_values.max():.3f}]")
                
                # 检查颜色反转
                cmap = "jet_r" if self.chk_reverse_color.isChecked() else "jet"

                # 使用用户设置的vmin/vmax范围（与等值面一致）
                self.plotter.add_mesh(
                    thresh_grid, scalars="v", cmap=cmap,
                    clim=[vmin, vmax], opacity=1.0, show_scalar_bar=True,
                    scalar_bar_args={
                        'title': 'kg/m³',
                        'vertical': True,
                        'position_x': 0.9,
                        'position_y': 0.2,
                        'height': 0.6,
                        'width': 0.05
                    }
                )
                
                # 添加体素边缘线（白色网格效果）
                self.plotter.add_mesh(
                    thresh_grid.extract_all_edges(), 
                    color='white', line_width=0.5, opacity=0.8
                )
            else:
                print(f"  WARNING: No voxels above threshold! Data range [{vol.min():.3f}, {vol.max():.3f}] vs thresh={thresh:.3f}")
        except Exception as e:
            print(f"  Exception in threshold: {e}")

        # 添加外框线
        self.plotter.add_mesh(grid.outline(), color='black', line_width=1.5)

        # 添加坐标轴标签和密网格
        self.plotter.show_bounds(
            grid='back', location='outer', ticks='both',
            xlabel='X (m)', ylabel='Y (m)', zlabel='Depth (m)',
            color='black', font_size=8,
            n_xlabels=6, n_ylabels=6, n_zlabels=6,  # 增加刻度数量
            all_edges=True  # 显示所有边缘网格线
        )
        
        # 添加XYZ三个背景平面的密网格
        x_min, x_max = x_range[0] + offset_x, x_range[1] + offset_x
        y_min, y_max = y_range
        z_min, z_max = 0, z_extent
        grid_resolution = 15  # 网格密度
        grid_color = 'gray'
        grid_line_width = 0.3
        grid_opacity = 0.4
        
        # 底部网格 (z=0 平面, XY平面)
        floor_grid = pv.Plane(
            center=((x_min + x_max) / 2, (y_min + y_max) / 2, z_min),
            direction=(0, 0, 1),
            i_size=x_max - x_min,
            j_size=y_max - y_min,
            i_resolution=grid_resolution,
            j_resolution=grid_resolution
        )
        self.plotter.add_mesh(floor_grid.extract_all_edges(), color=grid_color, line_width=grid_line_width, opacity=grid_opacity)
        
        # 后墙网格 (y=y_min 平面, XZ平面)
        back_grid = pv.Plane(
            center=((x_min + x_max) / 2, y_min, (z_min + z_max) / 2),
            direction=(0, 1, 0),
            i_size=x_max - x_min,
            j_size=z_max - z_min,
            i_resolution=grid_resolution,
            j_resolution=grid_resolution
        )
        self.plotter.add_mesh(back_grid.extract_all_edges(), color=grid_color, line_width=grid_line_width, opacity=grid_opacity)
        
        # 左墙网格 (x=x_min 平面, YZ平面)
        left_grid = pv.Plane(
            center=(x_min, (y_min + y_max) / 2, (z_min + z_max) / 2),
            direction=(1, 0, 0),
            i_size=y_max - y_min,
            j_size=z_max - z_min,
            i_resolution=grid_resolution,
            j_resolution=grid_resolution
        )
        self.plotter.add_mesh(left_grid.extract_all_edges(), color=grid_color, line_width=grid_line_width, opacity=grid_opacity)

    def _update_voxel_matplotlib(self, vmin, vmax, threshold):
        """回退：使用 matplotlib 渲染体素"""
        # 强制坐标从0开始
        x_extent = self.data_mgr.x_range[1] - self.data_mgr.x_range[0]
        y_extent = self.data_mgr.y_range[1] - self.data_mgr.y_range[0]
        x_range = (0, x_extent)
        y_range = (0, y_extent)
        
        # 计算 z_extent：优先使用 pred_real，否则使用 density_3d
        if self.pred_real is not None:
            z_extent = self.pred_real.shape[0] * self.dz
        elif hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None:
            z_extent = self.data_mgr.density_3d.shape[0] * self.dz
        else:
            return  # 没有可用数据

        reverse_color = self.chk_reverse_color.isChecked()
        show_negative = self.chk_show_negative.isChecked()

        self.fig_voxel.clear()

        has_true_density = hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None

        if has_true_density:
            from scipy.ndimage import zoom as nd_zoom
            true_density = self.data_mgr.density_3d
            
            if self.pred_real is not None:
                # 有预测结果：调整真实密度尺寸并显示对比
                if true_density.shape != self.pred_real.shape:
                    scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                    true_density_resized = nd_zoom(true_density, scale_factors, order=1)
                else:
                    true_density_resized = true_density.copy()

                # 检查是否强制最大值
                if self.chk_force_true_max.isChecked():
                    true_center = (true_density_resized.min() + true_density_resized.max()) / 2
                    # 使用负值确保低值区域不被显示（threshold会过滤掉负值）
                    true_density_forced = np.full_like(true_density_resized, -9999.0)
                    mask = true_density_resized > true_center
                    true_density_forced[mask] = vmax
                    print(f"[Matplotlib Voxel] Force True Max: {mask.sum()} points set to {vmax}")
                    true_density_for_plot = true_density_forced
                else:
                    true_density_for_plot = true_density_resized

                ax1 = self.fig_voxel.add_subplot(121, projection='3d')
                Visualizer.plot_3d_voxels(
                    ax1, true_density_for_plot, x_range, y_range, z_extent,
                    vmin, vmax, 0.1, "True Model",
                    reverse_color=reverse_color, show_negative=show_negative
                )

                ax2 = self.fig_voxel.add_subplot(122, projection='3d')
                Visualizer.plot_3d_voxels(
                    ax2, self.pred_real, x_range, y_range, z_extent,
                    vmin, vmax, threshold, "Pred Model",
                    reverse_color=reverse_color, show_negative=show_negative
                )
            else:
                # 只有真实密度（VTI加载后未运行反演）：单模型显示
                ax = self.fig_voxel.add_subplot(111, projection='3d')
                Visualizer.plot_3d_voxels(
                    ax, true_density, x_range, y_range, z_extent,
                    vmin, vmax, threshold, "VTI Model",
                    reverse_color=reverse_color, show_negative=show_negative
                )
        else:
            ax = self.fig_voxel.add_subplot(111, projection='3d')
            Visualizer.plot_3d_voxels(
                ax, self.pred_real, x_range, y_range, z_extent,
                vmin, vmax, threshold, "3D Voxel View",
                reverse_color=reverse_color, show_negative=show_negative
            )

        self.fig_voxel.tight_layout()
        self.canvas_voxel.draw()

    def _update_isosurface_view(self):
        """更新 3D 等值面视图 - 更平滑专业的可视化"""
        has_true_density = hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None
        
        # 如果既没有真实密度也没有预测密度，直接返回
        if self.pred_real is None and not has_true_density:
            return

        try:
            vmin = float(self.ed_rho_min.text())
            vmax = float(self.ed_rho_max.text())
        except:
            vmin, vmax = -150, 350

        threshold = self.sl_thresh.value() / 100.0

        x_range = self.data_mgr.x_range
        y_range = self.data_mgr.y_range
        show_negative = self.chk_show_negative.isChecked()

        self.fig_isosurface.clear()

        if has_true_density:
            from scipy.ndimage import zoom as nd_zoom
            true_density = self.data_mgr.density_3d
            
            # 计算 z_extent：优先使用 pred_real 的形状，否则使用 true_density
            if self.pred_real is not None:
                z_extent = self.pred_real.shape[0] * self.dz
            else:
                z_extent = true_density.shape[0] * self.dz

            if self.pred_real is not None:
                # 有预测结果：显示双图对比
                if true_density.shape != self.pred_real.shape:
                    scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                    true_density_resized = nd_zoom(true_density, scale_factors, order=1)
                else:
                    true_density_resized = true_density.copy()

                # 左图：真实模型等值面
                ax1 = self.fig_isosurface.add_subplot(121, projection='3d')
                Visualizer.plot_3d_isosurface(
                    ax1, true_density_resized, x_range, y_range, z_extent,
                    vmin, vmax, threshold, f"True Model: Isosurface (>{int(threshold*100)}%)",
                    show_negative=show_negative
                )

                # 右图：预测模型等值面
                ax2 = self.fig_isosurface.add_subplot(122, projection='3d')
                Visualizer.plot_3d_isosurface(
                    ax2, self.pred_real, x_range, y_range, z_extent,
                    vmin, vmax, threshold, f"Pred Model: Isosurface (>{int(threshold*100)}%)",
                    show_negative=show_negative
                )
            else:
                # 只有真实密度（VTI加载后未运行反演）：单图显示
                ax = self.fig_isosurface.add_subplot(111, projection='3d')
                Visualizer.plot_3d_isosurface(
                    ax, true_density, x_range, y_range, z_extent,
                    vmin, vmax, threshold, f"VTI Model: Isosurface (>{int(threshold*100)}%)",
                    show_negative=show_negative
                )
        else:
            # 只有预测结果
            z_extent = self.pred_real.shape[0] * self.dz
            ax = self.fig_isosurface.add_subplot(111, projection='3d')
            Visualizer.plot_3d_isosurface(
                ax, self.pred_real, x_range, y_range, z_extent,
                vmin, vmax, threshold, "3D Isosurface View",
                show_negative=show_negative
            )

        self.fig_isosurface.tight_layout()
        self.canvas_isosurface.draw()

    def _on_save(self):
        """保存图像 - 只保存当前模块的所有子图"""
        # 选择保存目录
        from PySide6.QtWidgets import QFileDialog
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
        if not save_dir:
            return
        
        tab_idx = self.tabs.currentIndex()
        saved_files = []
        
        if tab_idx == 0:  # 2D 切片
            names_2d = ["gzz_or_true", "z_slice", "y_slice", "x_slice"]
            for i, ax in enumerate(self.axes_2d.flat):
                if ax.get_images():
                    fig_single = Figure(figsize=(6, 5))
                    fig_single.patch.set_facecolor('white')
                    ax_new = fig_single.add_subplot(111)
                    
                    for im in ax.get_images():
                        extent = im.get_extent()
                        data = im.get_array()
                        cmap = im.get_cmap()
                        vmin, vmax = im.get_clim()
                        im_new = ax_new.imshow(data, cmap=cmap, extent=extent, 
                                               vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax_new)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = fig_single.colorbar(im_new, cax=cax)
                        cbar.ax.set_title('kg/m³', fontsize=9)
                    
                    ax_new.set_xlabel(ax.get_xlabel())
                    ax_new.set_ylabel(ax.get_ylabel())
                    
                    name = names_2d[i] if i < len(names_2d) else f"2d_{i+1}"
                    path = os.path.join(save_dir, f"{name}.png")
                    fig_single.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_single)
                    saved_files.append(path)
        
        elif tab_idx == 1:  # 真实2D切片 (新增)
            names_true_2d = ["true_gzz_or_info", "true_z_slice", "true_y_slice", "true_x_slice"]
            for i, ax in enumerate(self.axes_true_2d.flat):
                if ax.get_images():
                    fig_single = Figure(figsize=(6, 5))
                    fig_single.patch.set_facecolor('white')
                    ax_new = fig_single.add_subplot(111)
                    
                    for im in ax.get_images():
                        extent = im.get_extent()
                        data = im.get_array()
                        cmap = im.get_cmap()
                        vmin, vmax = im.get_clim()
                        im_new = ax_new.imshow(data, cmap=cmap, extent=extent, 
                                               vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax_new)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = fig_single.colorbar(im_new, cax=cax)
                        cbar.ax.set_title('kg/m³', fontsize=9)
                    
                    ax_new.set_xlabel(ax.get_xlabel())
                    ax_new.set_ylabel(ax.get_ylabel())
                    
                    name = names_true_2d[i] if i < len(names_true_2d) else f"true_2d_{i+1}"
                    path = os.path.join(save_dir, f"{name}.png")
                    fig_single.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_single)
                    saved_files.append(path)
        
        elif tab_idx == 2:  # Gzz 残差
            names_gzz = ["observed_gzz", "forward_gzz", "residual", "histogram"]
            gzz_idx = 0
            for ax in self.fig_gzz_residual.axes:
                fig_single = Figure(figsize=(6, 5))
                fig_single.patch.set_facecolor('white')
                ax_new = fig_single.add_subplot(111)
                
                if ax.get_images():  # 图像类型的子图
                    for im in ax.get_images():
                        extent = im.get_extent()
                        data = im.get_array()
                        cmap = im.get_cmap()
                        vmin, vmax = im.get_clim()
                        im_new = ax_new.imshow(data, cmap=cmap, extent=extent, 
                                               vmin=vmin, vmax=vmax, aspect='equal', origin='lower')
                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax_new)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = fig_single.colorbar(im_new, cax=cax)
                        cbar.ax.set_title('E', fontsize=9)  # Eötvös单位
                    
                    ax_new.set_xlabel(ax.get_xlabel())
                    ax_new.set_ylabel(ax.get_ylabel())
                    
                    name = names_gzz[gzz_idx] if gzz_idx < len(names_gzz) else f"gzz_{gzz_idx+1}"
                    path = os.path.join(save_dir, f"{name}.png")
                    fig_single.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig_single)
                    saved_files.append(path)
                    gzz_idx += 1
                elif ax.patches:  # 直方图
                    # 直接从GUI复制直方图的bars和lines
                    for patch in ax.patches:
                        rect = plt.Rectangle((patch.get_x(), patch.get_y()),
                                             patch.get_width(), patch.get_height(),
                                             facecolor=patch.get_facecolor(),
                                             edgecolor=patch.get_edgecolor(),
                                             alpha=patch.get_alpha())
                        ax_new.add_patch(rect)
                    
                    # 复制lines（拟合曲线和零线）
                    for line in ax.get_lines():
                        ax_new.plot(line.get_xdata(), line.get_ydata(),
                                   color=line.get_color(),
                                   linestyle=line.get_linestyle(),
                                   linewidth=line.get_linewidth(),
                                   alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
                                   label=line.get_label())
                    
                    # 复制坐标轴设置
                    ax_new.set_xlim(ax.get_xlim())
                    ax_new.set_ylim(ax.get_ylim())
                    ax_new.set_xlabel(ax.get_xlabel())
                    ax_new.set_ylabel(ax.get_ylabel())
                    
                    # 复制图例
                    if ax.get_legend():
                        ax_new.legend(loc='upper right', fontsize=8, frameon=False)
                    
                    # 复制网格
                    ax_new.grid(True, alpha=0.3)
                    
                    path = os.path.join(save_dir, "histogram.png")
                    fig_single.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                    saved_files.append(path)
                    
                    plt.close(fig_single)
                    gzz_idx += 1
                    
        elif tab_idx == 3:  # 3D 切片 - 分别保存 X/Y/Z 三个方向的 3D 透视图
            if self.pred_real is not None:
                x_m, y_m, z_m, show_x, show_y, show_z = self._get_slice_indices()
                try:
                    vmin = float(self.ed_rho_min.text())
                    vmax = float(self.ed_rho_max.text())
                except:
                    vmin, vmax = -150, 350
                
                x_range = self.data_mgr.x_range
                y_range = self.data_mgr.y_range
                z_extent = self.pred_real.shape[0] * self.dz
                
                from matplotlib import cm
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                
                # X 切片 3D 图
                fig_x = Figure(figsize=(12, 10))
                fig_x.patch.set_facecolor('white')
                ax_x = fig_x.add_subplot(111, projection='3d')
                Visualizer.plot_3d_slices(
                    ax_x, self.pred_real, x_range, y_range, z_extent,
                    x_m, y_m, z_m, vmin, vmax,
                    show_x=True, show_y=False, show_z=False,
                    title=""
                )
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array(self.pred_real)
                cbar = fig_x.colorbar(mappable, ax=ax_x, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(direction='in')
                cbar.ax.set_title('kg/m³', fontsize=9)
                fig_x.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "3d_x_slice.png")
                fig_x.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_x)
                saved_files.append(path)
                
                # Y 切片 3D 图
                fig_y = Figure(figsize=(12, 10))
                fig_y.patch.set_facecolor('white')
                ax_y = fig_y.add_subplot(111, projection='3d')
                Visualizer.plot_3d_slices(
                    ax_y, self.pred_real, x_range, y_range, z_extent,
                    x_m, y_m, z_m, vmin, vmax,
                    show_x=False, show_y=True, show_z=False,
                    title=""
                )
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array(self.pred_real)
                cbar = fig_y.colorbar(mappable, ax=ax_y, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(direction='in')
                cbar.ax.set_title('kg/m³', fontsize=9)
                fig_y.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "3d_y_slice.png")
                fig_y.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_y)
                saved_files.append(path)
                
                # Z 切片 3D 图
                fig_z = Figure(figsize=(12, 10))
                fig_z.patch.set_facecolor('white')
                ax_z = fig_z.add_subplot(111, projection='3d')
                Visualizer.plot_3d_slices(
                    ax_z, self.pred_real, x_range, y_range, z_extent,
                    x_m, y_m, z_m, vmin, vmax,
                    show_x=False, show_y=False, show_z=True,
                    title=""
                )
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array(self.pred_real)
                cbar = fig_z.colorbar(mappable, ax=ax_z, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(direction='in')
                cbar.ax.set_title('kg/m³', fontsize=9)
                fig_z.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "3d_z_slice.png")
                fig_z.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_z)
                saved_files.append(path)
            
        elif tab_idx == 4:  # 3D 真实切片 (新增)
            if hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None:
                x_m, y_m, z_m, show_x, show_y, show_z = self._get_slice_indices()
                try:
                    vmin = float(self.ed_rho_min.text())
                    vmax = float(self.ed_rho_max.text())
                except:
                    vmin, vmax = -150, 350
                
                x_range = self.data_mgr.x_range
                y_range = self.data_mgr.y_range
                true_density = self.data_mgr.density_3d
                
                # 确保尺寸匹配
                if self.pred_real is not None and true_density.shape != self.pred_real.shape:
                    from scipy.ndimage import zoom as nd_zoom
                    scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                    true_density = nd_zoom(true_density, scale_factors, order=1)
                
                # 将真实密度的背景值(0)替换为中间值，使背景显示为绿色
                background_value = (vmin + vmax) / 2
                true_density = true_density.copy()
                true_density[true_density == 0] = background_value
                
                z_extent = true_density.shape[0] * self.dz
                
                from matplotlib import cm
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                
                # X 切片
                fig_x = Figure(figsize=(12, 10))
                fig_x.patch.set_facecolor('white')
                ax_x = fig_x.add_subplot(111, projection='3d')
                Visualizer.plot_3d_slices(
                    ax_x, true_density, x_range, y_range, z_extent,
                    x_m, y_m, z_m, vmin, vmax,
                    show_x=True, show_y=False, show_z=False,
                    title=""
                )
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array(true_density)
                cbar = fig_x.colorbar(mappable, ax=ax_x, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(direction='in')
                cbar.ax.set_title('kg/m³', fontsize=9)
                fig_x.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "true_3d_x_slice.png")
                fig_x.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_x)
                saved_files.append(path)
                
                # Y 切片
                fig_y = Figure(figsize=(12, 10))
                fig_y.patch.set_facecolor('white')
                ax_y = fig_y.add_subplot(111, projection='3d')
                Visualizer.plot_3d_slices(
                    ax_y, true_density, x_range, y_range, z_extent,
                    x_m, y_m, z_m, vmin, vmax,
                    show_x=False, show_y=True, show_z=False,
                    title=""
                )
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array(true_density)
                cbar = fig_y.colorbar(mappable, ax=ax_y, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(direction='in')
                cbar.ax.set_title('kg/m³', fontsize=9)
                fig_y.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "true_3d_y_slice.png")
                fig_y.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_y)
                saved_files.append(path)
                
                # Z 切片
                fig_z = Figure(figsize=(12, 10))
                fig_z.patch.set_facecolor('white')
                ax_z = fig_z.add_subplot(111, projection='3d')
                Visualizer.plot_3d_slices(
                    ax_z, true_density, x_range, y_range, z_extent,
                    x_m, y_m, z_m, vmin, vmax,
                    show_x=False, show_y=False, show_z=True,
                    title=""
                )
                mappable = cm.ScalarMappable(norm=norm, cmap='jet')
                mappable.set_array(true_density)
                cbar = fig_z.colorbar(mappable, ax=ax_z, shrink=0.6, pad=0.02)
                cbar.ax.tick_params(direction='in')
                cbar.ax.set_title('kg/m³', fontsize=9)
                fig_z.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "true_3d_z_slice.png")
                fig_z.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_z)
                saved_files.append(path)

        elif tab_idx == 5:  # 3D 体素 - 分别保存真实和预测模型
            try:
                vmin = float(self.ed_rho_min.text())
                vmax = float(self.ed_rho_max.text())
            except:
                vmin, vmax = -150, 350
            threshold = self.sl_thresh.value() / 100.0
            x_range = self.data_mgr.x_range
            y_range = self.data_mgr.y_range
            
            has_true_density = hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None
            
            # 使用Matplotlib分别保存真实和预测模型
            reverse_color = self.chk_reverse_color.isChecked()
            
            # 强制坐标从0开始
            x_extent = x_range[1] - x_range[0]
            y_extent = y_range[1] - y_range[0]
            x_range_0 = (0, x_extent)
            y_range_0 = (0, y_extent)
            
            if has_true_density:
                from scipy.ndimage import zoom as nd_zoom
                true_density = self.data_mgr.density_3d
                z_extent = true_density.shape[0] * self.dz
                
                if self.pred_real is not None and true_density.shape != self.pred_real.shape:
                    scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                    true_density_resized = nd_zoom(true_density, scale_factors, order=1)
                else:
                    true_density_resized = true_density.copy()
                
                # 检查是否强制最大值
                if self.chk_force_true_max.isChecked():
                    true_center = (true_density_resized.min() + true_density_resized.max()) / 2
                    true_density_forced = np.full_like(true_density_resized, -9999.0)
                    mask = true_density_resized > true_center
                    true_density_forced[mask] = vmax
                    true_density_for_plot = true_density_forced
                else:
                    true_density_for_plot = true_density_resized
                
                # 保存真实模型
                fig_true = Figure(figsize=(10, 8))
                fig_true.patch.set_facecolor('white')
                ax_true = fig_true.add_subplot(111, projection='3d')
                Visualizer.plot_3d_voxels(
                    ax_true, true_density_for_plot, x_range_0, y_range_0, z_extent,
                    vmin, vmax, 0.1, "True Model",
                    reverse_color=reverse_color
                )
                path = os.path.join(save_dir, "voxel_true.png")
                fig_true.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
                fig_true.savefig(path, dpi=300, facecolor='white', pad_inches=0.3)
                plt.close(fig_true)
                saved_files.append(path)
            
            if self.pred_real is not None:
                z_extent = self.pred_real.shape[0] * self.dz
                
                # 保存预测模型
                fig_pred = Figure(figsize=(10, 8))
                fig_pred.patch.set_facecolor('white')
                ax_pred = fig_pred.add_subplot(111, projection='3d')
                Visualizer.plot_3d_voxels(
                    ax_pred, self.pred_real, x_range_0, y_range_0, z_extent,
                    vmin, vmax, threshold, "Pred Model",
                    reverse_color=reverse_color
                )
                path = os.path.join(save_dir, "voxel_pred.png")
                fig_pred.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
                fig_pred.savefig(path, dpi=300, facecolor='white', pad_inches=0.3)
                plt.close(fig_pred)
                saved_files.append(path)

        elif tab_idx == 6:  # 3D 等值面 - 分别保存真实和预测模型
            try:
                vmin = float(self.ed_rho_min.text())
                vmax = float(self.ed_rho_max.text())
            except:
                vmin, vmax = -150, 350
            threshold = self.sl_thresh.value() / 100.0
            x_range = self.data_mgr.x_range
            y_range = self.data_mgr.y_range
            
            has_true_density = hasattr(self.data_mgr, 'density_3d') and self.data_mgr.density_3d is not None
            
            if has_true_density:
                from scipy.ndimage import zoom as nd_zoom
                true_density = self.data_mgr.density_3d
                z_extent = true_density.shape[0] * self.dz
                
                if self.pred_real is not None and true_density.shape != self.pred_real.shape:
                    scale_factors = [self.pred_real.shape[i] / true_density.shape[i] for i in range(3)]
                    true_density = nd_zoom(true_density, scale_factors, order=1)
                    z_extent = self.pred_real.shape[0] * self.dz
                
                # 保存真实模型等值面
                fig_true = Figure(figsize=(12, 10))
                fig_true.patch.set_facecolor('white')
                ax_true = fig_true.add_subplot(111, projection='3d')
                Visualizer.plot_3d_isosurface(
                    ax_true, true_density, x_range, y_range, z_extent,
                    vmin, vmax, threshold, ""
                )
                fig_true.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "isosurface_true.png")
                fig_true.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_true)
                saved_files.append(path)
            
            if self.pred_real is not None:
                z_extent = self.pred_real.shape[0] * self.dz
                # 保存预测模型等值面
                fig_pred = Figure(figsize=(12, 10))
                fig_pred.patch.set_facecolor('white')
                ax_pred = fig_pred.add_subplot(111, projection='3d')
                Visualizer.plot_3d_isosurface(
                    ax_pred, self.pred_real, x_range, y_range, z_extent,
                    vmin, vmax, threshold, ""
                )
                fig_pred.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
                path = os.path.join(save_dir, "isosurface_pred.png")
                fig_pred.savefig(path, dpi=300, facecolor='white', pad_inches=0.2)
                plt.close(fig_pred)
                saved_files.append(path)
        
        self.status.showMessage(f"已保存 {len(saved_files)} 张图片到: {save_dir}")


# ============== 8. 主函数 ==============
def main():
    print("=" * 50)
    print("V4 Gravity Inversion Pro")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"V4 Available: {HAS_V4}")
    print()

    app = QApplication(sys.argv)

    # 设置样式
    app.setStyle('Fusion')

    window = V4GUIProWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
