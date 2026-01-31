import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
@dataclass
class V4Config:
    exp_name: str = "Gravity_Inversion_V4_PhysicsInformed"

    data_mode: str = 'joint'

    grid_shape: Tuple[int, int, int] = (16, 32, 32)
    dx: float = 100.0
    dz: float = 100.0

    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    decoder_channels: Tuple[int, ...] = (192, 96, 48)
    lifting_channels: int = 96
    num_transformer_layers: int = 6
    num_heads: int = 8
    use_position_encoding: bool = True
    use_depth_attention: bool = True

    lr: float = 1e-4
    batch_size: int = 16
    epochs: int = 300
    steps_per_epoch: int = 500

    w_depth: float = 1.0
    w_focus: float = 0.3
    w_gdl: float = 0.2
    w_physics: float = 0.3
    w_edge: float = 0.5
    w_l1: float = 0.001
    w_morph: float = 0.2
    w_adv: float = 0.0
    w_boundary: float = 0.5
    depth_beta: float = 2.0
    focus_beta: float = 30.0

    use_augmentation: bool = True
    elastic_alpha: float = 30.0
    elastic_sigma: float = 4.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "/home/jszxgx/ysw/deeplearn/checkpoints_v4_new"
    num_workers: int = 4
    resume_path: Optional[str] = None


class DifferentiableForward(nn.Module):
    """可微分重力正演算子 - 用于物理一致性损 """

    def __init__(self, shape: Tuple[int, int, int], dx: float, dz: float):
        super().__init__()
        self.nz, self.ny, self.nx = shape
        self.dx = dx
        self.dz = dz
        self.G = 6.674e-11

        kernel = self._build_kernel()
        self.register_buffer('kernel', torch.from_numpy(kernel).cfloat())

    def _build_kernel(self):
        pad_y, pad_x = self.ny // 2, self.nx // 2
        ext_ny, ext_nx = self.ny + 2 * pad_y, self.nx + 2 * pad_x

        ky = np.fft.fftfreq(ext_ny, d=self.dx) * 2 * np.pi
        kx = np.fft.fftfreq(ext_nx, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)
        K[0, 0] = 1e-6

        depths = np.arange(self.nz) * self.dz + (self.dz / 2)
        kernel_stack = []
        for z in depths:
            exp_arg = np.clip(-K * z, -100, 0)
            layer_kernel = 2 * np.pi * self.G * np.exp(exp_arg) * self.dz
            kernel_stack.append(layer_kernel)

        return np.array(kernel_stack)

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        original_dtype = density.dtype
        density = density.float()

        B = density.shape[0]
        rho = density.squeeze(1) * 1000.0

        rho = torch.clamp(rho, -1e6, 1e6)

        pad_y, pad_x = self.ny // 2, self.nx // 2
        rho_padded = F.pad(rho, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)

        rho_fft = torch.fft.fft2(rho_padded, dim=(-2, -1))

        field_fft = (rho_fft * self.kernel.unsqueeze(0)).sum(dim=1)

        field_padded = torch.fft.ifft2(field_fft, dim=(-2, -1)).real

        field = field_padded[:, pad_y:-pad_y, pad_x:-pad_x]

        result = field.unsqueeze(1) * 1e5
        result = torch.clamp(result, -1e6, 1e6)

        if torch.isnan(result).any():
            result = torch.nan_to_num(result, nan=0.0)

        return result


class FastGravityForward:
    """快速正演（用于数据生成）"""

    def __init__(self, shape: Tuple[int, int, int], dx: float, dz: float, mode: str = 'gz'):
        self.nz, self.ny, self.nx = shape
        self.dx, self.dz = dx, dz
        self.mode = mode
        self.G = 6.674e-11
        self._kernel_fft = self._build_kernel()

    def _build_kernel(self):
        pad_y, pad_x = self.ny // 2, self.nx // 2
        self.pad_y, self.pad_x = pad_y, pad_x
        ext_ny, ext_nx = self.ny + 2 * pad_y, self.nx + 2 * pad_x

        ky = np.fft.fftfreq(ext_ny, d=self.dx) * 2 * np.pi
        kx = np.fft.fftfreq(ext_nx, d=self.dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX ** 2 + KY ** 2)
        K[0, 0] = 1e-10

        depths = np.arange(self.nz) * self.dz + (self.dz / 2)

        if self.mode == 'joint':
            kernel_gz, kernel_gzz = [], []
            for z in depths:
                base = 2 * np.pi * self.G * np.exp(-K * z) * self.dz
                kernel_gz.append(base)
                kernel_gzz.append(-base * K)
            return np.stack([np.array(kernel_gz), np.array(kernel_gzz)], axis=0)
        else:
            kernel_stack = []
            for z in depths:
                layer = 2 * np.pi * self.G * np.exp(-K * z) * self.dz
                if self.mode == 'gzz':
                    layer = -layer * K
                kernel_stack.append(layer)
            return np.array(kernel_stack)

    def forward(self, density: np.ndarray) -> np.ndarray:
        rho = density * 1000.0
        rho_padded = np.pad(rho, ((0, 0), (self.pad_y, self.pad_y), (self.pad_x, self.pad_x)), mode='constant')
        rho_fft = np.fft.fft2(rho_padded, axes=(1, 2))

        if self.mode == 'joint':
            field_fft = np.sum(rho_fft[None, ...] * self._kernel_fft, axis=1)
            field_padded = np.fft.ifft2(field_fft, axes=(1, 2)).real
            field = field_padded[:, self.pad_y:-self.pad_y, self.pad_x:-self.pad_x]
            field[0] *= 1e5
            field[1] *= 1e9
            return field
        else:
            field_fft = np.sum(rho_fft * self._kernel_fft, axis=0)
            field_padded = np.fft.ifft2(field_fft).real
            field = field_padded[self.pad_y:-self.pad_y, self.pad_x:-self.pad_x]
            return field * (1e5 if self.mode == 'gz' else 1e9)


class GeoModelGenerator:
    """程序化地质模型生成器"""

    def __init__(self, shape):
        self.nz, self.ny, self.nx = shape
        self.shape = shape
        z = np.linspace(0, 1, self.nz)
        y = np.linspace(0, 1, self.ny)
        x = np.linspace(0, 1, self.nx)
        self.Z, self.Y, self.X = np.meshgrid(z, y, x, indexing='ij')

    def generate_voronoi(self, num_cells=12):
        seeds = np.random.rand(num_cells, 3) * np.array([self.nz, self.ny, self.nx])
        densities = np.random.uniform(0.0, 1.0, num_cells)
        densities[np.random.rand(num_cells) > 0.2] = 0.0

        tree = cKDTree(seeds)
        grid_points = np.stack([
            np.arange(self.nz).repeat(self.ny * self.nx),
            np.tile(np.arange(self.ny).repeat(self.nx), self.nz),
            np.tile(np.arange(self.nx), self.nz * self.ny)
        ], axis=1)
        _, indices = tree.query(grid_points)
        return densities[indices].reshape(self.shape)

    def generate_prism(self):
        model = np.zeros(self.shape)
        n_prisms = np.random.randint(1, 4)
        for _ in range(n_prisms):
            z0, z1 = sorted(np.random.randint(2, self.nz - 2, 2))
            y0, y1 = sorted(np.random.randint(4, self.ny - 4, 2))
            x0, x1 = sorted(np.random.randint(4, self.nx - 4, 2))
            val = np.random.uniform(0.3, 1.0) * (1 if np.random.rand() > 0.5 else -1)
            model[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = val
        return np.clip(model, -1, 1)

    def generate_sphere(self):
        model = np.zeros(self.shape)
        n_spheres = np.random.randint(1, 4)
        for _ in range(n_spheres):
            cz = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            cx = np.random.uniform(0.2, 0.8)
            r = np.random.uniform(0.1, 0.3)
            dist = np.sqrt((self.Z - cz) ** 2 + (self.Y - cy) ** 2 + (self.X - cx) ** 2)
            val = np.random.uniform(0.5, 1.0) * (1 if np.random.rand() > 0.5 else -1)
            model[dist < r] = val
        return model

    def generate_trapezoid(self):
        """生成梯形体 - 模拟锐利边缘的地质体"""
        model = np.zeros(self.shape)

        z_start = np.random.randint(2, 6)
        z_end = np.random.randint(z_start + 4, self.nz - 2)
        base_size = np.random.randint(4, 10)
        taper_rate = np.random.uniform(0.2, 0.5)
        density = np.random.uniform(0.6, 1.0) * (1 if np.random.rand() > 0.5 else -1)

        center_y, center_x = self.ny // 2 + np.random.randint(-4, 5), self.nx // 2 + np.random.randint(-4, 5)

        for z in range(z_start, z_end):
            depth_ratio = (z - z_start) / max(1, z_end - z_start)
            half_size = int(base_size * (1 + taper_rate * depth_ratio))

            y0 = max(0, center_y - half_size)
            y1 = min(self.ny, center_y + half_size)
            x0 = max(0, center_x - half_size)
            x1 = min(self.nx, center_x + half_size)

            model[z, y0:y1, x0:x1] = density

        return np.clip(model, -1, 1)

    def generate_staircase(self):
        """阶梯体生成器 - 模拟逐层递进的地质结构"""
        model = np.zeros(self.shape)

        num_steps = np.random.randint(4, 8)
        direction = np.random.choice(['x', 'y', 'diagonal'])
        density = np.random.uniform(0.6, 1.0) * (1 if np.random.rand() > 0.5 else -1)
        step_size = 2

        for i in range(num_steps):
            z_start = 2 + i * 2
            z_end = min(z_start + 2, self.nz - 1)

            if direction == 'x':
                x_center = 6 + i * 3
                y_center = self.ny // 2
            elif direction == 'y':
                x_center = self.nx // 2
                y_center = 6 + i * 3
            else:
                x_center = 6 + i * 3
                y_center = 6 + i * 2

            x_start = max(0, x_center - step_size)
            x_end = min(self.nx, x_center + step_size + 1)
            y_start = max(0, y_center - step_size)
            y_end = min(self.ny, y_center + step_size + 1)

            model[z_start:z_end, y_start:y_end, x_start:x_end] = density

        return np.clip(model, -1, 1)

    def generate_nested(self):
        """镶嵌体生成器 - 盒中盒结构，训练边界识别"""
        model = np.zeros(self.shape)

        outer_density = np.random.uniform(0.3, 0.6)
        outer_z = (3, self.nz - 3)
        outer_y = (6, self.ny - 6)
        outer_x = (6, self.nx - 6)
        model[outer_z[0]:outer_z[1], outer_y[0]:outer_y[1], outer_x[0]:outer_x[1]] = outer_density

        inner_density = np.random.uniform(0.7, 1.0) * (1 if np.random.rand() > 0.5 else -1)
        margin = np.random.randint(2, 5)
        inner_z = (outer_z[0] + margin, outer_z[1] - margin)
        inner_y = (outer_y[0] + margin, outer_y[1] - margin)
        inner_x = (outer_x[0] + margin, outer_x[1] - margin)

        if inner_z[1] > inner_z[0] and inner_y[1] > inner_y[0] and inner_x[1] > inner_x[0]:
            model[inner_z[0]:inner_z[1], inner_y[0]:inner_y[1], inner_x[0]:inner_x[1]] = inner_density

        return np.clip(model, -1, 1)

    def generate_multiscale(self):
        """多尺度块生成器 - 大中小块组合"""
        model = np.zeros(self.shape)

        z0, z1 = 4, 12
        y0, y1 = 8, 24
        x0, x1 = 8, 24
        model[z0:z1, y0:y1, x0:x1] = np.random.uniform(0.3, 0.5)

        for _ in range(np.random.randint(2, 4)):
            size = np.random.randint(3, 6)
            cz = np.random.randint(2, self.nz - size)
            cy = np.random.randint(2, self.ny - size)
            cx = np.random.randint(2, self.nx - size)
            val = np.random.uniform(0.5, 0.8)
            model[cz:cz + size, cy:cy + size, cx:cx + size] = val

        for _ in range(np.random.randint(3, 6)):
            size = 2
            cz = np.random.randint(2, self.nz - size)
            cy = np.random.randint(2, self.ny - size)
            cx = np.random.randint(2, self.nx - size)
            val = np.random.uniform(0.8, 1.0) * (1 if np.random.rand() > 0.5 else -1)
            model[cz:cz + size, cy:cy + size, cx:cx + size] = val

        return np.clip(model, -1, 1)

    def generate_fault(self):
        """断层结构生成器 - 斜切的地质断层"""
        model = np.zeros(self.shape)

        fault_angle = np.random.uniform(30, 60)
        fault_offset = np.random.randint(2, 5)
        density_left = np.random.uniform(0.5, 0.8) * (1 if np.random.rand() > 0.5 else -1)
        density_right = np.random.uniform(0.3, 0.6) * (1 if np.random.rand() > 0.5 else -1)

        tan_angle = np.tan(np.radians(fault_angle))

        for z in range(self.nz):
            for y in range(self.ny):
                fault_x = int(self.nx / 2 + (z - self.nz / 2) * tan_angle)

                for x in range(max(0, fault_x - fault_offset)):
                    if 3 < z < self.nz - 2:
                        model[z, y, x] = density_left

                for x in range(min(self.nx, fault_x + fault_offset), self.nx):
                    if 3 + fault_offset < z < self.nz - 2:
                        model[z, y, x] = density_right

        return np.clip(model, -1, 1)

    def generate_perlin_terrain(self):
        """柏林噪声地形生成器 - 模拟自然地质变化"""
        from scipy.ndimage import gaussian_filter

        model = np.zeros(self.shape)

        noise_large = np.random.randn(self.nz // 4, self.ny // 4, self.nx // 4)
        noise_large = gaussian_filter(noise_large, sigma=1)
        from scipy.ndimage import zoom
        noise_large = zoom(noise_large, (4, 4, 4), order=1)[:self.nz, :self.ny, :self.nx]

        noise_mid = np.random.randn(self.nz // 2, self.ny // 2, self.nx // 2)
        noise_mid = gaussian_filter(noise_mid, sigma=0.5)
        noise_mid = zoom(noise_mid, (2, 2, 2), order=1)[:self.nz, :self.ny, :self.nx]

        noise_small = np.random.randn(self.nz, self.ny, self.nx)
        noise_small = gaussian_filter(noise_small, sigma=0.3)

        model = 0.5 * noise_large + 0.3 * noise_mid + 0.2 * noise_small

        threshold = np.percentile(model, 70)
        model[model < threshold] = 0
        model[model >= threshold] = (model[model >= threshold] - threshold) / (model.max() - threshold + 1e-6)

        return np.clip(model, -1, 1)

    def generate_fractal(self):
        """分形结构生成器 - 模拟自相似地质结构"""
        model = np.zeros(self.shape)

        def add_box(z0, z1, y0, y1, x0, x1, density, depth=0, max_depth=3):
            if depth >= max_depth:
                return
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                return

            model[z0:z1, y0:y1, x0:x1] = density

            if np.random.rand() < 0.7:
                sub_size_z = max(1, (z1 - z0) // 3)
                sub_size_y = max(1, (y1 - y0) // 3)
                sub_size_x = max(1, (x1 - x0) // 3)

                corners = [
                    (z0, y0, x0),
                    (z0, y0, x1 - sub_size_x),
                    (z0, y1 - sub_size_y, x0),
                    (z1 - sub_size_z, y0, x0),
                ]

                for cz, cy, cx in corners:
                    if np.random.rand() < 0.5:
                        new_density = density * np.random.uniform(0.8, 1.2)
                        add_box(
                            cz, cz + sub_size_z,
                            cy, cy + sub_size_y,
                            cx, cx + sub_size_x,
                            np.clip(new_density, -1, 1),
                                depth + 1, max_depth
                        )

        initial_density = np.random.uniform(0.4, 0.7)
        add_box(2, self.nz - 2, 4, self.ny - 4, 4, self.nx - 4, initial_density)

        return np.clip(model, -1, 1)

    def generate_figure8_staircase(self):
        model = np.zeros(self.shape)

        step_count = np.random.randint(4, 7)
        step_size = 2
        density = np.random.uniform(0.7, 1.0)

        for i in range(step_count):
            x_center = 6 + i * 3
            z_center = 3 + i * 2
            y_center = 10 + np.random.randint(-2, 3)

            x_start, x_end = max(0, x_center - step_size // 2), min(self.nx, x_center + step_size // 2 + 1)
            y_start, y_end = max(0, y_center - step_size // 2), min(self.ny, y_center + step_size // 2 + 1)
            z_start, z_end = max(0, z_center - step_size // 2), min(self.nz, z_center + step_size // 2 + 1)

            model[z_start:z_end, y_start:y_end, x_start:x_end] = density

        for i in range(step_count):
            x_center = self.nx - 7 - i * 3
            z_center = 3 + i * 2
            y_center = 22 + np.random.randint(-2, 3)

            x_start, x_end = max(0, x_center - step_size // 2), min(self.nx, x_center + step_size // 2 + 1)
            y_start, y_end = max(0, y_center - step_size // 2), min(self.ny, y_center + step_size // 2 + 1)
            z_start, z_end = max(0, z_center - step_size // 2), min(self.nz, z_center + step_size // 2 + 1)

            model[z_start:z_end, y_start:y_end, x_start:x_end] = density * 0.9

        return np.clip(model, -1, 1)

    def generate_separated_prisms(self):
        model = np.zeros(self.shape)

        direction = np.random.choice(['x', 'y', 'z'])

        if np.random.rand() > 0.5:
            density1 = np.random.uniform(0.5, 1.0)
            density2 = -np.random.uniform(0.4, 0.8)
        else:
            density1 = np.random.uniform(0.5, 1.0)
            density2 = np.random.uniform(0.5, 1.0)
            if np.random.rand() > 0.5:
                density1, density2 = -density1, -density2

        gap = np.random.randint(2, 6)

        if direction == 'x':
            mid_x = self.nx // 2
            x1_start, x1_end = 4, mid_x - gap
            x2_start, x2_end = mid_x + gap, self.nx - 4
            y_start, y_end = 6, self.ny - 6
            z_start, z_end = 3, self.nz - 3

            if x1_end > x1_start and x2_end > x2_start:
                model[z_start:z_end, y_start:y_end, x1_start:x1_end] = density1
                model[z_start:z_end, y_start:y_end, x2_start:x2_end] = density2

        elif direction == 'y':
            mid_y = self.ny // 2
            y1_start, y1_end = 4, mid_y - gap
            y2_start, y2_end = mid_y + gap, self.ny - 4
            x_start, x_end = 6, self.nx - 6
            z_start, z_end = 3, self.nz - 3

            if y1_end > y1_start and y2_end > y2_start:
                model[z_start:z_end, y1_start:y1_end, x_start:x_end] = density1
                model[z_start:z_end, y2_start:y2_end, x_start:x_end] = density2

        else:
            mid_z = self.nz // 2
            z1_start, z1_end = 2, mid_z - gap // 2
            z2_start, z2_end = mid_z + gap // 2, self.nz - 2
            x_start, x_end = 6, self.nx - 6
            y_start, y_end = 6, self.ny - 6

            if z1_end > z1_start and z2_end > z2_start:
                model[z1_start:z1_end, y_start:y_end, x_start:x_end] = density1
                model[z2_start:z2_end, y_start:y_end, x_start:x_end] = density2

        return np.clip(model, -1, 1)

    def generate_hollow_box(self):
        model = np.zeros(self.shape)

        shell_density = np.random.uniform(0.6, 1.0) * (1 if np.random.rand() > 0.5 else -1)
        thickness = np.random.randint(2, 4)

        z_out = (3, self.nz - 3)
        y_out = (5, self.ny - 5)
        x_out = (5, self.nx - 5)

        z_in = (z_out[0] + thickness, z_out[1] - thickness)
        y_in = (y_out[0] + thickness, y_out[1] - thickness)
        x_in = (x_out[0] + thickness, x_out[1] - thickness)

        model[z_out[0]:z_out[1], y_out[0]:y_out[1], x_out[0]:x_out[1]] = shell_density

        if z_in[1] > z_in[0] and y_in[1] > y_in[0] and x_in[1] > x_in[0]:
            model[z_in[0]:z_in[1], y_in[0]:y_in[1], x_in[0]:x_in[1]] = 0.0

        return np.clip(model, -1, 1)

    def generate_layered_density(self):
        model = np.zeros(self.shape)

        num_layers = np.random.randint(3, 6)
        layer_height = self.nz // num_layers

        for i in range(num_layers):
            z_start = i * layer_height + 1
            z_end = min((i + 1) * layer_height - 1, self.nz - 1)

            if np.random.rand() > 0.3:
                density = np.random.uniform(0.4, 1.0)
                y_start, y_end = 5, self.ny - 5
                x_start, x_end = 5, self.nx - 5
                model[z_start:z_end, y_start:y_end, x_start:x_end] = density

        return np.clip(model, -1, 1)

    def generate_scattered_anomalies(self):
        model = np.zeros(self.shape)

        n_bodies = np.random.randint(3, 7)

        n_positive = np.random.randint(1, n_bodies)
        n_negative = n_bodies - n_positive

        centers = []

        for i in range(n_bodies):
            size_type = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
            if size_type == 'small':
                sz, sy, sx = np.random.randint(2, 4), np.random.randint(3, 5), np.random.randint(3, 5)
            elif size_type == 'medium':
                sz, sy, sx = np.random.randint(3, 5), np.random.randint(4, 7), np.random.randint(4, 7)
            else:
                sz, sy, sx = np.random.randint(4, 7), np.random.randint(6, 10), np.random.randint(6, 10)

            for _ in range(20):
                cz = np.random.randint(2, self.nz - sz - 1)
                cy = np.random.randint(4, self.ny - sy - 3)
                cx = np.random.randint(4, self.nx - sx - 3)

                overlap = False
                for (oz, oy, ox) in centers:
                    if abs(cz - oz) < 4 and abs(cy - oy) < 6 and abs(cx - ox) < 6:
                        overlap = True
                        break
                if not overlap:
                    break

            centers.append((cz, cy, cx))

            if i < n_positive:
                density = np.random.uniform(0.5, 1.0)
            else:
                density = -np.random.uniform(0.4, 0.9)

            z0, z1 = cz, min(cz + sz, self.nz)
            y0, y1 = cy, min(cy + sy, self.ny)
            x0, x1 = cx, min(cx + sx, self.nx)
            model[z0:z1, y0:y1, x0:x1] = density

        return np.clip(model, -1, 1)

    def generate_realistic_anomaly(self):
        """真实异常体生成器 """
        model = np.zeros(self.shape)

        main_cz = np.random.randint(2, 5)
        main_cy = np.random.randint(10, 22)
        main_cx = np.random.randint(10, 22)
        main_sz = np.random.randint(4, 8)
        main_sy = np.random.randint(6, 12)
        main_sx = np.random.randint(6, 12)
        main_density = np.random.uniform(0.6, 1.0)

        z0, z1 = main_cz, min(main_cz + main_sz, self.nz)
        y0, y1 = max(0, main_cy - main_sy // 2), min(self.ny, main_cy + main_sy // 2)
        x0, x1 = max(0, main_cx - main_sx // 2), min(self.nx, main_cx + main_sx // 2)
        model[z0:z1, y0:y1, x0:x1] = main_density

        for _ in range(np.random.randint(1, 4)):
            sec_cz = np.random.randint(3, 10)
            sec_cy = np.random.randint(5, self.ny - 5)
            sec_cx = np.random.randint(5, self.nx - 5)
            sec_sz = np.random.randint(2, 5)
            sec_sy = np.random.randint(3, 7)
            sec_sx = np.random.randint(3, 7)
            sec_density = np.random.uniform(0.4, 0.8) * (1 if np.random.rand() > 0.5 else -1)

            z0, z1 = sec_cz, min(sec_cz + sec_sz, self.nz)
            y0, y1 = max(0, sec_cy - sec_sy // 2), min(self.ny, sec_cy + sec_sy // 2)
            x0, x1 = max(0, sec_cx - sec_sx // 2), min(self.nx, sec_cx + sec_sx // 2)
            model[z0:z1, y0:y1, x0:x1] = sec_density

        if np.random.rand() > 0.5:
            deep_z = np.random.randint(8, 12)
            model[deep_z:, :, :] += np.random.uniform(-0.2, 0.2)

        return np.clip(model, -1, 1)

    def mixup_generator(self):
        """混合生成器 - 包含真实场景模拟以改善Gzz反演"""
        generators = {
            'voronoi': (self.generate_voronoi, 0.03),
            'prism': (self.generate_prism, 0.16),
            'sphere': (self.generate_sphere, 0.03),
            'trapezoid': (self.generate_trapezoid, 0.06),
            'staircase': (self.generate_staircase, 0.06),
            'nested': (self.generate_nested, 0.06),
            'multiscale': (self.generate_multiscale, 0.05),
            'fault': (self.generate_fault, 0.03),
            'perlin': (self.generate_perlin_terrain, 0.03),
            'fractal': (self.generate_fractal, 0.02),
            'figure8': (self.generate_figure8_staircase, 0.03),
            'separated': (self.generate_separated_prisms, 0.14),
            'hollow': (self.generate_hollow_box, 0.05),
            'layered': (self.generate_layered_density, 0.03),
            'scattered': (self.generate_scattered_anomalies, 0.12),
            'realistic': (self.generate_realistic_anomaly, 0.10),
        }

        names = list(generators.keys())
        probs = [generators[n][1] for n in names]

        choice = np.random.choice(names, p=probs)
        return generators[choice][0]()

    def simple_generator(self):
        """简单生成器 - 仅使用基础形状，用于课程学习初期"""
        simple_generators = {
            'prism': (self.generate_prism, 0.35),
            'sphere': (self.generate_sphere, 0.25),
            'trapezoid': (self.generate_trapezoid, 0.25),
            'nested': (self.generate_nested, 0.15),
        }

        names = list(simple_generators.keys())
        probs = [simple_generators[n][1] for n in names]

        choice = np.random.choice(names, p=probs)
        return simple_generators[choice][0]()


class AdvancedAugmentation:
    """高级数据增强"""

    @staticmethod
    def elastic_deformation(density, alpha=30, sigma=4):
        """弹性形变 - 模拟地质褶皱"""
        shape = density.shape
        dx = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma)
        dy = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma)
        dz = ndimage.gaussian_filter(np.random.randn(*shape) * alpha, sigma)

        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                              np.arange(shape[2]), indexing='ij')
        indices = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1)
        ]
        return ndimage.map_coordinates(density, indices, order=1, mode='reflect')

    @staticmethod
    def depth_shift(density, max_shift=2):
        """深度偏移"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return density
        return np.roll(density, shift, axis=0)

    @staticmethod
    def add_geological_noise(density, noise_level=0.03):
        """地质噪声"""
        noise = np.random.randn(*density.shape) * noise_level
        noise = ndimage.gaussian_filter(noise, sigma=1)
        return np.clip(density + noise, -1, 1)

    @staticmethod
    def random_flip(density):
        """随机翻转"""
        if np.random.rand() > 0.5:
            density = np.flip(density, axis=2).copy()
        if np.random.rand() > 0.5:
            density = np.flip(density, axis=1).copy()
        return density

    @staticmethod
    def random_scale(density, scale_range=(0.8, 1.2)):
        """随机缩放密度值"""
        scale = np.random.uniform(*scale_range)
        return np.clip(density * scale, -1, 1)


class V4Dataset(Dataset):
    """V4 数据集 - 支持高级增强和课程学习"""

    def __init__(self, config: V4Config, epoch_len: int, mode: str = 'train'):
        self.config = config
        self.epoch_len = epoch_len
        self.mode = mode
        self.gen = None
        self.forward_op = None
        self.augmentor = AdvancedAugmentation()
        self.current_epoch = 0
        self.curriculum_threshold = 30

    def set_epoch(self, epoch: int):
        """设置当前 epoch,用于课程学习"""
        self.current_epoch = epoch

    def _init_workers(self):
        if self.gen is None:
            self.gen = GeoModelGenerator(self.config.grid_shape)
            self.forward_op = FastGravityForward(
                self.config.grid_shape,
                self.config.dx,
                self.config.dz,
                mode=self.config.data_mode
            )

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        self._init_workers()
        nz, ny, nx = self.config.grid_shape

        if self.current_epoch < self.curriculum_threshold:
            complex_prob = self.current_epoch / self.curriculum_threshold * 0.5
            if np.random.rand() > complex_prob:
                density = self.gen.simple_generator()
            else:
                density = self.gen.mixup_generator()
        else:
            density = self.gen.mixup_generator()

        if self.mode == 'train' and self.config.use_augmentation:
            density = self.augmentor.random_flip(density)
            density = self.augmentor.random_scale(density)
            if np.random.rand() > 0.7:
                density = self.augmentor.elastic_deformation(
                    density,
                    self.config.elastic_alpha,
                    self.config.elastic_sigma
                )
            if np.random.rand() > 0.5:
                density = self.augmentor.depth_shift(density)
            if np.random.rand() > 0.5:
                density = self.augmentor.add_geological_noise(density)

        field_data = self.forward_op.forward(density)

        if self.mode == 'train':
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, np.std(field_data) * noise_level, field_data.shape)
            field_data = field_data + noise

        density_tensor = torch.from_numpy(density.copy()).float().unsqueeze(0)

        if self.config.data_mode == 'joint':
            gz = field_data[0]
            gzz = field_data[1]

            gz_max = np.abs(gz).max() + 1e-8
            gzz_max = np.abs(gzz).max() + 1e-8
            gz_norm = gz / gz_max
            gzz_norm = gzz / gzz_max

            gz_vol = torch.from_numpy(gz_norm).float().unsqueeze(0).expand(nz, -1, -1)
            gzz_vol = torch.from_numpy(gzz_norm).float().unsqueeze(0).expand(nz, -1, -1)
            z_indices = np.linspace(0, 1, nz)
            z_map = np.tile(z_indices[:, None, None], (1, ny, nx))
            z_tensor = torch.from_numpy(z_map).float()
            input_vol = torch.stack([gz_vol, gzz_vol, z_tensor], dim=0)

            obs_gravity = torch.from_numpy(field_data[0].copy()).float().unsqueeze(0)
        else:
            gz = field_data
            gz_max = np.abs(gz).max() + 1e-8
            gz_norm = gz / gz_max

            gz_vol = torch.from_numpy(gz_norm).float().unsqueeze(0).expand(nz, -1, -1)
            z_indices = np.linspace(0, 1, nz)
            z_map = np.tile(z_indices[:, None, None], (1, ny, nx))
            z_tensor = torch.from_numpy(z_map).float()
            input_vol = torch.stack([gz_vol, z_tensor], dim=0)
            obs_gravity = torch.from_numpy(field_data.copy()).float().unsqueeze(0)

        return input_vol, density_tensor, obs_gravity


class SEBlock3D(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResBlock2D(nn.Module):
    """2D 残差块"""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + res)


class ResBlock3D(nn.Module):
    """3D 残差块 with SE"""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.se = SEBlock3D(out_c)
        self.act = nn.GELU()

        self.shortcut = nn.Identity()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, 1, bias=False),
                nn.BatchNorm3d(out_c)
            )

    def forward(self, x):
        res = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return self.act(x + res)


class Encoder2D(nn.Module):
    """2D Encoder - 提取表面场特征"""

    def __init__(self, in_channels, channels=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU()
        )

        self.stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.stages.append(nn.Sequential(
                ResBlock2D(channels[i], channels[i + 1], stride=2),
                ResBlock2D(channels[i + 1], channels[i + 1])
            ))

        self.out_channels = channels[-1]

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return x, features


class ChannelToDepthLifting(nn.Module):
    """将 2D 特征的通道维映射到 3D 的深度维"""

    def __init__(self, in_channels, out_depth, out_channels):
        super().__init__()
        self.out_depth = out_depth
        self.out_channels = out_channels
        self.expand = nn.Conv2d(in_channels, out_depth * out_channels, 1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.expand(x)
        B, _, H, W = x.shape
        x = x.view(B, self.out_depth, self.out_channels, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.act(self.bn(x))


class Sinusoidal3DPositionEncoding(nn.Module):
    """3D 正弦位置编码"""

    def __init__(self, channels, max_len=64):
        super().__init__()
        self.channels = channels

        d_model = channels // 3
        pe = torch.zeros(max_len, max_len, max_len, channels)

        for dim in range(3):
            for i in range(d_model):
                div_term = 10000 ** (2 * i / d_model)
                for pos in range(max_len):
                    if dim == 0:
                        pe[pos, :, :, dim * d_model + i] = np.sin(pos / div_term) if i % 2 == 0 else np.cos(
                            pos / div_term)
                    elif dim == 1:
                        pe[:, pos, :, dim * d_model + i] = np.sin(pos / div_term) if i % 2 == 0 else np.cos(
                            pos / div_term)
                    else:
                        pe[:, :, pos, dim * d_model + i] = np.sin(pos / div_term) if i % 2 == 0 else np.cos(
                            pos / div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        D, H, W = x.shape[2:]
        pos_enc = self.pe[:D, :H, :W, :self.channels].permute(3, 0, 1, 2)
        return x + pos_enc.unsqueeze(0)


class DepthAttention(nn.Module):
    """深度注意力机制"""

    def __init__(self, channels, depth):
        super().__init__()
        self.depth_query = nn.Parameter(torch.randn(1, channels, depth, 1, 1) * 0.02)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        attn = torch.sigmoid(torch.sum(x * self.depth_query, dim=1, keepdim=True))
        return x * (1 + attn)


class TransformerLayer3D(nn.Module):

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 4, channels),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + res


class TransformerBottleneck3D(nn.Module):

    def __init__(self, channels, depth, num_heads=8, num_layers=4, use_pos_enc=True):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            self.pos_enc = Sinusoidal3DPositionEncoding(channels)

        self.layers = nn.ModuleList([
            TransformerLayer3D(channels, num_heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        if self.use_pos_enc:
            x = self.pos_enc(x)

        B, C, D, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)

        for layer in self.layers:
            x_flat = layer(x_flat)

        return x_flat.permute(0, 2, 1).view(B, C, D, H, W)


class Decoder3D(nn.Module):
    """3D Decoder - 只在 H/W 方向上采样,D 保持不变"""

    def __init__(self, in_channels, channels=(128, 64, 32), use_depth_attn=True, depth=16):
        super().__init__()
        self.use_depth_attn = use_depth_attn
        self.depth = depth

        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.depth_attns = nn.ModuleList() if use_depth_attn else None

        self.multi_scale_heads = nn.ModuleList()

        prev_c = in_channels
        for i, c in enumerate(channels):
            self.ups.append(nn.ConvTranspose3d(prev_c, c, kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            self.blocks.append(ResBlock3D(c, c))
            if use_depth_attn:
                self.depth_attns.append(DepthAttention(c, depth))
            self.multi_scale_heads.append(nn.Conv3d(c, 1, 1))
            prev_c = c

        self.final = nn.Conv3d(channels[-1], 1, 1)

        self.scale_weights = nn.Parameter(torch.ones(len(channels) + 1) / (len(channels) + 1))

    def forward(self, x, return_multi_scale=False):
        multi_scale_outputs = []

        for i, (up, block) in enumerate(zip(self.ups, self.blocks)):
            x = up(x)
            x = block(x)
            if self.use_depth_attn:
                x = self.depth_attns[i](x)

            scale_pred = self.multi_scale_heads[i](x)
            multi_scale_outputs.append(scale_pred)

        final_out = self.final(x)
        multi_scale_outputs.append(final_out)

        target_size = final_out.shape[2:]

        if return_multi_scale:
            return final_out, multi_scale_outputs

        weights = F.softmax(self.scale_weights, dim=0)
        fused = torch.zeros_like(final_out)

        for i, scale_out in enumerate(multi_scale_outputs):
            if scale_out.shape[2:] != target_size:
                scale_out = F.interpolate(scale_out, size=target_size, mode='trilinear', align_corners=False)
            fused = fused + weights[i] * scale_out

        return fused


class CrossDimAttention(nn.Module):
    """跨维度注意力模块 - 融合 2D 和 3D 特征"""

    def __init__(self, channels_2d, channels_3d, depth):
        super().__init__()
        self.depth = depth

        self.proj_2d = nn.Sequential(
            nn.Conv2d(channels_2d, channels_3d, 1),
            nn.BatchNorm2d(channels_3d),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv3d(channels_3d * 2, channels_3d // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels_3d // 4, 1, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(channels_3d, channels_3d, 3, padding=1),
            nn.InstanceNorm3d(channels_3d),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_3d, feat_2d):
        B, C3d, D, H, W = feat_3d.shape

        feat_2d_proj = self.proj_2d(feat_2d)

        feat_2d_expanded = feat_2d_proj.unsqueeze(2).expand(-1, -1, D, -1, -1)

        combined = torch.cat([feat_3d, feat_2d_expanded], dim=1)
        attn = self.attention(combined)

        enhanced = feat_3d + attn * feat_2d_expanded

        return self.fusion(enhanced)


class PhysicsInformedUNet(nn.Module):

    def __init__(self, config: V4Config):
        super().__init__()
        self.config = config

        in_c = 3 if config.data_mode == 'joint' else 2

        self.encoder_2d = Encoder2D(in_c, config.encoder_channels)

        nz = config.grid_shape[0]
        self.lifting = ChannelToDepthLifting(
            config.encoder_channels[-1],
            nz,
            config.lifting_channels
        )

        self.transformer = TransformerBottleneck3D(
            config.lifting_channels,
            nz,
            config.num_heads,
            config.num_transformer_layers,
            config.use_position_encoding
        )

        self.cross_dim_attention = CrossDimAttention(
            channels_2d=config.encoder_channels[-1],
            channels_3d=config.lifting_channels,
            depth=nz
        )

        self.decoder_3d = Decoder3D(
            config.lifting_channels,
            config.decoder_channels,
            config.use_depth_attention,
            depth=nz
        )

        self.unsharp_mask = UnsharpMask3D(amount=0.3, sigma=1.0)
        self.use_unsharp = True

    def forward(self, x):
        x_2d = x[:, :, 0, :, :]

        feat_2d, encoder_features = self.encoder_2d(x_2d)

        feat_3d = self.lifting(feat_2d)

        feat_3d = self.transformer(feat_3d)

        feat_3d = self.cross_dim_attention(feat_3d, feat_2d)

        out = self.decoder_3d(feat_3d)

        if self.use_unsharp:
            out = self.unsharp_mask(out)

        return out


class Morphology3D(nn.Module):
    """3D 形态学操作 (可微分近似)"""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size, stride=1, padding=kernel_size // 2)
        self.neg_pool = nn.MaxPool3d(kernel_size, stride=1, padding=kernel_size // 2)

    def dilate(self, x):
        """膨胀操作"""
        return self.pool(x)

    def erode(self, x):
        """腐蚀操作 (通过负值膨胀实现)"""
        return -self.neg_pool(-x)

    def opening(self, x):
        """开运算 (先腐蚀后膨胀) - 去除小噪声"""
        return self.dilate(self.erode(x))

    def closing(self, x):
        """闭运算 (先膨胀后腐蚀) - 填充小孔"""
        return self.erode(self.dilate(x))


class Discriminator3D(nn.Module):
    """3D 判别器 - 用于对抗训练"""

    def __init__(self, in_channels=1):
        super().__init__()

        def conv_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 4, stride, 1, bias=False),
                nn.InstanceNorm3d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.Conv3d(128, 1, kernel_size=(2, 4, 4), stride=1, padding=0)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1).mean(dim=1)


class UnsharpMask3D(nn.Module):
    """3D Unsharp Mask 边界增强模块

    通过减去模糊版本来增强边界:
    enhanced = original + amount * (original - blurred)
    """

    def __init__(self, amount=0.5, sigma=1.0):
        super().__init__()
        self.amount = amount
        self.sigma = sigma

        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        self.register_buffer('kernel_z', gaussian_1d.view(1, 1, -1, 1, 1))
        self.register_buffer('kernel_y', gaussian_1d.view(1, 1, 1, -1, 1))
        self.register_buffer('kernel_x', gaussian_1d.view(1, 1, 1, 1, -1))
        self.pad = kernel_size // 2

    def forward(self, x):
        blurred = F.conv3d(x, self.kernel_z.expand(x.size(1), -1, -1, -1, -1),
                           padding=(self.pad, 0, 0), groups=x.size(1))
        blurred = F.conv3d(blurred, self.kernel_y.expand(x.size(1), -1, -1, -1, -1),
                           padding=(0, self.pad, 0), groups=x.size(1))
        blurred = F.conv3d(blurred, self.kernel_x.expand(x.size(1), -1, -1, -1, -1),
                           padding=(0, 0, self.pad), groups=x.size(1))

        enhanced = x + self.amount * (x - blurred)

        return enhanced


class DepthWeightedLoss(nn.Module):
    """深度加权 MSE"""

    def __init__(self, beta=1.5, epsilon=0.1):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self._weights = None

    def forward(self, pred, target):
        D = pred.shape[2]

        if self._weights is None or self._weights.shape[0] != D:
            z_indices = torch.arange(1, D + 1, device=pred.device).float()
            self._weights = torch.pow(z_indices + self.epsilon, self.beta)
            self._weights = self._weights / self._weights.mean()
            self._weights = self._weights.view(1, 1, D, 1, 1)

        return torch.mean(self._weights * (pred - target) ** 2)


class ComprehensiveLoss(nn.Module):
    """综合损失函数"""

    def __init__(self, config: V4Config, forward_op: DifferentiableForward):
        super().__init__()
        self.config = config
        self.forward_op = forward_op
        self.depth_loss = DepthWeightedLoss(config.depth_beta)
        self.morphology = Morphology3D(kernel_size=3)

    def _gradient_loss(self, pred, target):
        """梯度差异损失"""

        def get_grad(x):
            dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
            dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
            dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
            return dz, dy, dx

        dz_p, dy_p, dx_p = get_grad(pred)
        dz_t, dy_t, dx_t = get_grad(target)

        return (torch.mean((dz_p - dz_t) ** 2) +
                torch.mean((dy_p - dy_t) ** 2) +
                torch.mean((dx_p - dx_t) ** 2))

    def _edge_loss(self, pred, target):
        """边缘增强损失 - 惩罚边缘模糊"""

        def compute_edge(x):
            dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
            dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
            dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
            edge_z = torch.abs(dz)
            edge_y = torch.abs(dy)
            edge_x = torch.abs(dx)
            return edge_z, edge_y, edge_x

        edge_z_p, edge_y_p, edge_x_p = compute_edge(pred)
        edge_z_t, edge_y_t, edge_x_t = compute_edge(target)

        weight_z = 1.0 + 5.0 * edge_z_t.detach()
        weight_y = 1.0 + 5.0 * edge_y_t.detach()
        weight_x = 1.0 + 5.0 * edge_x_t.detach()

        loss_z = torch.mean(weight_z * (edge_z_p - edge_z_t) ** 2)
        loss_y = torch.mean(weight_y * (edge_y_p - edge_y_t) ** 2)
        loss_x = torch.mean(weight_x * (edge_x_p - edge_x_t) ** 2)

        return loss_z + loss_y + loss_x

    def _morphology_loss(self, pred, target):
        """形态学一致性损失 - 使用腐蚀膨胀约束形状"""
        pred_opened = self.morphology.opening(pred)
        target_opened = self.morphology.opening(target)

        pred_closed = self.morphology.closing(pred)
        target_closed = self.morphology.closing(target)

        loss_open = F.mse_loss(pred_opened, pred.detach())
        loss_close = F.mse_loss(pred_closed, pred.detach())

        loss_match = F.mse_loss(pred_opened, target_opened) + F.mse_loss(pred_closed, target_closed)

        return loss_open + loss_close + 0.5 * loss_match

    def _boundary_loss(self, pred, target):
        """边界感知损失 - 使用 Sobel 算子检测边界并加强惩罚"""

        def sobel_3d(x):
            dz = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
            dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
            dx = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
            return dz, dy, dx

        dz_t, dy_t, dx_t = sobel_3d(target)
        dz_p, dy_p, dx_p = sobel_3d(pred)

        edge_t_z = dz_t / (dz_t.max() + 1e-6)
        edge_t_y = dy_t / (dy_t.max() + 1e-6)
        edge_t_x = dx_t / (dx_t.max() + 1e-6)

        weight_z = 1.0 + 10.0 * edge_t_z.detach()
        weight_y = 1.0 + 10.0 * edge_t_y.detach()
        weight_x = 1.0 + 10.0 * edge_t_x.detach()

        loss_z = torch.mean(weight_z * (dz_p - dz_t) ** 2)
        loss_y = torch.mean(weight_y * (dy_p - dy_t) ** 2)
        loss_x = torch.mean(weight_x * (dx_p - dx_t) ** 2)

        edge_sharpness = torch.mean(edge_t_z * (1.0 - torch.tanh(dz_p * 5)))

        return loss_z + loss_y + loss_x + 0.5 * edge_sharpness

    def forward(self, pred, target, obs_gravity):
        losses = {}

        losses['depth'] = self.depth_loss(pred, target) * self.config.w_depth

        focus_weight = 1.0 + self.config.focus_beta * torch.abs(target)
        losses['focus'] = torch.mean(focus_weight * (pred - target) ** 2) * self.config.w_focus

        losses['gdl'] = self._gradient_loss(pred, target) * self.config.w_gdl

        losses['edge'] = self._edge_loss(pred, target) * self.config.w_edge

        pred_gravity = self.forward_op(pred)
        obs_gravity = obs_gravity.float()
        physics_loss = F.mse_loss(pred_gravity, obs_gravity)

        if torch.isnan(physics_loss) or torch.isinf(physics_loss):
            physics_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        losses['physics'] = physics_loss * self.config.w_physics

        losses['l1'] = torch.mean(torch.abs(pred)) * self.config.w_l1

        losses['morph'] = self._morphology_loss(pred, target) * self.config.w_morph

        losses['boundary'] = self._boundary_loss(pred, target) * self.config.w_boundary

        losses['contrast'] = self._contrast_loss(pred, target) * 0.5

        total = sum(losses.values())
        return total, losses

    def _contrast_loss(self, pred, target):
        """对比度损失 - 确保预测能区分不同密度区域"""
        target_flat = target.view(target.size(0), -1)
        pred_flat = pred.view(pred.size(0), -1)

        target_min = target_flat.min(dim=1)[0]
        target_max = target_flat.max(dim=1)[0]
        pred_min = pred_flat.min(dim=1)[0]
        pred_max = pred_flat.max(dim=1)[0]

        range_loss = F.mse_loss(pred_min, target_min) + F.mse_loss(pred_max, target_max)

        target_var = target_flat.var(dim=1)
        pred_var = pred_flat.var(dim=1)
        var_loss = F.mse_loss(pred_var, target_var)

        sign_target = torch.sign(target)
        sign_pred = torch.sign(pred)
        sign_loss = F.mse_loss(sign_pred, sign_target)

        return range_loss + var_loss + 0.5 * sign_loss


class AdaptiveLossBalancer:
    """损失权重自适应平衡器

    在训练初期自动调整各损失项权重，使它们在同一数量级。
    """

    def __init__(self, loss_names, target_scale=1.0, momentum=0.9):
        self.loss_names = loss_names
        self.target_scale = target_scale
        self.momentum = momentum
        self.running_scales = {name: 1.0 for name in loss_names}
        self.initialized = False

    def update(self, losses: Dict[str, torch.Tensor]):
        """更新运行时缩放因子"""
        for name, loss in losses.items():
            if name not in self.running_scales:
                continue

            loss_val = loss.item() if torch.is_tensor(loss) else loss
            if loss_val < 1e-10:
                continue

            scale = self.target_scale / (loss_val + 1e-10)

            if not self.initialized:
                self.running_scales[name] = scale
            else:
                self.running_scales[name] = (
                        self.momentum * self.running_scales[name] +
                        (1 - self.momentum) * scale
                )

        self.initialized = True

    def get_balanced_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """返回平衡后的总损失"""
        total = 0
        for name, loss in losses.items():
            scale = self.running_scales.get(name, 1.0)
            total = total + loss * scale
        return total

    def get_scales(self) -> Dict[str, float]:
        return self.running_scales.copy()


def verify_gradient_flow(model, forward_op, device='cuda'):
    """验证梯度流是否正确传播

    检查物理损失是否能正确反向传播到模型参数
    """
    print("=" * 60)
    print("梯度流验证 (Gradient Flow Verification)")
    print("=" * 60)

    model.train()
    model.to(device)
    forward_op.to(device)

    dummy_input = torch.randn(1, 3, 16, 32, 32, device=device, requires_grad=True)
    dummy_gravity = torch.randn(1, 1, 32, 32, device=device)

    density_pred = model(dummy_input)
    gravity_pred = forward_op(density_pred)

    physics_loss = F.mse_loss(gravity_pred, dummy_gravity)

    physics_loss.backward()

    has_grad = False
    grad_info = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                has_grad = True
                grad_info.append((name, grad_norm))

    if has_grad:
        print("梯度流验证通过！物理损失可以正确反向传播。")
        print(f"\n前 5 层梯度范数:")
        for name, norm in grad_info[:5]:
            print(f"  {name}: {norm:.6f}")
    else:
        print(" 梯度流验证失败！物理损失无法反向传播。")
        print(" 请检查 DifferentiableForward 是否使用了 numpy 或 detach 操作。")

    print("=" * 60)
    return has_grad


class V4Metrics:

    @staticmethod
    def deep_anomaly_iou(pred, target, depth_threshold=8, density_threshold=0.3):
        """深层异常体 IoU"""
        deep_pred = pred[:, :, depth_threshold:, :, :]
        deep_target = target[:, :, depth_threshold:, :, :]

        pred_binary = (torch.abs(deep_pred) > density_threshold).float()
        target_binary = (torch.abs(deep_target) > density_threshold).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        return (intersection / (union + 1e-6)).item()

    @staticmethod
    def depth_wise_mse(pred, target):
        """逐深度 MSE"""
        D = pred.shape[2]
        mse_per_depth = []
        for d in range(D):
            mse = F.mse_loss(pred[:, :, d, :, :], target[:, :, d, :, :])
            mse_per_depth.append(mse.item())
        return mse_per_depth

    @staticmethod
    def relative_error(pred, target):
        """相对误差"""
        return (torch.abs(pred - target) / (torch.abs(target) + 1e-6)).mean().item()


class HighResVisualizer:
    def __init__(self, save_dir: str, dpi: int = 300, upsampling: int = 4):
        self.save_dir = os.path.join(save_dir, "vis_highres")
        os.makedirs(self.save_dir, exist_ok=True)
        self.dpi = dpi
        self.upsampling = upsampling

    def _upsample(self, data: np.ndarray) -> np.ndarray:
        """上采样以获得更平滑的可视化"""
        from scipy.ndimage import zoom
        return zoom(data, self.upsampling, order=1)

    def save_epoch_result(self, epoch: int,
                          obs_gravity: np.ndarray,
                          pred_gravity: np.ndarray,
                          true_density: np.ndarray,
                          pred_density: np.ndarray,
                          dx: float = 100.0,
                          dz: float = 100.0):
        fig = plt.figure(figsize=(16, 14))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1.2, 1.5],
                      hspace=0.25, wspace=0.3)

        fig.suptitle(f"High-Res 3D Inversion Result - Epoch {epoch}",
                     fontsize=14, fontweight='bold')

        nz, ny, nx = true_density.shape
        extent_xy = [0, nx * dx / 1000, ny * dx / 1000, 0]
        depth_km = nz * dz / 1000

        obs_up = self._upsample(obs_gravity)
        pred_up = self._upsample(pred_gravity)
        residual = obs_gravity - pred_gravity
        res_up = self._upsample(residual)

        extent_up = [0, nx * dx / 1000, ny * dx / 1000, 0]

        ax1 = fig.add_subplot(gs[0, 0])
        vmax = max(abs(obs_up.max()), abs(obs_up.min()))
        im1 = ax1.imshow(obs_up, cmap='jet', extent=extent_up, vmin=-vmax, vmax=vmax)
        ax1.set_title("Observed Gravity (Upsampled)", fontsize=10)
        ax1.set_xlabel("X (km)")
        ax1.set_ylabel("Y (km)")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(pred_up, cmap='jet', extent=extent_up, vmin=-vmax, vmax=vmax)
        ax2.set_title("Predicted Gravity (Upsampled)", fontsize=10)
        ax2.set_xlabel("X (km)")
        ax2.set_ylabel("Y (km)")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        res_max = max(abs(res_up.max()), abs(res_up.min()))
        im3 = ax3.imshow(res_up, cmap='RdBu_r', extent=extent_up, vmin=-res_max, vmax=res_max)
        ax3.set_title("Raw Residuals", fontsize=10)
        ax3.set_xlabel("X (km)")
        ax3.set_ylabel("Y (km)")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        self._plot_3d_slices(ax4, true_density, dx, dz, "True Model: Smooth Slices")

        ax5 = fig.add_subplot(gs[1, 1], projection='3d')
        self._plot_3d_slices(ax5, pred_density, dx, dz, "Pred Model: Smooth Slices")

        ax_info = fig.add_subplot(gs[1, 2])
        ax_info.axis('off')
        info_text = (
            f"Info:\n"
            f"Enhanced Visualization\n"
            f"Upsampling: {self.upsampling}x\n"
            f"DPI: {self.dpi}\n"
            f"Depth: {depth_km:.2f}km\n"
            f"\n"
            f"Grid: {nz}×{ny}×{nx}\n"
            f"dx: {dx}m, dz: {dz}m\n"
            f"\n"
            f"Gravity Range:\n"
            f"  Obs: [{obs_gravity.min():.2f}, {obs_gravity.max():.2f}]\n"
            f"  Pred: [{pred_gravity.min():.2f}, {pred_gravity.max():.2f}]\n"
            f"  RMSE: {np.sqrt(np.mean(residual ** 2)):.4f}"
        )
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax6 = fig.add_subplot(gs[2, 0], projection='3d')
        self._plot_voxel_body(ax6, true_density, dx, dz, "True Model: Voxel Body\n(Solid > 30%)")

        ax7 = fig.add_subplot(gs[2, 1], projection='3d')
        self._plot_voxel_body(ax7, pred_density, dx, dz, "Pred Model: Voxel Body\n(Solid > 30%)")

        ax8 = fig.add_subplot(gs[2, 2])
        error = np.abs(pred_density - true_density)
        depth_error = error.mean(axis=(1, 2))
        depths = np.arange(nz) * dz / 1000
        ax8.barh(depths, depth_error, height=dz / 1000 * 0.8, color='coral', edgecolor='darkred')
        ax8.set_xlabel("Mean Absolute Error", fontsize=10)
        ax8.set_ylabel("Depth (km)", fontsize=10)
        ax8.set_title("Error by Depth", fontsize=10)
        ax8.invert_yaxis()
        ax8.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_dir, f"epoch_{epoch:04d}_highres.png")
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Saved high-res visualization: {save_path}")

        return save_path

    def _plot_3d_slices(self, ax, density: np.ndarray, dx: float, dz: float, title: str):
        """绘制 3D 平滑切片"""
        nz, ny, nx = density.shape

        x = np.linspace(0, nx * dx / 1000, nx)
        y = np.linspace(0, ny * dx / 1000, ny)
        X, Y = np.meshgrid(x, y)

        mid_z = nz // 2
        Z_slice = np.ones_like(X) * mid_z * dz / 1000

        slice_data = density[mid_z, :, :]

        norm = plt.Normalize(vmin=-1, vmax=1)
        colors = plt.cm.viridis(norm(slice_data))

        ax.plot_surface(X, Y, Z_slice, facecolors=colors, alpha=0.9, shade=True)

        y_proj = np.linspace(0, ny * dx / 1000, ny)
        z_proj = np.linspace(0, nz * dz / 1000, nz)
        Y_proj, Z_proj = np.meshgrid(y_proj, z_proj)
        X_proj = np.zeros_like(Y_proj)
        yz_slice = density[:, :, nx // 2]
        colors_yz = plt.cm.viridis(norm(yz_slice))
        ax.plot_surface(X_proj, Y_proj, Z_proj, facecolors=colors_yz, alpha=0.6)

        x_proj = np.linspace(0, nx * dx / 1000, nx)
        X_proj2, Z_proj2 = np.meshgrid(x_proj, z_proj)
        Y_proj2 = np.ones_like(X_proj2) * ny * dx / 1000
        xz_slice = density[:, ny // 2, :]
        colors_xz = plt.cm.viridis(norm(xz_slice))
        ax.plot_surface(X_proj2, Y_proj2, Z_proj2, facecolors=colors_xz, alpha=0.6)

        ax.set_xlabel("X (km)", fontsize=8)
        ax.set_ylabel("Y (km)", fontsize=8)
        ax.set_zlabel("Depth (km)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.invert_zaxis()
        ax.view_init(elev=25, azim=-60)

    def _plot_voxel_body(self, ax, density: np.ndarray, dx: float, dz: float,
                         title: str, threshold: float = 0.3):
        """绘制 3D 体素体"""
        nz, ny, nx = density.shape

        voxels = np.abs(density) > threshold

        if not voxels.any():
            ax.text(0.5, 0.5, 0.5, "No voxels above threshold",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            return

        colors = np.empty(voxels.shape + (4,))
        colors[..., :3] = plt.cm.YlGn(density / density.max())[..., :3]
        colors[density < 0, :3] = plt.cm.RdPu(-density / density.min())[density < 0, :3]
        colors[..., 3] = 0.8 * voxels.astype(float)

        ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.1)

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.view_init(elev=25, azim=-60)


class V4Trainer:
    """V4 训练器"""

    def __init__(self, config: V4Config):
        self.config = config
        self.device = config.device

        os.makedirs(config.save_dir, exist_ok=True)

        self.train_set = V4Dataset(config, config.batch_size * config.steps_per_epoch, 'train')
        self.val_set = V4Dataset(config, config.batch_size * 50, 'val')

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        self.model = PhysicsInformedUNet(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        if config.resume_path and os.path.exists(config.resume_path):
            print(f"\n[微调模式] 正在加载预训练权重: {config.resume_path}")
            checkpoint = torch.load(config.resume_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  - 已从 checkpoint 中提取 'model_state_dict'")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"  ✅ 权重加载成功!")
            print(f"  - 缺失键: {len(missing)}, 多余键: {len(unexpected)}")
            if len(missing) > 0:
                print(f"  - 缺失键 (前5个): {missing[:5]}")
            print()

        self.forward_op = DifferentiableForward(
            config.grid_shape,
            config.dx,
            config.dz
        ).to(self.device)

        self.criterion = ComprehensiveLoss(config, self.forward_op)

        self.discriminator = Discriminator3D(in_channels=1).to(self.device)
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=1e-4)

        self.optimizer_D = optim.AdamW(self.discriminator.parameters(), lr=config.lr * 0.5, weight_decay=1e-4)

        # 使用余弦退火调度器替代固定学习率
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )

        self.scaler = GradScaler('cuda') if 'cuda' in self.device else None

        self.best_loss = float('inf')
        self.history_best_loss = float('inf')
        self.history_file = os.path.join(config.save_dir, 'history_best.json')
        self._load_history_best()

        self.metrics = V4Metrics()

        self.visualizer = HighResVisualizer(config.save_dir, dpi=300, upsampling=4)

        self.train_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'deep_iou': [],
            'learning_rate': [],
            'loss_components': {
                'depth': [], 'focus': [], 'gdl': [], 'edge': [],
                'physics': [], 'l1': [], 'morph': [], 'adv': [], 'boundary': []
            }
        }
        
        self.history_json_path = os.path.join(config.save_dir, 'training_history.json')
        self.start_epoch = 1
        if config.resume_path and os.path.exists(self.history_json_path):
            self._load_train_history()
    
    def _load_train_history(self):
        """加载之前的训练历史（微调追加模式）"""
        import json
        try:
            with open(self.history_json_path, 'r') as f:
                prev_history = json.load(f)
            
            for key in ['epochs', 'train_loss', 'val_loss', 'deep_iou', 'learning_rate']:
                if key in prev_history:
                    self.train_history[key] = prev_history[key]
            
            if 'loss_components' in prev_history:
                for comp_key in self.train_history['loss_components']:
                    if comp_key in prev_history['loss_components']:
                        self.train_history['loss_components'][comp_key] = prev_history['loss_components'][comp_key]
            
            if len(self.train_history['epochs']) > 0:
                self.start_epoch = max(self.train_history['epochs']) + 1
                print(f"[微调模式] 已加载之前的训练历史 ({len(self.train_history['epochs'])} epochs)")
                print(f"           将从 Epoch {self.start_epoch} 继续记录")
        except Exception as e:
            print(f"[警告] 加载训练历史失败: {e}，将从头开始记录")

    def _load_history_best(self):
        """加载历史最佳记录"""
        import json
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                self.history_best_loss = data.get('best_loss', float('inf'))
                print(f"[历史记录] 加载历史最佳: val_loss = {self.history_best_loss:.6f}")
                print(f"           来自: {data.get('timestamp', 'unknown')}")
            except Exception as e:
                print(f"[警告] 加载历史记录失败: {e}")
                self.history_best_loss = float('inf')
        else:
            print(f"[历史记录] 无历史记录，将创建新记录")

    def _save_history_best(self, epoch, val_loss, deep_iou):
        """保存历史最佳记录"""
        import json
        from datetime import datetime
        data = {
            'best_loss': val_loss,
            'deep_iou': deep_iou,
            'epoch': epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'lr': self.config.lr,
                'w_depth': self.config.w_depth,
                'w_physics': self.config.w_physics,
            }
        }
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[历史记录] 已更新历史最佳: val_loss = {val_loss:.6f}")

    @torch.no_grad()
    def visualize_sample(self, epoch: int):
        """生成一个验证样本的可视化"""
        self.model.eval()

        inputs, targets, obs_gravity = next(iter(self.val_loader))
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)
        pred_gravity = self.forward_op(outputs)

        obs_grav_np = obs_gravity[0, 0].cpu().numpy()
        pred_grav_np = pred_gravity[0, 0].cpu().numpy()
        true_den_np = targets[0, 0].cpu().numpy()
        pred_den_np = outputs[0, 0].cpu().numpy()

        self.visualizer.save_epoch_result(
            epoch, obs_grav_np, pred_grav_np,
            true_den_np, pred_den_np,
            self.config.dx, self.config.dz
        )

    def train_epoch(self, epoch):
        self.model.train()
        self.discriminator.train()
        total_loss = 0
        total_loss_D = 0
        loss_components = {'depth': 0, 'focus': 0, 'gdl': 0, 'edge': 0, 'physics': 0, 'l1': 0, 'morph': 0, 'adv': 0,
                           'boundary': 0}

        for batch_idx, (inputs, targets, obs_gravity) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            obs_gravity = obs_gravity.to(self.device)

            loss_D = torch.tensor(0.0)
            if batch_idx % 2 == 0:
                self.optimizer_D.zero_grad()

                with torch.no_grad():
                    fake_density = self.model(inputs)

                real_score = self.discriminator(targets)
                fake_score = self.discriminator(fake_density.detach())

                loss_D_real = F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
                loss_D_fake = F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
                loss_D = (loss_D_real + loss_D_fake) * 0.5

                loss_D.backward()
                self.optimizer_D.step()

            total_loss_D += loss_D.item()

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss, losses = self.criterion(outputs, targets, obs_gravity)

                    fake_score_G = self.discriminator(outputs)
                    loss_adv = F.binary_cross_entropy_with_logits(fake_score_G, torch.ones_like(fake_score_G))
                    loss = loss + loss_adv * self.config.w_adv
                    losses['adv'] = loss_adv

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss, losses = self.criterion(outputs, targets, obs_gravity)

                fake_score_G = self.discriminator(outputs)
                loss_adv = F.binary_cross_entropy_with_logits(fake_score_G, torch.ones_like(fake_score_G))
                loss = loss + loss_adv * self.config.w_adv
                losses['adv'] = loss_adv

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            for k, v in losses.items():
                if k in loss_components:
                    loss_components[k] += v.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss_G: {loss.item():.4f} Loss_D: {loss_D.item():.4f} "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / len(self.train_loader)
        for k in loss_components:
            loss_components[k] /= len(self.train_loader)

        return avg_loss, loss_components

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        deep_ious = []

        for inputs, targets, obs_gravity in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            obs_gravity = obs_gravity.to(self.device)

            outputs = self.model(inputs)
            loss, _ = self.criterion(outputs, targets, obs_gravity)
            total_loss += loss.item()

            deep_iou = self.metrics.deep_anomaly_iou(outputs, targets)
            deep_ious.append(deep_iou)

        return total_loss / len(self.val_loader), np.mean(deep_ious)

    def save_checkpoint(self, epoch, loss, deep_iou, is_best=False, is_history_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'deep_iou': deep_iou,
            'config': self.config
        }

        torch.save(state, os.path.join(self.config.save_dir, 'latest.pth'))

        if is_history_best:
            torch.save(state, os.path.join(self.config.save_dir, 'best_model.pth'))
            self._save_history_best(epoch, loss, deep_iou)
            print(f" 新的历史最佳模型! val_loss: {loss:.6f} (超越历史 {self.history_best_loss:.6f}) ")

    def plot_learning_curves(self):
        """生成学习曲线图"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
        })
        
        epochs = self.train_history['epochs']
        train_loss = self.train_history['train_loss']
        val_loss = self.train_history['val_loss']
        deep_iou = self.train_history['deep_iou']
        learning_rate = self.train_history['learning_rate']
        loss_components = self.train_history['loss_components']
        
        if len(epochs) == 0:
            print("[Warning] 无训练历史数据，跳过曲线图生成")
            return
        
        fig1, axes1 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        fig1.subplots_adjust(hspace=0.08, left=0.12, right=0.95, top=0.92, bottom=0.10)
        
        ax1 = axes1[0]
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1.5)
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=1.5)
        
        best_epoch = epochs[np.argmin(val_loss)]
        best_val = min(val_loss)
        ax1.scatter([best_epoch], [best_val], c='green', s=80, zorder=5, 
                   marker='*', label=f'Best Val ({best_val:.4f})')
        ax1.axhline(y=best_val, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right', framealpha=0.9)
        ax1.set_title('(a) Training and Validation Loss', loc='left', fontweight='bold')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax2 = axes1[1]
        ax2.plot(epochs, deep_iou, 'g-', linewidth=1.5)
        ax2.fill_between(epochs, 0, deep_iou, alpha=0.2, color='green')
        
        best_iou_epoch = epochs[np.argmax(deep_iou)]
        best_iou = max(deep_iou)
        ax2.scatter([best_iou_epoch], [best_iou], c='darkgreen', s=80, zorder=5, 
                   marker='*', label=f'Best IoU ({best_iou:.4f})')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Deep Anomaly IoU')
        ax2.set_ylim(0, max(1.0, max(deep_iou) * 1.1))
        ax2.legend(loc='lower right', framealpha=0.9)
        ax2.set_title('(b) Deep Anomaly Intersection over Union', loc='left', fontweight='bold')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        fig1.suptitle('Training Progress Overview', fontsize=14, fontweight='bold', y=0.98)
        
        save_path1 = os.path.join(self.config.save_dir, 'learning_curves_overview.png')
        fig1.savefig(save_path1, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close(fig1)
        print(f"  [1/3] 保存: {save_path1}")
        
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6))
        fig2.subplots_adjust(hspace=0.35, wspace=0.28, left=0.08, right=0.96, 
                            top=0.88, bottom=0.10)
        
        component_info = {
            'depth': ('#1f77b4', 'Depth-weighted MSE', '(a)'),
            'focus': ('#ff7f0e', 'Focus Loss', '(b)'),
            'gdl': ('#2ca02c', 'Gradient Diff Loss', '(c)'),
            'physics': ('#d62728', 'Physics Consistency', '(d)'),
            'edge': ('#9467bd', 'Edge Enhancement', '(e)'),
            'boundary': ('#8c564b', 'Boundary-aware', '(f)'),
        }
        
        for idx, (key, (color, label, subplot_label)) in enumerate(component_info.items()):
            ax = axes2.flat[idx]
            if key in loss_components and len(loss_components[key]) > 0:
                values = loss_components[key]
                ax.plot(epochs[:len(values)], values, color=color, linewidth=1.5)
                ax.fill_between(epochs[:len(values)], 0, values, alpha=0.15, color=color)
                
                if len(values) > 0:
                    final_val = values[-1]
                    ax.annotate(f'{final_val:.4f}', 
                               xy=(epochs[-1], final_val),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color=color)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{subplot_label} {label}', loc='left', fontsize=10, fontweight='bold')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        fig2.suptitle('Individual Loss Components During Training', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        save_path2 = os.path.join(self.config.save_dir, 'learning_curves_components.png')
        fig2.savefig(save_path2, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig2)
        print(f"  [2/3] 保存: {save_path2}")
        
        fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
        fig3.subplots_adjust(wspace=0.25, left=0.10, right=0.95, top=0.88, bottom=0.15)
        
        ax3_1 = axes3[0]
        ax3_1.semilogy(epochs, learning_rate, 'b-', linewidth=1.5)
        ax3_1.set_xlabel('Epoch')
        ax3_1.set_ylabel('Learning Rate (log scale)')
        ax3_1.set_title('(a) Learning Rate Schedule', loc='left', fontweight='bold')
        ax3_1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax3_1.axhline(y=learning_rate[0], color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax3_1.text(epochs[0], learning_rate[0] * 1.2, f'Initial: {learning_rate[0]:.2e}', 
                  fontsize=8, color='gray')
        ax3_1.text(epochs[-1], learning_rate[-1] * 0.8, f'Final: {learning_rate[-1]:.2e}', 
                  fontsize=8, color='gray', ha='right')
        
        ax3_2 = axes3[1]
        gap = [t - v for t, v in zip(train_loss, val_loss)]
        
        color1, color2 = '#1f77b4', '#d62728'
        ax3_2.plot(epochs, train_loss, color=color1, label='Train Loss', linewidth=1.5)
        ax3_2.plot(epochs, val_loss, color=color2, label='Val Loss', linewidth=1.5)
        ax3_2.fill_between(epochs, train_loss, val_loss, alpha=0.15, color='gray', 
                          label='Generalization Gap')
        
        ax3_2.set_xlabel('Epoch')
        ax3_2.set_ylabel('Loss')
        ax3_2.set_title('(b) Generalization Analysis', loc='left', fontweight='bold')
        ax3_2.legend(loc='upper right', framealpha=0.9)
        ax3_2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        if len(gap) > 10:
            mean_gap = np.mean(gap[len(gap)//2:])
            if mean_gap > 0.1:
                ax3_2.annotate('Potential Overfitting', 
                              xy=(epochs[-1], train_loss[-1]),
                              xytext=(-60, 20), textcoords='offset points',
                              fontsize=8, color='orange',
                              arrowprops=dict(arrowstyle='->', color='orange', lw=0.8))
        
        fig3.suptitle('Training Dynamics Analysis', fontsize=14, fontweight='bold', y=0.98)
        
        save_path3 = os.path.join(self.config.save_dir, 'learning_curves_analysis.png')
        fig3.savefig(save_path3, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig3)
        print(f"  [3/3] 保存: {save_path3}")
        
        import json
        history_path = os.path.join(self.config.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'epochs': epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'deep_iou': deep_iou,
                'learning_rate': learning_rate,
                'loss_components': {k: v for k, v in loss_components.items() if len(v) > 0},
                'best_epoch': int(best_epoch),
                'best_val_loss': float(best_val),
                'best_iou_epoch': int(best_iou_epoch),
                'best_deep_iou': float(best_iou),
            }, f, indent=2)
        print(f"  [JSON] 保存: {history_path}")
        
        print(f"\n{'='*60}")
        print(f"学习曲线图生成完成！")
        print(f"  - 总览图: {save_path1}")
        print(f"  - 分量图: {save_path2}")
        print(f"  - 分析图: {save_path3}")
        print(f"  - 历史数据: {history_path}")
        print(f"{'='*60}\n")

    def train(self):
        print(f"Starting V4 Training on {self.device}")
        print(f"Config: {self.config}")
        print(f"\n[历史最佳] 需要超越的目标: val_loss < {self.history_best_loss:.6f}")

        print("\n[Pre-training] 验证梯度流...")
        grad_ok = verify_gradient_flow(self.model, self.forward_op, self.device)
        if not grad_ok:
            print("警告：梯度流验证失败，物理损失可能无效！\n")

        loss_balancer = AdaptiveLossBalancer(
            ['depth', 'focus', 'gdl', 'physics'],
            target_scale=0.5,
            momentum=0.95
        )

        for epoch in range(1, self.config.epochs + 1):
            self.train_set.set_epoch(epoch)

            train_loss, loss_components = self.train_epoch(epoch)

            if epoch <= 5:
                loss_balancer.update(loss_components)
                scales = loss_balancer.get_scales()
                print(f"\n[Epoch {epoch}] 损失权重自适应:")
                for name, scale in scales.items():
                    print(f"  {name}: {scale:.4f}")

            val_loss, deep_iou = self.validate()

            current_lr = self.scheduler.get_last_lr()[0]
            record_epoch = self.start_epoch + epoch - 1
            self.train_history['epochs'].append(record_epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['deep_iou'].append(deep_iou)
            self.train_history['learning_rate'].append(current_lr)
            for k, v in loss_components.items():
                if k in self.train_history['loss_components']:
                    self.train_history['loss_components'][k].append(v)

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            is_history_best = val_loss < self.history_best_loss
            if is_history_best:
                self.history_best_loss = val_loss

            self.save_checkpoint(epoch, val_loss, deep_iou, is_best, is_history_best)

            if epoch % 10 == 0 or is_history_best:
                try:
                    self.visualize_sample(epoch)
                except Exception as e:
                    print(f"Visualization failed: {e}")

            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  - Depth: {loss_components['depth']:.4f}")
            print(f"  - Focus: {loss_components['focus']:.4f}")
            print(f"  - GDL: {loss_components['gdl']:.4f}")
            print(f"  - Physics: {loss_components['physics']:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Deep IoU: {deep_iou:.4f}")
            print(f"本次最佳: {self.best_loss:.4f} | 历史最佳: {self.history_best_loss:.4f}")
            if is_history_best:
                print(f" 突破历史记录")
            print(f"{'=' * 60}\n")

        print("\n[Training Complete] 学习曲线图...")
        self.plot_learning_curves()
        print("学习曲线图已保存!")


def main():
    parser = argparse.ArgumentParser(description='TransUNet V4 Training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--verify-only', action='store_true', help='仅运行梯度验证')
    args = parser.parse_args()

    config = V4Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_path=args.resume
    )

    if args.verify_only:
        model = PhysicsInformedUNet(config).to(config.device)
        forward_op = DifferentiableForward(
            config.grid_shape, config.dx, config.dz
        ).to(config.device)
        verify_gradient_flow(model, forward_op, config.device)
    else:
        trainer = V4Trainer(config)
        trainer.train()


if __name__ == '__main__':
    main()

