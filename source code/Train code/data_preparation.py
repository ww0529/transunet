import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from config import V4Config


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
            # 增加负密度概率从30%到50%
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
            val = np.random.uniform(0.5, 1.0) * (1 if np.random.rand() > 0.5 else -1)  # 50%负密度
            model[dist < r] = val
        return model

    def generate_trapezoid(self):
        """生成梯形体 - 模拟锐利边缘的地质体"""
        model = np.zeros(self.shape)

        # 随机参数
        z_start = np.random.randint(2, 6)
        z_end = np.random.randint(z_start + 4, self.nz - 2)
        base_size = np.random.randint(4, 10)
        taper_rate = np.random.uniform(0.2, 0.5)  # 梯形擬合率
        density = np.random.uniform(0.6, 1.0) * (1 if np.random.rand() > 0.5 else -1)  # 50%负密度

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

        # 随机参数
        num_steps = np.random.randint(4, 8)
        direction = np.random.choice(['x', 'y', 'diagonal'])
        density = np.random.uniform(0.6, 1.0) * (1 if np.random.rand() > 0.5 else -1)  # 50%负密度
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
            else:  # diagonal
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

        # 外层盒子
        outer_density = np.random.uniform(0.3, 0.6)
        outer_z = (3, self.nz - 3)
        outer_y = (6, self.ny - 6)
        outer_x = (6, self.nx - 6)
        model[outer_z[0]:outer_z[1], outer_y[0]:outer_y[1], outer_x[0]:outer_x[1]] = outer_density

        # 内层盒子 (密度不同)
        inner_density = np.random.uniform(0.7, 1.0) * (1 if np.random.rand() > 0.5 else -1)  # 50%负密度
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

        # 大块 (1个)
        z0, z1 = 4, 12
        y0, y1 = 8, 24
        x0, x1 = 8, 24
        model[z0:z1, y0:y1, x0:x1] = np.random.uniform(0.3, 0.5)

        # 中块 (2-3个)
        for _ in range(np.random.randint(2, 4)):
            size = np.random.randint(3, 6)
            cz = np.random.randint(2, self.nz - size)
            cy = np.random.randint(2, self.ny - size)
            cx = np.random.randint(2, self.nx - size)
            val = np.random.uniform(0.5, 0.8)
            model[cz:cz + size, cy:cy + size, cx:cx + size] = val

        # 小块 (3-5个) - 高密度，锐利边缘
        for _ in range(np.random.randint(3, 6)):
            size = 2
            cz = np.random.randint(2, self.nz - size)
            cy = np.random.randint(2, self.ny - size)
            cx = np.random.randint(2, self.nx - size)
            val = np.random.uniform(0.8, 1.0) * (1 if np.random.rand() > 0.5 else -1)  # 50%负密度
            model[cz:cz + size, cy:cy + size, cx:cx + size] = val

        return np.clip(model, -1, 1)

    def generate_fault(self):
        """断层结构生成器 - 斜切的地质断层"""
        model = np.zeros(self.shape)

        # 断层参数
        fault_angle = np.random.uniform(30, 60)  # 断层倾角 (度)
        fault_offset = np.random.randint(2, 5)  # 断层位移
        density_left = np.random.uniform(0.5, 0.8) * (1 if np.random.rand() > 0.5 else -1)
        density_right = np.random.uniform(0.3, 0.6) * (1 if np.random.rand() > 0.5 else -1)

        # 创建斜切面
        tan_angle = np.tan(np.radians(fault_angle))

        for z in range(self.nz):
            for y in range(self.ny):
                # 断层面位置
                fault_x = int(self.nx / 2 + (z - self.nz / 2) * tan_angle)

                # 左侧
                for x in range(max(0, fault_x - fault_offset)):
                    if 3 < z < self.nz - 2:
                        model[z, y, x] = density_left

                # 右侧 (位移)
                for x in range(min(self.nx, fault_x + fault_offset), self.nx):
                    if 3 + fault_offset < z < self.nz - 2:
                        model[z, y, x] = density_right

        return np.clip(model, -1, 1)

    def generate_perlin_terrain(self):
        """柏林噪声地形生成器 - 模拟自然地质变化"""
        from scipy.ndimage import gaussian_filter

        # 多尺度噪声叠加
        model = np.zeros(self.shape)

        # 大尺度变化
        noise_large = np.random.randn(self.nz // 4, self.ny // 4, self.nx // 4)
        noise_large = gaussian_filter(noise_large, sigma=1)
        from scipy.ndimage import zoom
        noise_large = zoom(noise_large, (4, 4, 4), order=1)[:self.nz, :self.ny, :self.nx]

        # 中尺度变化
        noise_mid = np.random.randn(self.nz // 2, self.ny // 2, self.nx // 2)
        noise_mid = gaussian_filter(noise_mid, sigma=0.5)
        noise_mid = zoom(noise_mid, (2, 2, 2), order=1)[:self.nz, :self.ny, :self.nx]

        # 小尺度细节
        noise_small = np.random.randn(self.nz, self.ny, self.nx)
        noise_small = gaussian_filter(noise_small, sigma=0.3)

        # 组合
        model = 0.5 * noise_large + 0.3 * noise_mid + 0.2 * noise_small

        # 阈值化：只保留高密度区域
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

            # 填充当前盒子
            model[z0:z1, y0:y1, x0:x1] = density

            # 随机决定是否递归
            if np.random.rand() < 0.7:
                # 在角落添加子盒子
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

        # 初始大盒子
        initial_density = np.random.uniform(0.4, 0.7)
        add_box(2, self.nz - 2, 4, self.ny - 4, 4, self.nx - 4, initial_density)

        return np.clip(model, -1, 1)

    def generate_figure8_staircase(self):
        """八字形阶梯生成器 - 两条交叉的阶梯结构"""
        model = np.zeros(self.shape)

        step_count = np.random.randint(4, 7)
        step_size = 2
        density = np.random.uniform(0.7, 1.0)

        # 第一条阶梯: 从左上到右下
        for i in range(step_count):
            x_center = 6 + i * 3
            z_center = 3 + i * 2
            y_center = 10 + np.random.randint(-2, 3)

            x_start, x_end = max(0, x_center - step_size // 2), min(self.nx, x_center + step_size // 2 + 1)
            y_start, y_end = max(0, y_center - step_size // 2), min(self.ny, y_center + step_size // 2 + 1)
            z_start, z_end = max(0, z_center - step_size // 2), min(self.nz, z_center + step_size // 2 + 1)

            model[z_start:z_end, y_start:y_end, x_start:x_end] = density

        # 第二条阶梯: 从右上到左下
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
        """分离双长方体生成器 - 训练模型识别两个独立密度体和中间空隙"""
        model = np.zeros(self.shape)

        # 随机选择分离方向
        direction = np.random.choice(['x', 'y', 'z'])

        # 两个长方体的密度：50%概率一正一负
        if np.random.rand() > 0.5:
            # 一正一负
            density1 = np.random.uniform(0.5, 1.0)
            density2 = -np.random.uniform(0.4, 0.8)
        else:
            # 同符号
            density1 = np.random.uniform(0.5, 1.0)
            density2 = np.random.uniform(0.5, 1.0)
            if np.random.rand() > 0.5:
                density1, density2 = -density1, -density2  # 50% 全负

        # 分离间隙大小
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

        else:  # direction == 'z'
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
        """空心盒子生成器 - 训练模型识别中空结构"""
        model = np.zeros(self.shape)

        shell_density = np.random.uniform(0.6, 1.0) * (1 if np.random.rand() > 0.5 else -1)  # 50%负密度
        # 壳厚度
        thickness = np.random.randint(2, 4)

        # 外壳边界
        z_out = (3, self.nz - 3)
        y_out = (5, self.ny - 5)
        x_out = (5, self.nx - 5)

        # 内部边界（空心部分）
        z_in = (z_out[0] + thickness, z_out[1] - thickness)
        y_in = (y_out[0] + thickness, y_out[1] - thickness)
        x_in = (x_out[0] + thickness, x_out[1] - thickness)

        # 先填充整个外壳
        model[z_out[0]:z_out[1], y_out[0]:y_out[1], x_out[0]:x_out[1]] = shell_density

        # 再将内部掏空
        if z_in[1] > z_in[0] and y_in[1] > y_in[0] and x_in[1] > x_in[0]:
            model[z_in[0]:z_in[1], y_in[0]:y_in[1], x_in[0]:x_in[1]] = 0.0

        return np.clip(model, -1, 1)

    def generate_layered_density(self):
        """层状密度生成器 - 不同密度的水平层，夹杂空隙"""
        model = np.zeros(self.shape)

        num_layers = np.random.randint(3, 6)
        layer_height = self.nz // num_layers

        for i in range(num_layers):
            z_start = i * layer_height + 1
            z_end = min((i + 1) * layer_height - 1, self.nz - 1)

            # 随机决定是否为空隙层
            if np.random.rand() > 0.3:  # 70% 概率有密度
                density = np.random.uniform(0.4, 1.0)
                y_start, y_end = 5, self.ny - 5
                x_start, x_end = 5, self.nx - 5
                model[z_start:z_end, y_start:y_end, x_start:x_end] = density
            # 否则保持为 0（空隙）

        return np.clip(model, -1, 1)

    def generate_scattered_anomalies(self):
        """分散多异常体生成器 - 模拟真实地质场景中多个分散的正负密度异常"""
        model = np.zeros(self.shape)

        # 生成 3-6 个分散的异常体
        n_bodies = np.random.randint(3, 7)

        # 确保正负密度都有
        n_positive = np.random.randint(1, n_bodies)
        n_negative = n_bodies - n_positive

        centers = []  # 记录中心避免重叠

        for i in range(n_bodies):
            # 随机大小（小中大）
            size_type = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
            if size_type == 'small':
                sz, sy, sx = np.random.randint(2, 4), np.random.randint(3, 5), np.random.randint(3, 5)
            elif size_type == 'medium':
                sz, sy, sx = np.random.randint(3, 5), np.random.randint(4, 7), np.random.randint(4, 7)
            else:
                sz, sy, sx = np.random.randint(4, 7), np.random.randint(6, 10), np.random.randint(6, 10)

            # 随机位置（避免边缘）
            for _ in range(20):  # 最多尝试20次找不重叠位置
                cz = np.random.randint(2, self.nz - sz - 1)
                cy = np.random.randint(4, self.ny - sy - 3)
                cx = np.random.randint(4, self.nx - sx - 3)

                # 检查是否与已有中心距离足够远
                overlap = False
                for (oz, oy, ox) in centers:
                    if abs(cz - oz) < 4 and abs(cy - oy) < 6 and abs(cx - ox) < 6:
                        overlap = True
                        break
                if not overlap:
                    break

            centers.append((cz, cy, cx))

            # 正负密度
            if i < n_positive:
                density = np.random.uniform(0.5, 1.0)
            else:
                density = -np.random.uniform(0.4, 0.9)

            # 填充
            z0, z1 = cz, min(cz + sz, self.nz)
            y0, y1 = cy, min(cy + sy, self.ny)
            x0, x1 = cx, min(cx + sx, self.nx)
            model[z0:z1, y0:y1, x0:x1] = density

        return np.clip(model, -1, 1)

    def generate_realistic_anomaly(self):
        """真实异常体生成器 - 模拟类似Gzz.txt的多峰复杂异常"""
        model = np.zeros(self.shape)

        # 主异常体（较大，正密度）
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

        # 次级异常体（1-3个，混合正负）
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

        # 深层弱异常（模拟背景变化）
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
            'scattered': (self.generate_scattered_anomalies, 0.12),  # 新增: 分散多异常体
            'realistic': (self.generate_realistic_anomaly, 0.10),  # 新增: 真实异常模拟
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


class FastGravityForward:
    """NumPy 版本的快速正演（用于数据生成）"""

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
            field[0] *= 1e5  # mGal
            field[1] *= 1e9  # Eötvös
            return field
        else:
            field_fft = np.sum(rho_fft * self._kernel_fft, axis=0)
            field_padded = np.fft.ifft2(field_fft).real
            field = field_padded[self.pad_y:-self.pad_y, self.pad_x:-self.pad_x]
            return field * (1e5 if self.mode == 'gz' else 1e9)


# ==========================================
# [数据层] Advanced Dataset with Augmentation
# ==========================================


class AdvancedAugmentation:
    """Advanced data augmentation."""

    @staticmethod
    def elastic_deformation(density, alpha=30, sigma=4):
        """Elastic deformation simulating geological folding."""
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
        """Depth shift augmentation."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return density
        return np.roll(density, shift, axis=0)

    @staticmethod
    def add_geological_noise(density, noise_level=0.03):
        """Geological noise injection."""
        noise = np.random.randn(*density.shape) * noise_level
        noise = ndimage.gaussian_filter(noise, sigma=1)
        return np.clip(density + noise, -1, 1)

    @staticmethod
    def random_flip(density):
        """Random flip augmentation."""
        if np.random.rand() > 0.5:
            density = np.flip(density, axis=2).copy()
        if np.random.rand() > 0.5:
            density = np.flip(density, axis=1).copy()
        return density

    @staticmethod
    def random_scale(density, scale_range=(0.8, 1.2)):
        """Random density value scaling."""
        scale = np.random.uniform(*scale_range)
        return np.clip(density * scale, -1, 1)


class V4Dataset(Dataset):
    """Dataset with support for advanced augmentation and curriculum learning."""

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
        """Set the current epoch for curriculum learning scheduling."""
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


def create_dataloaders(config: V4Config, train_epoch_len: int = 500, val_epoch_len: int = 100):
    """Create train and validation dataloaders."""
    train_dataset = V4Dataset(config, train_epoch_len, mode='train')
    val_dataset = V4Dataset(config, val_epoch_len, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset, val_dataset
