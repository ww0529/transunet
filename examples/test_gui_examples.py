"""
测试脚本：为三个例子生成可视化图片
使用 jgui.py 的逻辑来处理和可视化 VTI 模型
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import zoom

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入生成函数
from examples.generate_vti_models import (
    generate_figure8_staircase,
    generate_center_prism,
    generate_dual_prism,
    _save_as_vti
)


def create_synthetic_gravity(density_model, dx=100.0, dz=100.0):
    """
    从密度模型计算合成重力异常 (Gzz)
    使用简化的正演算子
    """
    nz, ny, nx = density_model.shape
    G = 6.674e-11
    rho = density_model / 1000.0  # kg/m³ -> g/cm³

    gzz_calc = np.zeros((ny, nx))
    for iz in range(nz):
        z = (iz + 0.5) * dz
        weight = 1.0 / (1.0 + (z / (3 * dz))**2)
        gzz_calc += rho[iz, :, :] * weight

    gzz_calc = gzz_calc * G * 1e9 * 100

    # 安全检查
    gzz_calc = np.nan_to_num(gzz_calc, nan=0.0, posinf=0.0, neginf=0.0)
    return gzz_calc


def visualize_model(density_model, title, output_path, dx=100.0, dz=100.0):
    """
    为单个模型生成完整的可视化图
    包括：重力异常、3D切片、体素体
    """
    nz, ny, nx = density_model.shape

    # 计算合成重力
    gzz = create_synthetic_gravity(density_model, dx, dz)

    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(f"{title}\nSynthetic Model Visualization", fontsize=16, fontweight='bold')

    # 坐标范围
    x_range = (0, nx * dx)
    y_range = (0, ny * dx)
    z_extent = nz * dz

    # ===== 第一行 =====
    # (0,0) 重力异常
    ax1 = fig.add_subplot(gs[0, 0])
    extent_xy = [0, nx * dx / 1000, ny * dx / 1000, 0]
    im1 = ax1.imshow(gzz, cmap='jet', extent=extent_xy)
    ax1.set_title("Synthetic Gravity Anomaly (Gzz)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("X (km)")
    ax1.set_ylabel("Y (km)")
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.ax.set_title("Eötvös", fontsize=9)

    # (0,1) 密度Z切片 (中间层)
    ax2 = fig.add_subplot(gs[0, 1])
    mid_z = nz // 2
    im2 = ax2.imshow(density_model[mid_z, :, :], cmap='jet', extent=extent_xy)
    ax2.set_title(f"Density Z-Slice (D={mid_z*dz:.0f}m)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("X (km)")
    ax2.set_ylabel("Y (km)")
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.ax.set_title("kg/m³", fontsize=9)

    # (0,2) 密度Y切片 (中间线)
    ax3 = fig.add_subplot(gs[0, 2])
    mid_y = ny // 2
    extent_xz = [0, nx * dx / 1000, nz * dz / 1000, 0]
    im3 = ax3.imshow(density_model[:, mid_y, :], cmap='jet', extent=extent_xz)
    ax3.set_title(f"Density Y-Slice (Y={mid_y*dx:.0f}m)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("X (km)")
    ax3.set_ylabel("Depth (km)")
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.ax.set_title("kg/m³", fontsize=9)

    # ===== 第二行 =====
    # (1,0) 3D切片视图
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    _plot_3d_slices(ax4, density_model, x_range, y_range, z_extent, dx, dz)
    ax4.set_title("3D Orthogonal Slices", fontsize=11, fontweight='bold')

    # (1,1) 3D体素视图
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    _plot_3d_voxels(ax5, density_model, x_range, y_range, z_extent, dx, dz)
    ax5.set_title("3D Voxel Body", fontsize=11, fontweight='bold')

    # (1,2) 统计信息
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    info_text = (
        f"Model Information:\n"
        f"{'='*30}\n"
        f"Grid Size: {nz}×{ny}×{nx}\n"
        f"Spacing: dx={dx}m, dz={dz}m\n"
        f"Total Depth: {nz*dz:.0f}m\n"
        f"Total Area: {nx*dx/1000:.1f}×{ny*dx/1000:.1f} km²\n\n"
        f"Density Range:\n"
        f"  Min: {density_model.min():.3f} kg/m³\n"
        f"  Max: {density_model.max():.3f} kg/m³\n"
        f"  Non-zero voxels: {np.count_nonzero(density_model)}\n\n"
        f"Gravity Anomaly:\n"
        f"  Min: {gzz.min():.2e} Eötvös\n"
        f"  Max: {gzz.max():.2e} Eötvös\n"
        f"  RMS: {np.sqrt(np.mean(gzz**2)):.2e} Eötvös"
    )

    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {output_path}")
    plt.close(fig)


def _plot_3d_slices(ax, volume, x_range, y_range, z_extent, dx, dz):
    """绘制3D正交切片"""
    nz, ny, nx = volume.shape

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    z = np.linspace(0, z_extent, nz)

    cx, cy, cz = nx // 2, ny // 2, nz // 2

    from matplotlib import cm
    vmin, vmax = volume.min(), volume.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Z切片 (水平)
    XX, YY = np.meshgrid(x, y)
    ZZ = np.full_like(XX, z[cz])
    colors_z = cm.jet(norm(volume[cz, :, :]))
    ax.plot_surface(XX, YY, ZZ, facecolors=colors_z, shade=False, alpha=0.8)

    # Y切片 (垂直)
    XX_y, ZZ_y = np.meshgrid(x, z)
    YY_y = np.full_like(XX_y, y[cy])
    colors_y = cm.jet(norm(volume[:, cy, :]))
    ax.plot_surface(XX_y, YY_y, ZZ_y, facecolors=colors_y, shade=False, alpha=0.8)

    # X切片 (垂直)
    YY_x, ZZ_x = np.meshgrid(y, z)
    XX_x = np.full_like(YY_x, x[cx])
    colors_x = cm.jet(norm(volume[:, :, cx]))
    ax.plot_surface(XX_x, YY_x, ZZ_x, facecolors=colors_x, shade=False, alpha=0.8)

    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Depth (m)', fontsize=9)
    ax.invert_zaxis()
    ax.view_init(elev=25, azim=-45)


def _plot_3d_voxels(ax, volume, x_range, y_range, z_extent, dx, dz):
    """绘制3D体素体"""
    nz, ny, nx = volume.shape

    # 使用中位数作为背景
    center = np.median(volume)
    anomaly_range = max(abs(volume.max() - center), abs(volume.min() - center))

    if anomaly_range < 0.01:
        ax.text(0.5, 0.5, 0.5, "No significant anomaly",
                ha='center', va='center', transform=ax.transAxes)
        return

    # 阈值
    threshold = 0.3
    pos_thresh = center + threshold * anomaly_range
    neg_thresh = center - threshold * anomaly_range
    voxels = (volume > pos_thresh) | (volume < neg_thresh)

    if not voxels.any():
        ax.text(0.5, 0.5, 0.5, "No voxels above threshold",
                ha='center', va='center', transform=ax.transAxes)
        return

    # 创建颜色
    from matplotlib import cm
    colors = np.empty(voxels.shape + (4,))
    dmax = volume.max() if volume.max() > 0 else 1.0
    dmin = volume.min() if volume.min() < 0 else -1.0

    norm_pos = np.clip(volume / dmax, 0, 1)
    colors[..., :3] = cm.YlGn(0.3 + 0.7 * norm_pos)[..., :3]

    if (volume < 0).any():
        norm_neg = np.clip(-volume / (-dmin), 0, 1)
        neg_mask = volume < 0
        colors[neg_mask, :3] = cm.RdPu(0.3 + 0.7 * norm_neg[neg_mask])[:, :3]

    colors[..., 3] = 0.8 * voxels.astype(float)

    # 创建坐标网格
    dx_grid = (x_range[1] - x_range[0]) / nx
    dy_grid = (y_range[1] - y_range[0]) / ny
    dz_grid = z_extent / nz

    x = np.linspace(x_range[0], x_range[1], nx + 1)
    y = np.linspace(y_range[0], y_range[1], ny + 1)
    z = np.linspace(0, z_extent, nz + 1)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # 转置以匹配voxels格式
    filled_t = np.transpose(voxels, (2, 1, 0))
    colors_t = np.transpose(colors, (2, 1, 0, 3))

    ax.voxels(X, Y, Z, filled_t, facecolors=colors_t, edgecolor='none', alpha=0.8)

    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Depth (m)', fontsize=9)
    ax.invert_zaxis()
    ax.view_init(elev=25, azim=-45)


def main():
    print("=" * 60)
    print("Test GUI Examples - Generate Visualizations")
    print("=" * 60)

    # 输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # 三个例子
    examples = [
        ("Complex Terrain", generate_figure8_staircase, "complex_terrain.png"),
        ("Center Prism", generate_center_prism, "center_prism.png"),
        ("Dual Prism", generate_dual_prism, "dual_prism.png"),
    ]

    for i, (name, gen_func, output_file) in enumerate(examples, 1):
        print(f"\n[{i}/3] Generating {name}...")

        # 生成模型
        model = gen_func()
        print(f"  Model shape: {model.shape}")
        print(f"  Value range: [{model.min():.3f}, {model.max():.3f}]")
        print(f"  Non-zero voxels: {np.count_nonzero(model)}")

        # 生成可视化
        output_path = os.path.join(output_dir, output_file)
        visualize_model(model, name, output_path)

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
