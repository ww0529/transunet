from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import zoom

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Run this script with the same Python environment used by jgui.py."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_CODE_DIR = PROJECT_ROOT / "source code"
if str(SOURCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE_DIR))

from config import V4Config
from data_preparation import FastGravityForward
from train_code import PhysicsInformedUNet


matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    }
)


@dataclass
class LoadedData:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    dx: float
    dy: float
    dz: float
    gzz_32: np.ndarray
    gz_raw: Optional[np.ndarray] = None
    gzz_raw: Optional[np.ndarray] = None
    density_raw: Optional[np.ndarray] = None
    density_is_normalized: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone prediction with the same data flow and 3D "
            "rendering as jgui.py."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "best_model.pth",
        help="Model checkpoint.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "examples",
        help="Directory containing synthetic VTI examples.",
    )
    parser.add_argument(
        "--field-file",
        type=Path,
        default=PROJECT_ROOT / "Field data example" / "Gzz.txt",
        help="Field Gzz text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "test_code",
        help="Directory for prediction artifacts.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--skip-field-data",
        action="store_true",
        help="Skip Field data example/Gzz.txt.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def build_config_from_checkpoint(raw_config: Any) -> V4Config:
    config = V4Config()
    if raw_config is None:
        return config
    if isinstance(raw_config, V4Config):
        return raw_config
    if isinstance(raw_config, dict):
        source = raw_config
    elif hasattr(raw_config, "__dict__"):
        source = {
            key: value
            for key, value in vars(raw_config).items()
            if not key.startswith("_")
        }
    else:
        return config

    valid = {key: value for key, value in source.items() if hasattr(config, key)}
    try:
        return V4Config(**valid)
    except Exception:
        for key, value in valid.items():
            setattr(config, key, value)
        return config


def load_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[torch.nn.Module, V4Config]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = build_config_from_checkpoint(raw_config)
    config.device = str(device)
    config.data_mode = str(config.data_mode).lower()
    config.grid_shape = tuple(int(value) for value in config.grid_shape)

    model = PhysicsInformedUNet(config).to(device)
    missing, unexpected = model.load_state_dict(
        extract_state_dict(checkpoint), strict=False
    )
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        f"  data_mode={config.data_mode}, grid_shape={config.grid_shape}, "
        f"device={device}"
    )
    print(f"  missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    return model, config


def load_vti(path: Path) -> LoadedData:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError("VTK is required to read VTI files.") from exc

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image_data = reader.GetOutput()

    dimensions = image_data.GetDimensions()
    spacing = image_data.GetSpacing()
    origin = image_data.GetOrigin()
    nx, ny, nz = dimensions

    point_data = image_data.GetPointData()
    cell_data = image_data.GetCellData()
    data_flat = None
    if point_data.GetScalars() is not None:
        data_flat = vtk_to_numpy(point_data.GetScalars())
    elif point_data.GetNumberOfArrays() > 0:
        data_flat = vtk_to_numpy(point_data.GetArray(0))
    elif cell_data.GetScalars() is not None:
        data_flat = vtk_to_numpy(cell_data.GetScalars())
        nx, ny, nz = max(1, nx - 1), max(1, ny - 1), max(1, nz - 1)
    elif cell_data.GetNumberOfArrays() > 0:
        data_flat = vtk_to_numpy(cell_data.GetArray(0))
        nx, ny, nz = max(1, nx - 1), max(1, ny - 1), max(1, nz - 1)

    if data_flat is None:
        raise ValueError(f"No scalar array found in {path}")

    try:
        density = np.asarray(data_flat).reshape((nz, ny, nx), order="F")
    except ValueError:
        density = np.asarray(data_flat).reshape((nz, ny, nx), order="C")

    normalized = nz > 1 and np.nanmax(np.abs(density)) <= 1.5
    model_density = density if normalized else density / 1000.0
    forward = FastGravityForward(
        (nz, ny, nx),
        dx=float(spacing[0]),
        dz=float(spacing[2]) if len(spacing) > 2 else 100.0,
        mode="joint",
    )
    gz_raw, gzz_raw = forward.forward(model_density)

    print(f"[VTI] {path.name}")
    print(f"  shape={(nz, ny, nx)}, normalized={normalized}")
    print(f"  Gzz range=[{gzz_raw.min():.6g}, {gzz_raw.max():.6g}]")

    return LoadedData(
        x_range=(
            float(origin[0]),
            float(origin[0] + spacing[0] * (nx - 1)),
        ),
        y_range=(
            float(origin[1]),
            float(origin[1] + spacing[1] * (ny - 1)),
        ),
        dx=float(spacing[0]),
        dy=float(spacing[1]),
        dz=float(spacing[2]) if len(spacing) > 2 else 100.0,
        gzz_32=gzz_raw,
        gz_raw=gz_raw,
        gzz_raw=gzz_raw,
        density_raw=np.asarray(density),
        density_is_normalized=normalized,
    )


def load_text_table(path: Path) -> np.ndarray:
    try:
        data = np.loadtxt(path, delimiter=",")
    except Exception:
        data = np.loadtxt(path)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid numeric table: {path}")
    return data


def load_gzz_text(path: Path) -> LoadedData:
    data = load_text_table(path)
    if data.shape[1] < 3:
        raise ValueError(f"Gzz text data must have at least 3 columns: {path}")

    x_values = data[:, 0]
    y_values = data[:, 1]
    gzz_values = (
        data[:, 3]
        if data.shape[1] >= 4 and len(np.unique(data[:, 2])) == 1
        else data[:, 2]
    )
    unique_x = np.unique(x_values)
    unique_y = np.unique(y_values)
    nx, ny = len(unique_x), len(unique_y)
    dx = float(unique_x[1] - unique_x[0]) if nx > 1 else 100.0
    dy = float(unique_y[1] - unique_y[0]) if ny > 1 else 100.0

    gzz_values = gzz_values.copy()
    if np.max(np.abs(gzz_values)) < 1e-4:
        gzz_values *= 1e9
    gzz_raw = gzz_values.reshape(ny, nx)
    gzz_32 = (
        zoom(gzz_raw, (32 / ny, 32 / nx), order=3)
        if (ny, nx) != (32, 32)
        else gzz_raw.copy()
    )

    print(f"[Gzz] {path.name}")
    print(
        f"  original_shape={(ny, nx)}, "
        f"range=[{gzz_raw.min():.6g}, {gzz_raw.max():.6g}] E"
    )
    return LoadedData(
        x_range=(float(unique_x.min()), float(unique_x.max())),
        y_range=(float(unique_y.min()), float(unique_y.max())),
        dx=dx,
        dy=dy,
        dz=100.0,
        gzz_32=gzz_32,
        gzz_raw=gzz_raw,
    )


def estimate_gz_from_gzz(gzz_e: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Same Fourier Gzz-to-Gz estimate used by DataManager in jgui.py."""
    gzz = np.asarray(gzz_e, dtype=float)
    ny, nx = gzz.shape
    kx = np.fft.fftfreq(nx, d=max(abs(float(dx)), 1e-6)) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=max(abs(float(dy)), 1e-6)) * 2.0 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    wavenumber = np.sqrt(kx_grid**2 + ky_grid**2)
    positive_k = wavenumber[wavenumber > 0.0]
    if positive_k.size == 0 or np.max(np.abs(gzz)) <= 1e-12:
        return np.zeros_like(gzz, dtype=float)

    gzz_si_fft = np.fft.fft2(gzz * 1e-9)
    k_floor = float(np.min(positive_k)) * 0.25
    inverse_filter = np.zeros_like(wavenumber)
    nonzero = wavenumber > 0.0
    inverse_filter[nonzero] = wavenumber[nonzero] / (
        wavenumber[nonzero] ** 2 + k_floor**2
    )
    gz_si_fft = -gzz_si_fft * inverse_filter
    gz_si_fft[0, 0] = 0.0
    return (np.fft.ifft2(gz_si_fft).real * 1e5).astype(float, copy=False)


def expected_density_polarity(
    rho_min: float, rho_max: float
) -> Optional[float]:
    if rho_min > rho_max:
        rho_min, rho_max = rho_max, rho_min
    if rho_min >= 0.0 and rho_max > 0.0:
        return 1.0
    if rho_max <= 0.0 and rho_min < 0.0:
        return -1.0
    if rho_max > abs(rho_min):
        return 1.0
    if abs(rho_min) > rho_max:
        return -1.0
    return None


def align_gzz_for_expected_density(
    gzz: np.ndarray, expected_polarity: Optional[float]
) -> np.ndarray:
    if expected_polarity is None or expected_polarity == 0:
        return gzz
    dominant = float(gzz.ravel()[np.argmax(np.abs(gzz))])
    if abs(dominant) <= 1e-12:
        return gzz
    return -gzz if np.sign(dominant) == np.sign(expected_polarity) else gzz


def build_input_tensor(
    data: LoadedData,
    config: V4Config,
    rho_min: float = -100.0,
    rho_max: float = 350.0,
) -> torch.Tensor:
    nz, target_ny, target_nx = (int(value) for value in config.grid_shape)

    if data.density_raw is not None:
        raw = np.asarray(data.density_raw, dtype=float)
        model_density = raw if data.density_is_normalized else raw / 1000.0
        tnz, tny, tnx = model_density.shape
        forward = FastGravityForward(
            (tnz, tny, tnx), dx=data.dx, dz=data.dz, mode="joint"
        )
        gz_raw, gzz_raw = forward.forward(model_density)
        gz_map = zoom(
            gz_raw,
            (target_ny / gz_raw.shape[0], target_nx / gz_raw.shape[1]),
            order=1,
        )
        gzz_map = zoom(
            gzz_raw,
            (target_ny / gzz_raw.shape[0], target_nx / gzz_raw.shape[1]),
            order=1,
        )
    else:
        gzz_map = np.asarray(data.gzz_32, dtype=float)
        gzz_map = align_gzz_for_expected_density(
            gzz_map, expected_density_polarity(rho_min, rho_max)
        )
        gz_map = estimate_gz_from_gzz(gzz_map, data.dx, data.dy)

    gzz_norm = gzz_map / (np.abs(gzz_map).max() + 1e-8)
    gz_norm = gz_map / (np.abs(gz_map).max() + 1e-8)
    z_indices = np.linspace(0, 1, nz)
    z_map = np.tile(z_indices[:, None, None], (1, target_ny, target_nx))
    z_tensor = torch.from_numpy(z_map).float()

    if str(config.data_mode).lower() == "joint":
        gz_tensor = (
            torch.from_numpy(gz_norm)
            .float()
            .unsqueeze(0)
            .expand(nz, -1, -1)
        )
        gzz_tensor = (
            torch.from_numpy(gzz_norm)
            .float()
            .unsqueeze(0)
            .expand(nz, -1, -1)
        )
        input_volume = torch.stack([gz_tensor, gzz_tensor, z_tensor], dim=0)
    else:
        primary = gz_norm if str(config.data_mode).lower() == "gz" else gzz_norm
        primary_tensor = (
            torch.from_numpy(primary)
            .float()
            .unsqueeze(0)
            .expand(nz, -1, -1)
        )
        input_volume = torch.stack([primary_tensor, z_tensor], dim=0)

    return input_volume.unsqueeze(0)


def normalize_to_real(
    pred_norm: np.ndarray,
    rho_min: float = -100.0,
    rho_max: float = 350.0,
    invert: bool = False,
) -> np.ndarray:
    """Exact DensityProcessor.normalize_to_real() behavior from jgui.py."""
    pred = np.asarray(pred_norm, dtype=float)
    if invert:
        pred = -pred
    positive = np.maximum(pred, 0.0) * float(rho_max)
    negative = np.minimum(pred, 0.0) * abs(float(rho_min))
    return (positive + negative).astype(np.float32)


class Visualizer:
    @staticmethod
    def anomaly_levels(volume: np.ndarray, threshold: float) -> Tuple[float, float]:
        values = np.asarray(volume, dtype=float)
        finite = values[np.isfinite(values)]
        amplitude = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 1e-8)
        level = max(float(threshold), 0.0) * amplitude
        return level, -level

    @staticmethod
    def plot_3d_voxels(
        ax,
        volume: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_extent: float,
        vmin: float,
        vmax: float,
        threshold: float = 0.3,
        title: str = "",
        reverse_color: bool = False,
        show_negative: bool = False,
        threshold_relative_to_data: bool = False,
    ) -> None:
        """The exact jgui fallback used when scikit-image is unavailable."""
        ax.clear()
        ax.set_facecolor("none")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.xaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.3)
        ax.yaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.3)
        ax.zaxis._axinfo["grid"]["color"] = (0.8, 0.8, 0.8, 0.3)

        nz, ny, nx = volume.shape
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_extent, 0)

        max_size = 16
        if nx > max_size or ny > max_size or nz > max_size:
            target_shape = (
                min(nz, max_size),
                min(ny, max_size),
                min(nx, max_size),
            )
            scale = tuple(
                target / source
                for target, source in zip(target_shape, (nz, ny, nx))
            )
            vol_ds = zoom(volume, scale, order=0, mode="nearest")
        else:
            vol_ds = np.asarray(volume)

        nz_ds, ny_ds, nx_ds = vol_ds.shape
        flat_vol = vol_ds.flatten()
        data_min, data_max = flat_vol.min(), flat_vol.max()
        data_range = data_max - data_min
        print(f"[Voxel Debug] {title}")
        print(f"  Data range: [{data_min:.2f}, {data_max:.2f}]")

        if data_range < 1e-6:
            ax.set_title(f"{title}\n(No significant anomaly)")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Depth (m)")
            return

        vmin, vmax = float(vmin), float(vmax)
        pos_thresh, neg_thresh = Visualizer.anomaly_levels(vol_ds, threshold)
        filled = (
            (vol_ds > pos_thresh) | (vol_ds < neg_thresh)
            if show_negative
            else vol_ds > pos_thresh
        )

        if filled.sum() == 0:
            ax.set_title(f"{title}\n(No voxels above threshold)")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Depth (m)")
            return

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = np.zeros((*vol_ds.shape, 4))
        for iz in range(nz_ds):
            for iy in range(ny_ds):
                for ix in range(nx_ds):
                    if filled[iz, iy, ix]:
                        norm_value = norm(vol_ds[iz, iy, ix])
                        if reverse_color:
                            norm_value = 1.0 - norm_value
                        colors[iz, iy, ix] = cm.jet(norm_value)

        x_extent = x_range[1] - x_range[0]
        y_extent = y_range[1] - y_range[0]
        x = np.linspace(0, x_extent, nx_ds + 1)
        y = np.linspace(0, y_extent, ny_ds + 1)
        z = np.linspace(0, z_extent, nz_ds + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        filled_t = np.transpose(filled, (2, 1, 0))
        colors_t = np.transpose(colors, (2, 1, 0, 3))
        ax.voxels(
            X,
            Y,
            Z,
            filled_t,
            facecolors=colors_t,
            edgecolor="none",
            alpha=0.8,
        )

        mappable = cm.ScalarMappable(norm=norm, cmap="jet")
        mappable.set_array(vol_ds)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6)
        cbar.ax.tick_params(direction="in")
        cbar.ax.set_title("kg/m³", fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        ax.set_xlim(0, x_extent)
        ax.set_ylim(0, y_extent)
        ax.set_zlim(z_extent, 0)
        ax.view_init(elev=25, azim=225)

    @staticmethod
    def plot_3d_isosurface(
        ax,
        volume: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_extent: float,
        vmin: float,
        vmax: float,
        threshold: float = 0.3,
        title: str = "",
        show_negative: bool = False,
        threshold_relative_to_data: bool = False,
    ) -> None:
        """Copied from jgui.py so standalone PNGs use the same renderer."""
        ax.clear()
        try:
            from skimage import measure
        except ImportError:
            measure = None

        ax.set_facecolor("none")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
        ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))

        nz, ny, nx = volume.shape
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_extent, 0)

        data_min, data_max = volume.min(), volume.max()
        data_range = data_max - data_min
        print(f"[Isosurface Debug] {title}")
        print(f"  Data range: [{data_min:.2f}, {data_max:.2f}]")

        if data_range < 1e-6:
            ax.set_title(f"{title}\n(No significant anomaly)")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Depth (m)")
            return

        vmin, vmax = float(vmin), float(vmax)
        pos_level, neg_level = Visualizer.anomaly_levels(volume, threshold)
        print(f"  vmin={vmin:.2f}, vmax={vmax:.2f}, center=0.00")
        print(f"  pos_level={pos_level:.2f}, neg_level={neg_level:.2f}")

        spacing = (
            (x_range[1] - x_range[0]) / nx,
            (y_range[1] - y_range[0]) / ny,
            z_extent / nz,
        )
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        surfaces_drawn = 0

        def vtk_surface(level: float):
            try:
                import vtk
                from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
                from scipy.ndimage import map_coordinates
            except ImportError:
                return None

            volume_xyz = np.asarray(np.transpose(volume, (2, 1, 0)), dtype=np.float32)
            nx_v, ny_v, nz_v = volume_xyz.shape

            image = vtk.vtkImageData()
            image.SetDimensions(nx_v, ny_v, nz_v)
            image.SetSpacing(spacing)
            image.SetOrigin(x_range[0], y_range[0], 0.0)
            vtk_array = numpy_to_vtk(
                volume_xyz.ravel(order="F"),
                deep=True,
                array_type=vtk.VTK_FLOAT,
            )
            image.GetPointData().SetScalars(vtk_array)

            contour = vtk.vtkMarchingCubes()
            contour.SetInputData(image)
            contour.SetValue(0, float(level))
            contour.ComputeNormalsOn()
            contour.ComputeGradientsOn()
            contour.Update()

            tri = vtk.vtkTriangleFilter()
            tri.SetInputConnection(contour.GetOutputPort())
            tri.Update()

            poly = tri.GetOutput()
            if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfCells() == 0:
                return None

            points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float, copy=False)
            polys = vtk_to_numpy(poly.GetPolys().GetData())
            if polys.size == 0:
                return None
            faces = polys.reshape(-1, 4)[:, 1:4].astype(int, copy=False)

            coords = np.vstack(
                [
                    (points[:, 0] - x_range[0]) / max(spacing[0], 1e-8),
                    (points[:, 1] - y_range[0]) / max(spacing[1], 1e-8),
                    points[:, 2] / max(spacing[2], 1e-8),
                ]
            )
            vertex_values = map_coordinates(
                volume_xyz,
                coords,
                order=1,
                mode="nearest",
            )
            return points, faces, vertex_values

        if pos_level < data_max:
            try:
                if measure is not None:
                    verts, faces, _normals, values = measure.marching_cubes(
                        volume, level=pos_level, spacing=(spacing[2], spacing[1], spacing[0])
                    )
                    verts_plot = np.zeros_like(verts)
                    verts_plot[:, 0] = verts[:, 2] + x_range[0]
                    verts_plot[:, 1] = verts[:, 1] + y_range[0]
                    verts_plot[:, 2] = verts[:, 0]
                    face_values = values[faces].mean(axis=1)
                else:
                    vtk_result = vtk_surface(pos_level)
                    if vtk_result is None:
                        raise RuntimeError("VTK marching cubes returned no surface")
                    verts_plot, faces, values = vtk_result
                    face_values = values[faces].mean(axis=1)
                mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
                mesh.set_facecolor(cm.jet(norm(face_values)))
                mesh.set_edgecolor("none")
                ax.add_collection3d(mesh)
                surfaces_drawn += 1
                print(
                    f"  Positive isosurface: {len(faces)} faces "
                    f"at level {pos_level:.1f}"
                )
            except Exception as exc:
                print(f"  Positive isosurface failed: {exc}")

        if show_negative and neg_level > data_min:
            try:
                if measure is not None:
                    verts, faces, _normals, _values = measure.marching_cubes(
                        volume, level=neg_level, spacing=(spacing[2], spacing[1], spacing[0])
                    )
                    verts_plot = np.zeros_like(verts)
                    verts_plot[:, 0] = verts[:, 2] + x_range[0]
                    verts_plot[:, 1] = verts[:, 1] + y_range[0]
                    verts_plot[:, 2] = verts[:, 0]
                else:
                    vtk_result = vtk_surface(neg_level)
                    if vtk_result is None:
                        raise RuntimeError("VTK marching cubes returned no surface")
                    verts_plot, faces, _values = vtk_result
                mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
                mesh.set_facecolor(cm.jet(norm(vmin)))
                mesh.set_edgecolor("none")
                ax.add_collection3d(mesh)
                surfaces_drawn += 1
                print(
                    f"  Negative isosurface: {len(faces)} faces "
                    f"at level {neg_level:.1f}"
                )
            except Exception as exc:
                print(f"  Negative isosurface failed: {exc}")

        if surfaces_drawn == 0:
            ax.text(
                (x_range[0] + x_range[1]) / 2,
                (y_range[0] + y_range[1]) / 2,
                z_extent / 2,
                "No isosurface\n(adjust threshold)",
                ha="center",
                va="center",
                fontsize=12,
            )

        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(z_extent, 0)

        mappable = cm.ScalarMappable(norm=norm, cmap="jet")
        mappable.set_array([vmin, vmax])
        try:
            fig = ax.get_figure()
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
            cbar.ax.tick_params(direction="in")
            cbar.ax.set_title("kg/m³", fontsize=12)
        except Exception as exc:
            print(f"  Colorbar failed: {exc}")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        ax.set_title("")

        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 0.3))
        ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
        ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
        ax.zaxis._axinfo["grid"]["linewidth"] = 0.5
        ax.xaxis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.6)
        ax.yaxis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.6)
        ax.zaxis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(8))
        ax.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax.zaxis.set_major_locator(plt.MaxNLocator(8))

        grid_num = 15
        grid_color = (0.5, 0.5, 0.5, 0.3)
        x_grid = np.linspace(x_range[0], x_range[1], grid_num)
        y_grid = np.linspace(y_range[0], y_range[1], grid_num)
        X_floor, Y_floor = np.meshgrid(x_grid, y_grid)
        ax.plot_wireframe(
            X_floor,
            Y_floor,
            np.zeros_like(X_floor),
            color=grid_color,
            linewidth=0.3,
        )

        z_grid = np.linspace(0, z_extent, grid_num)
        X_back, Z_back = np.meshgrid(x_grid, z_grid)
        ax.plot_wireframe(
            X_back,
            np.full_like(X_back, y_range[0]),
            Z_back,
            color=grid_color,
            linewidth=0.3,
        )

        Y_left, Z_left = np.meshgrid(y_grid, z_grid)
        ax.plot_wireframe(
            np.full_like(Y_left, x_range[0]),
            Y_left,
            Z_left,
            color=grid_color,
            linewidth=0.3,
        )

        x_len = x_range[1] - x_range[0]
        y_len = y_range[1] - y_range[0]
        z_len = z_extent
        max_len = max(x_len, y_len, z_len)
        z_scale = max(1.0, max_len / z_len * 0.6) if z_len > 0 else 1.0
        ax.set_box_aspect(
            [x_len / max_len, y_len / max_len, z_len / max_len * z_scale]
        )
        ax.view_init(elev=25, azim=225)

def case_threshold(path: Path) -> float:
    text = f"{path.parent.name} {path.stem}".lower()
    return 0.15 if "example four" in text or "inverted_pyramid" in text else 0.30


def case_shows_negative(path: Path, data: Optional[LoadedData] = None) -> bool:
    if data is not None and data.density_raw is not None:
        values = np.asarray(data.density_raw, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            return bool(np.any(finite > 0.0) and np.any(finite < 0.0))

    text = f"{path.parent.name} {path.stem}".lower()
    return "example two" in text or "prisms" in text


def case_density_range(
    source_path: Path,
    data: Optional[LoadedData],
) -> Tuple[float, float]:
    if data is None or data.density_raw is None:
        return -100.0, 350.0

    values = np.asarray(data.density_raw, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -100.0, 350.0

    text = f"{source_path.parent.name} {source_path.stem}".lower()
    amplitude = float(np.max(np.abs(finite)))
    if amplitude <= 0.0:
        return 0.0, 1.0

    if np.any(finite < 0.0) and np.any(finite > 0.0):
        return -300.0 * amplitude, 300.0 * amplitude

    if "example four" in text or "inverted_pyramid" in text:
        return 0.0, 150.0

    return 0.0, 300.0 * amplitude


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return slug or "case"


def example_number(path: Path) -> int:
    match = re.search(r"example\s+(one|two|three|four)", path.parent.name.lower())
    values = {"one": 1, "two": 2, "three": 3, "four": 4}
    return values.get(match.group(1), 99) if match else 99


def save_prediction_artifacts(
    output_dir: Path,
    source_path: Path,
    prediction_norm: np.ndarray,
    data: LoadedData,
    prefix: int,
) -> Path:
    rho_min, rho_max = case_density_range(source_path, data)
    prediction_real = normalize_to_real(
        prediction_norm, rho_min=rho_min, rho_max=rho_max, invert=False
    )
    case_dir = output_dir / f"{prefix:02d}_{slugify(source_path.stem)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "pred_density.npy", prediction_real)

    fig = Figure(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    Visualizer.plot_3d_isosurface(
        ax,
        prediction_real,
        data.x_range,
        data.y_range,
        prediction_real.shape[0] * data.dz,
        rho_min,
        rho_max,
        case_threshold(source_path),
        "",
        show_negative=case_shows_negative(source_path, data),
    )
    fig.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.1)
    fig.savefig(
        case_dir / "isosurface_pred.png",
        dpi=300,
        facecolor="white",
        pad_inches=0.2,
    )
    plt.close(fig)

    print(f"Saved: {case_dir}")
    print(
        f"  prediction_norm=[{prediction_norm.min():.6g}, "
        f"{prediction_norm.max():.6g}]"
    )
    print(
        f"  prediction_real=[{prediction_real.min():.6g}, "
        f"{prediction_real.max():.6g}]"
    )
    print(
        f"  threshold={case_threshold(source_path):.0%}, "
        f"show_negative={case_shows_negative(source_path, data)}"
    )
    return case_dir


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    return model(input_tensor.to(device)).squeeze().cpu().numpy().astype(np.float32)


def run_case(
    model: torch.nn.Module,
    config: V4Config,
    device: torch.device,
    source_path: Path,
    output_dir: Path,
    prefix: int,
    is_field_data: bool = False,
) -> None:
    data = load_gzz_text(source_path) if is_field_data else load_vti(source_path)
    input_tensor = build_input_tensor(data, config)
    prediction_norm = predict(model, input_tensor, device)
    save_prediction_artifacts(output_dir, source_path, prediction_norm, data, prefix)


def find_example_files(models_dir: Path) -> List[Path]:
    files = list(models_dir.rglob("*.vti"))
    return sorted(files, key=lambda path: (example_number(path), str(path).lower()))


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    models_dir = args.models_dir.resolve()
    field_file = args.field_file.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    device = resolve_device(args.device)
    model, config = load_model(checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = find_example_files(models_dir)
    if not examples and (args.skip_field_data or not field_file.is_file()):
        raise FileNotFoundError(f"No VTI examples found under: {models_dir}")

    for index, path in enumerate(examples, start=1):
        run_case(model, config, device, path, output_dir, index)

    if not args.skip_field_data:
        if not field_file.is_file():
            raise FileNotFoundError(f"Field data file not found: {field_file}")
        run_case(
            model,
            config,
            device,
            field_file,
            output_dir,
            len(examples) + 1,
            is_field_data=True,
        )

    print(f"All prediction artifacts saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
