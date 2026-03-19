from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter, zoom

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run this script. Please use the same Python environment used for training/inference."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_module_paths() -> None:
    """Make local project imports work from a standalone script."""
    candidates = [
        PROJECT_ROOT / "source code",
        PROJECT_ROOT / "source code" / "Train code",
        PROJECT_ROOT / "source code" / "Dataset preparation",
    ]
    for path in candidates:
        if path.is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


configure_module_paths()

from config import V4Config  # noqa: E402
from data_preparation import FastGravityForward  # noqa: E402
from train_code import PhysicsInformedUNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inversion test script for the three example VTI models and the field Gzz dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "best_model.pth",
        help="Path to the trained checkpoint file.",
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=PROJECT_ROOT / "examples",
        help="Directory containing example VTI files.",
    )
    parser.add_argument(
        "--field-gzz",
        type=Path,
        default=PROJECT_ROOT / "Field data example" / "Gzz.txt",
        help="Path to the field Gzz text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "batch_test_results",
        help="Directory where figures, arrays, and summaries will be saved.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--rho-min",
        type=float,
        default=-150.0,
        help="Minimum physical density used for field-data export.",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=350.0,
        help="Maximum physical density used for field-data export.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if a_flat.size == 0 or b_flat.size == 0:
        return 0.0
    if np.allclose(a_flat.std(), 0.0) or np.allclose(b_flat.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def resize_2d(data: np.ndarray, shape: Tuple[int, int], order: int = 1) -> np.ndarray:
    if tuple(data.shape) == tuple(shape):
        return data.copy()
    sy = shape[0] / data.shape[0]
    sx = shape[1] / data.shape[1]
    return zoom(data, (sy, sx), order=order)


def resize_3d(data: np.ndarray, shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    if tuple(data.shape) == tuple(shape):
        return data.copy()
    sz = shape[0] / data.shape[0]
    sy = shape[1] / data.shape[1]
    sx = shape[2] / data.shape[2]
    return zoom(data, (sz, sy, sx), order=order)


def extract_state_dict(checkpoint: Any) -> Any:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def build_config_from_checkpoint(raw_cfg: Any) -> V4Config:
    cfg = V4Config()
    if raw_cfg is None:
        return cfg
    if isinstance(raw_cfg, V4Config):
        return raw_cfg
    if isinstance(raw_cfg, dict):
        src = raw_cfg
    elif hasattr(raw_cfg, "__dict__"):
        src = {k: v for k, v in vars(raw_cfg).items() if not k.startswith("_")}
    else:
        return cfg

    valid = {k: v for k, v in src.items() if hasattr(cfg, k)}
    try:
        return V4Config(**valid)
    except Exception:
        for key, value in valid.items():
            setattr(cfg, key, value)
        return cfg


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, V4Config, Dict[str, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    cfg = build_config_from_checkpoint(raw_cfg)
    cfg.device = str(device)
    cfg.data_mode = str(cfg.data_mode).lower()
    cfg.grid_shape = tuple(int(v) for v in cfg.grid_shape)

    model = PhysicsInformedUNet(cfg).to(device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    info = {
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return model, cfg, info


def normalize_to_physical_density(pred_norm: np.ndarray, rho_min: float, rho_max: float) -> np.ndarray:
    scale = (rho_max - rho_min) / 2.0
    offset = (rho_max + rho_min) / 2.0
    return pred_norm * scale + offset


def scale_physical_density_by_gzz(
    pred_physical: np.ndarray,
    gzz_obs: np.ndarray,
    dx: float,
    dz: float,
    background: float,
) -> Tuple[np.ndarray, float, bool]:
    anomaly = pred_physical - background
    if np.max(np.abs(anomaly)) < 1e-8:
        return pred_physical, 1.0, False

    forward = FastGravityForward(pred_physical.shape, dx=dx, dz=dz, mode="gzz")
    gzz_calc = forward.forward(anomaly / 1000.0)

    if gzz_calc.shape != gzz_obs.shape:
        gzz_obs = resize_2d(gzz_obs, gzz_calc.shape, order=3)

    obs_max = np.abs(gzz_obs).max()
    mask = np.abs(gzz_obs) > obs_max * 0.3 if obs_max > 0 else np.ones_like(gzz_obs, dtype=bool)

    numerator = float(np.sum(gzz_calc[mask] * gzz_obs[mask]))
    denominator = float(np.sum(gzz_calc[mask] * gzz_calc[mask]) + 1e-10)
    sign_flipped = False
    if numerator < 0:
        anomaly = -anomaly
        numerator = -numerator
        sign_flipped = True

    alpha = float(np.clip(numerator / denominator, 0.5, 5.0))
    scaled = background + anomaly * alpha

    min_density = background - 300.0
    max_density = background + 400.0
    scaled = np.clip(scaled, min_density, max_density)
    return scaled, alpha, sign_flipped


def load_gzz_txt(path: Path) -> Dict[str, Any]:
    try:
        raw = np.loadtxt(path, delimiter=",")
    except Exception:
        raw = np.loadtxt(path)

    if raw.ndim == 1:
        raise ValueError(f"Expected a 2D array or XYZ table in {path}")

    if raw.ndim == 2 and raw.shape[1] >= 3:
        x = raw[:, 0]
        y = raw[:, 1]
        gzz = raw[:, 2]
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        nx = len(unique_x)
        ny = len(unique_y)

        grid = np.full((ny, nx), np.nan, dtype=np.float64)
        x_index = {value: idx for idx, value in enumerate(unique_x)}
        y_index = {value: idx for idx, value in enumerate(unique_y)}
        for xi, yi, zi in zip(x, y, gzz):
            grid[y_index[yi], x_index[xi]] = zi
        if np.isnan(grid).any():
            raise ValueError(f"Failed to reshape {path} into a complete 2D grid.")

        dx = float(np.median(np.diff(unique_x))) if nx > 1 else 100.0
        dy = float(np.median(np.diff(unique_y))) if ny > 1 else 100.0
        x_range = (float(unique_x.min()), float(unique_x.max()))
        y_range = (float(unique_y.min()), float(unique_y.max()))
        gzz_map = grid
    else:
        gzz_map = np.asarray(raw, dtype=np.float64)
        ny, nx = gzz_map.shape
        dx = 100.0
        dy = 100.0
        x_range = (0.0, dx * (nx - 1))
        y_range = (0.0, dy * (ny - 1))

    if np.max(np.abs(gzz_map)) < 1e-4:
        gzz_map = gzz_map * 1e9

    return {
        "gzz": gzz_map,
        "x_range": x_range,
        "y_range": y_range,
        "dx": dx,
        "dy": dy,
        "shape": tuple(gzz_map.shape),
        "source": str(path),
    }


def load_vti_density(path: Path) -> Dict[str, Any]:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError("VTK is required to read .vti files. Please install it with `pip install vtk`.") from exc

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image_data = reader.GetOutput()

    dims = image_data.GetDimensions()
    spacing = image_data.GetSpacing()
    origin = image_data.GetOrigin()
    nx, ny, nz = dims

    data_flat = None
    point_data = image_data.GetPointData()
    if point_data.GetScalars() is not None:
        data_flat = vtk_to_numpy(point_data.GetScalars())
    elif point_data.GetNumberOfArrays() > 0:
        data_flat = vtk_to_numpy(point_data.GetArray(0))
    else:
        cell_data = image_data.GetCellData()
        if cell_data.GetScalars() is not None:
            data_flat = vtk_to_numpy(cell_data.GetScalars())
            nx, ny, nz = max(1, nx - 1), max(1, ny - 1), max(1, nz - 1)
        elif cell_data.GetNumberOfArrays() > 0:
            data_flat = vtk_to_numpy(cell_data.GetArray(0))
            nx, ny, nz = max(1, nx - 1), max(1, ny - 1), max(1, nz - 1)

    if data_flat is None:
        raise ValueError(f"No scalar array was found in {path}")

    total = data_flat.size
    target = nx * ny * nz
    if total != target:
        raise ValueError(f"Unexpected VTI data size in {path}: {total} vs {target}")

    try:
        density = data_flat.reshape((nz, ny, nx), order="F")
    except ValueError:
        density = data_flat.reshape((nz, ny, nx))

    return {
        "density": np.asarray(density, dtype=np.float32),
        "x_range": (float(origin[0]), float(origin[0] + spacing[0] * (nx - 1))),
        "y_range": (float(origin[1]), float(origin[1] + spacing[1] * (ny - 1))),
        "dx": float(spacing[0]) if len(spacing) > 0 else 100.0,
        "dy": float(spacing[1]) if len(spacing) > 1 else 100.0,
        "dz": float(spacing[2]) if len(spacing) > 2 else 100.0,
        "shape": tuple(density.shape),
        "source": str(path),
    }


def build_input_tensor_from_fields(
    gz_map: np.ndarray,
    gzz_map: Optional[np.ndarray],
    nz: int,
    data_mode: str,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    target_ny, target_nx = int(target_hw[0]), int(target_hw[1])
    gz_map = resize_2d(gz_map, (target_ny, target_nx), order=1)
    gz_norm = gz_map / (np.max(np.abs(gz_map)) + 1e-8)

    z_indices = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    z_map = np.tile(z_indices[:, None, None], (1, target_ny, target_nx))

    gz_vol = np.tile(gz_norm[None, :, :], (nz, 1, 1))
    if data_mode == "joint":
        if gzz_map is None:
            raise ValueError("Joint mode requires both gz and gzz inputs.")
        gzz_map = resize_2d(gzz_map, (target_ny, target_nx), order=1)
        gzz_norm = gzz_map / (np.max(np.abs(gzz_map)) + 1e-8)
        gzz_vol = np.tile(gzz_norm[None, :, :], (nz, 1, 1))
        input_vol = np.stack([gz_vol, gzz_vol, z_map], axis=0)
    else:
        input_vol = np.stack([gz_vol, z_map], axis=0)

    return torch.from_numpy(input_vol.astype(np.float32)).unsqueeze(0)


def build_vti_case_inputs(vti_data: Dict[str, Any], cfg: V4Config) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    density = np.asarray(vti_data["density"], dtype=np.float32)
    dx = float(vti_data["dx"])
    dz = float(vti_data["dz"])
    nz, ny, nx = density.shape
    target_hw = (int(cfg.grid_shape[1]), int(cfg.grid_shape[2]))

    if cfg.data_mode == "joint":
        forward = FastGravityForward((nz, ny, nx), dx=dx, dz=dz, mode="joint")
        gz_raw, gzz_raw = forward.forward(density)
    else:
        forward = FastGravityForward((nz, ny, nx), dx=dx, dz=dz, mode="gz")
        gz_raw = forward.forward(density)
        gzz_raw = None

    tensor = build_input_tensor_from_fields(
        gz_map=gz_raw,
        gzz_map=gzz_raw,
        nz=int(cfg.grid_shape[0]),
        data_mode=str(cfg.data_mode).lower(),
        target_hw=target_hw,
    )

    return tensor, {
        "gz": gz_raw,
        "gzz": gzz_raw if gzz_raw is not None else uniform_filter(gz_raw, size=5),
    }


def build_field_case_inputs(field_data: Dict[str, Any], cfg: V4Config) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    gzz_map = np.asarray(field_data["gzz"], dtype=np.float32)
    gz_map = uniform_filter(gzz_map, size=5)
    tensor = build_input_tensor_from_fields(
        gz_map=gz_map,
        gzz_map=gzz_map if str(cfg.data_mode).lower() == "joint" else None,
        nz=int(cfg.grid_shape[0]),
        data_mode=str(cfg.data_mode).lower(),
        target_hw=(int(cfg.grid_shape[1]), int(cfg.grid_shape[2])),
    )
    return tensor, {"gz": gz_map, "gzz": gzz_map}


def predict_density(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        pred = model(input_tensor.to(device)).squeeze().detach().cpu().numpy()
    return np.asarray(pred, dtype=np.float32)


def central_slice_indices(shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return shape[0] // 2, shape[1] // 2, shape[2] // 2


def save_array(path: Path, array: np.ndarray) -> None:
    ensure_parent(path.parent)
    np.save(path, np.asarray(array))


def compute_density_metrics(true_density: np.ndarray, pred_density: np.ndarray) -> Dict[str, float]:
    diff = pred_density - true_density
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    corr = safe_corrcoef(true_density, pred_density)
    return {"density_rmse": rmse, "density_mae": mae, "density_corr": corr}


def compute_field_metrics(obs_gzz: np.ndarray, pred_gzz: np.ndarray) -> Dict[str, float]:
    diff = pred_gzz - obs_gzz
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    corr = safe_corrcoef(obs_gzz, pred_gzz)
    return {"gzz_rmse": rmse, "gzz_mae": mae, "gzz_corr": corr}


def plot_slice(ax: plt.Axes, data: np.ndarray, title: str, xlabel: str, ylabel: str, cmap: str, vmin: float, vmax: float) -> None:
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_vti_comparison_figure(
    output_path: Path,
    case_name: str,
    true_density: np.ndarray,
    pred_density: np.ndarray,
    obs_gzz: np.ndarray,
    pred_gzz: np.ndarray,
    metrics: Dict[str, Any],
) -> None:
    ensure_parent(output_path.parent)
    cz, cy, cx = central_slice_indices(pred_density.shape)
    vmax_density = float(max(np.max(np.abs(true_density)), np.max(np.abs(pred_density)), 1e-6))
    vmax_gzz = float(max(np.max(np.abs(obs_gzz)), np.max(np.abs(pred_gzz)), 1e-6))
    residual = pred_gzz - obs_gzz
    vmax_res = float(max(np.max(np.abs(residual)), 1e-6))

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle(
        f"{case_name}\n"
        f"Density corr={metrics['density_corr']:.4f}, RMSE={metrics['density_rmse']:.4f} | "
        f"Gzz corr={metrics['gzz_corr']:.4f}, RMSE={metrics['gzz_rmse']:.4f}",
        fontsize=13,
    )

    plot_slice(axes[0, 0], true_density[cz], "True Density - Z Slice", "X", "Y", "seismic", -vmax_density, vmax_density)
    plot_slice(axes[0, 1], true_density[:, cy, :], "True Density - Y Slice", "X", "Depth", "seismic", -vmax_density, vmax_density)
    plot_slice(axes[0, 2], true_density[:, :, cx], "True Density - X Slice", "Y", "Depth", "seismic", -vmax_density, vmax_density)

    plot_slice(axes[1, 0], pred_density[cz], "Predicted Density - Z Slice", "X", "Y", "seismic", -vmax_density, vmax_density)
    plot_slice(axes[1, 1], pred_density[:, cy, :], "Predicted Density - Y Slice", "X", "Depth", "seismic", -vmax_density, vmax_density)
    plot_slice(axes[1, 2], pred_density[:, :, cx], "Predicted Density - X Slice", "Y", "Depth", "seismic", -vmax_density, vmax_density)

    plot_slice(axes[2, 0], obs_gzz, "Observed Gzz From True Model", "X", "Y", "jet", -vmax_gzz, vmax_gzz)
    plot_slice(axes[2, 1], pred_gzz, "Forward Gzz From Prediction", "X", "Y", "jet", -vmax_gzz, vmax_gzz)
    plot_slice(axes[2, 2], residual, "Gzz Residual (Pred - Obs)", "X", "Y", "RdBu_r", -vmax_res, vmax_res)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_field_prediction_figure(
    output_path: Path,
    case_name: str,
    pred_density_physical: np.ndarray,
    obs_gzz: np.ndarray,
    pred_gzz: np.ndarray,
    metrics: Dict[str, Any],
) -> None:
    ensure_parent(output_path.parent)
    cz, cy, cx = central_slice_indices(pred_density_physical.shape)
    density_min = float(np.min(pred_density_physical))
    density_max = float(np.max(pred_density_physical))
    if density_max - density_min < 1e-6:
        density_min -= 0.5
        density_max += 0.5
    vmax_gzz = float(max(np.max(np.abs(obs_gzz)), np.max(np.abs(pred_gzz)), 1e-6))
    residual = pred_gzz - obs_gzz
    vmax_res = float(max(np.max(np.abs(residual)), 1e-6))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"{case_name}\nGzz corr={metrics['gzz_corr']:.4f}, RMSE={metrics['gzz_rmse']:.4f}",
        fontsize=13,
    )

    plot_slice(axes[0, 0], obs_gzz, "Observed Gzz", "X", "Y", "jet", -vmax_gzz, vmax_gzz)
    plot_slice(axes[0, 1], pred_gzz, "Forward Gzz From Prediction", "X", "Y", "jet", -vmax_gzz, vmax_gzz)
    plot_slice(axes[0, 2], residual, "Gzz Residual (Pred - Obs)", "X", "Y", "RdBu_r", -vmax_res, vmax_res)

    plot_slice(axes[1, 0], pred_density_physical[cz], "Predicted Density - Z Slice (kg/m^3)", "X", "Y", "jet", density_min, density_max)
    plot_slice(axes[1, 1], pred_density_physical[:, cy, :], "Predicted Density - Y Slice (kg/m^3)", "X", "Depth", "jet", density_min, density_max)
    plot_slice(axes[1, 2], pred_density_physical[:, :, cx], "Predicted Density - X Slice (kg/m^3)", "Y", "Depth", "jet", density_min, density_max)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_case_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    ensure_parent(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)


def slugify_case_name(name: str) -> str:
    value = name.lower().replace(" ", "_")
    return "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-"})


def run_vti_case(
    model: torch.nn.Module,
    cfg: V4Config,
    device: torch.device,
    vti_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    case_name = vti_path.stem
    case_dir = output_dir / slugify_case_name(case_name)
    ensure_parent(case_dir)

    vti_data = load_vti_density(vti_path)
    input_tensor, obs_fields = build_vti_case_inputs(vti_data, cfg)
    pred_density = predict_density(model, input_tensor, device)

    true_density = resize_3d(np.asarray(vti_data["density"], dtype=np.float32), pred_density.shape, order=1)
    raw_corr = safe_corrcoef(true_density, pred_density)
    sign_flipped = raw_corr < 0
    pred_aligned = -pred_density if sign_flipped else pred_density

    forward_pred = FastGravityForward(pred_aligned.shape, dx=float(vti_data["dx"]), dz=float(vti_data["dz"]), mode="gzz")
    pred_gzz = forward_pred.forward(pred_aligned)
    obs_gzz = resize_2d(np.asarray(obs_fields["gzz"], dtype=np.float32), pred_gzz.shape, order=1)

    density_metrics = compute_density_metrics(true_density, pred_aligned)
    gzz_metrics = compute_field_metrics(obs_gzz, pred_gzz)

    metrics: Dict[str, Any] = {
        "case_type": "vti_example",
        "case_name": case_name,
        "source": str(vti_path),
        "grid_shape_model": list(map(int, cfg.grid_shape)),
        "grid_shape_true": list(map(int, vti_data["shape"])),
        "grid_shape_pred": list(map(int, pred_aligned.shape)),
        "data_mode": str(cfg.data_mode),
        "dx": safe_float(vti_data["dx"]),
        "dy": safe_float(vti_data["dy"]),
        "dz": safe_float(vti_data["dz"]),
        "raw_density_corr": raw_corr,
        "sign_flipped": sign_flipped,
        "true_density_min": safe_float(true_density.min()),
        "true_density_max": safe_float(true_density.max()),
        "pred_density_min": safe_float(pred_aligned.min()),
        "pred_density_max": safe_float(pred_aligned.max()),
        "obs_gzz_min": safe_float(obs_gzz.min()),
        "obs_gzz_max": safe_float(obs_gzz.max()),
        "pred_gzz_min": safe_float(pred_gzz.min()),
        "pred_gzz_max": safe_float(pred_gzz.max()),
        **density_metrics,
        **gzz_metrics,
    }

    save_array(case_dir / "true_density_resized.npy", true_density)
    save_array(case_dir / "pred_density_aligned.npy", pred_aligned)
    save_array(case_dir / "observed_gzz.npy", obs_gzz)
    save_array(case_dir / "predicted_gzz.npy", pred_gzz)
    save_vti_comparison_figure(case_dir / "comparison.png", case_name, true_density, pred_aligned, obs_gzz, pred_gzz, metrics)
    save_case_metrics(case_dir / "metrics.json", metrics)
    return metrics


def run_field_case(
    model: torch.nn.Module,
    cfg: V4Config,
    device: torch.device,
    field_path: Path,
    output_dir: Path,
    rho_min: float,
    rho_max: float,
) -> Dict[str, Any]:
    case_name = field_path.stem
    case_dir = output_dir / slugify_case_name(case_name)
    ensure_parent(case_dir)

    field_data = load_gzz_txt(field_path)
    input_tensor, obs_fields = build_field_case_inputs(field_data, cfg)
    pred_density_norm = predict_density(model, input_tensor, device)

    forward = FastGravityForward(pred_density_norm.shape, dx=float(field_data["dx"]), dz=float(cfg.dz), mode="gzz")
    pred_gzz_raw = forward.forward(pred_density_norm)
    obs_gzz = resize_2d(np.asarray(obs_fields["gzz"], dtype=np.float32), pred_gzz_raw.shape, order=1)

    raw_corr = safe_corrcoef(obs_gzz, pred_gzz_raw)
    sign_flipped = raw_corr < 0
    pred_density_aligned = -pred_density_norm if sign_flipped else pred_density_norm
    pred_gzz = -pred_gzz_raw if sign_flipped else pred_gzz_raw

    pred_density_physical = normalize_to_physical_density(pred_density_aligned, rho_min=rho_min, rho_max=rho_max)
    pred_density_scaled, scale_alpha, scale_sign_flipped = scale_physical_density_by_gzz(
        pred_density_physical,
        gzz_obs=obs_gzz,
        dx=float(field_data["dx"]),
        dz=float(cfg.dz),
        background=(rho_min + rho_max) / 2.0,
    )

    forward_scaled = FastGravityForward(pred_density_scaled.shape, dx=float(field_data["dx"]), dz=float(cfg.dz), mode="gzz")
    pred_gzz_scaled = forward_scaled.forward((pred_density_scaled - (rho_min + rho_max) / 2.0) / 1000.0)

    gzz_metrics = compute_field_metrics(obs_gzz, pred_gzz_scaled)
    metrics: Dict[str, Any] = {
        "case_type": "field_gzz",
        "case_name": case_name,
        "source": str(field_path),
        "grid_shape_model": list(map(int, cfg.grid_shape)),
        "grid_shape_pred": list(map(int, pred_density_scaled.shape)),
        "data_mode": str(cfg.data_mode),
        "dx": safe_float(field_data["dx"]),
        "dy": safe_float(field_data["dy"]),
        "dz": safe_float(cfg.dz),
        "raw_gzz_corr": raw_corr,
        "sign_flipped_from_gzz": sign_flipped,
        "sign_flipped_during_scaling": scale_sign_flipped,
        "scale_alpha": scale_alpha,
        "pred_density_norm_min": safe_float(pred_density_aligned.min()),
        "pred_density_norm_max": safe_float(pred_density_aligned.max()),
        "pred_density_physical_min": safe_float(pred_density_scaled.min()),
        "pred_density_physical_max": safe_float(pred_density_scaled.max()),
        "obs_gzz_min": safe_float(obs_gzz.min()),
        "obs_gzz_max": safe_float(obs_gzz.max()),
        "pred_gzz_min": safe_float(pred_gzz_scaled.min()),
        "pred_gzz_max": safe_float(pred_gzz_scaled.max()),
        **gzz_metrics,
    }

    save_array(case_dir / "pred_density_norm.npy", pred_density_aligned)
    save_array(case_dir / "pred_density_physical.npy", pred_density_scaled)
    save_array(case_dir / "observed_gzz.npy", obs_gzz)
    save_array(case_dir / "predicted_gzz.npy", pred_gzz_scaled)
    save_field_prediction_figure(case_dir / "prediction.png", case_name, pred_density_scaled, obs_gzz, pred_gzz_scaled, metrics)
    save_case_metrics(case_dir / "metrics.json", metrics)
    return metrics


def write_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_parent(output_dir)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    all_keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_example_vti_files(examples_dir: Path) -> List[Path]:
    return sorted(examples_dir.rglob("*.vti"))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = args.checkpoint.resolve()
    examples_dir = args.examples_dir.resolve()
    field_gzz_path = args.field_gzz.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not examples_dir.is_dir():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")
    if not field_gzz_path.is_file():
        raise FileNotFoundError(f"Field Gzz file not found: {field_gzz_path}")

    ensure_parent(output_dir)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, cfg, load_info = load_model(checkpoint_path, device=device)
    print(
        f"Model ready on {device}. data_mode={cfg.data_mode}, grid_shape={cfg.grid_shape}, "
        f"missing={load_info['missing_keys']}, unexpected={load_info['unexpected_keys']}"
    )

    summary_rows: List[Dict[str, Any]] = []

    vti_files = find_example_vti_files(examples_dir)
    print(f"Found {len(vti_files)} VTI example(s).")
    for vti_path in vti_files:
        print(f"Running VTI comparison: {vti_path}")
        metrics = run_vti_case(model, cfg, device, vti_path, output_dir / "examples")
        summary_rows.append(metrics)

    print(f"Running field-data prediction: {field_gzz_path}")
    field_metrics = run_field_case(
        model=model,
        cfg=cfg,
        device=device,
        field_path=field_gzz_path,
        output_dir=output_dir / "field_data",
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
    )
    summary_rows.append(field_metrics)

    write_summary(output_dir, summary_rows)
    print(f"All results were saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
