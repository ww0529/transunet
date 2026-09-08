from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run this script. Please use the same Python environment used for training."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


def configure_module_paths() -> None:
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

from config import V4Config  
from data_preparation import FastGravityForward  
from train_code import (  
    ComprehensiveLoss,
    DifferentiableForward,
    PhysicsInformedUNet,
    V4Metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate best_model.pth on VTI models stored in folders, "
            "following the validation logic from train_code.py."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "best_model.pth",
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "examples",
        help="Directory containing VTI model files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "test_code",
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size. Default is 1 to keep per-case reporting simple.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for evaluation.",
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def anomaly_levels(volume: np.ndarray, threshold: float) -> Tuple[float, float]:
    """Return positive and negative display levels from the data amplitude."""
    data = np.asarray(volume, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 0.0
    amplitude = float(np.max(np.abs(finite)))
    level = max(float(threshold), 0.0) * amplitude
    return level, -level


def case_threshold(case_name: str) -> float:
    """Match the jgui-style threshold split used for the inverted pyramid case."""
    return 0.15 if "pyramid" in case_name.lower() else 0.30


def case_shows_negative(case_name: str) -> bool:
    """Show negative anomaly surfaces for the two-prism case."""
    return "prisms" in case_name.lower()


def display_range_for_case() -> Tuple[float, float]:
    """Use the same display range requested for the true density reference."""
    return -100.0, 350.0


def plot_pred_isosurface(
    volume: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_extent: float,
    save_path: Path,
    threshold: float,
    vmin: float,
    vmax: float,
    show_negative: bool,
) -> None:
    """Render a jgui-style isosurface PNG for the predicted volume."""
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.clear()
    ax.set_facecolor("none")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))

    try:
        from skimage import measure
    except ImportError:
        measure = None

    nz, ny, nx = volume.shape
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_extent, 0)

    data_min = float(np.nanmin(volume))
    data_max = float(np.nanmax(volume))
    data_range = data_max - data_min

    if data_range < 1e-6:
        ax.text(
            0.5, 0.5, 0.5, "No significant anomaly",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Depth (m)")
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    pos_level, neg_level = anomaly_levels(volume, threshold)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    surfaces_drawn = 0

    if measure is None:
        filled = volume > pos_level
        if show_negative:
            filled = filled | (volume < neg_level)
        if filled.any():
            colors = np.zeros((*volume.shape, 4))
            nz, ny, nx = volume.shape
            x = np.linspace(0, x_range[1] - x_range[0], nx + 1)
            y = np.linspace(0, y_range[1] - y_range[0], ny + 1)
            z = np.linspace(0, z_extent, nz + 1)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            for iz in range(nz):
                for iy in range(ny):
                    for ix in range(nx):
                        if filled[iz, iy, ix]:
                            colors[iz, iy, ix] = cm.jet(norm(volume[iz, iy, ix]))
            ax.voxels(
                X,
                Y,
                Z,
                np.transpose(filled, (2, 1, 0)),
                facecolors=np.transpose(colors, (2, 1, 0, 3)),
                edgecolor="none",
                alpha=0.8,
            )
            surfaces_drawn = 1
    else:
        spacing = (
            z_extent / nz if nz else 1.0,
            (y_range[1] - y_range[0]) / ny if ny else 1.0,
            (x_range[1] - x_range[0]) / nx if nx else 1.0,
        )

        if pos_level < data_max:
            try:
                verts, faces, _normals, values = measure.marching_cubes(
                    volume, level=pos_level, spacing=spacing
                )
                verts_plot = np.zeros_like(verts)
                verts_plot[:, 0] = verts[:, 2] + x_range[0]
                verts_plot[:, 1] = verts[:, 1] + y_range[0]
                verts_plot[:, 2] = verts[:, 0]
                face_values = values[faces].mean(axis=1)
                face_colors = cm.jet(norm(face_values))
                mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
                mesh.set_facecolor(face_colors)
                mesh.set_edgecolor("none")
                ax.add_collection3d(mesh)
                surfaces_drawn += 1
            except Exception:
                pass

        if show_negative and neg_level > data_min:
            try:
                verts, faces, _normals, _values = measure.marching_cubes(
                    volume, level=neg_level, spacing=spacing
                )
                verts_plot = np.zeros_like(verts)
                verts_plot[:, 0] = verts[:, 2] + x_range[0]
                verts_plot[:, 1] = verts[:, 1] + y_range[0]
                verts_plot[:, 2] = verts[:, 0]
                face_colors = cm.jet(norm(vmin))
                mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
                mesh.set_facecolor(face_colors)
                mesh.set_edgecolor("none")
                ax.add_collection3d(mesh)
                surfaces_drawn += 1
            except Exception:
                pass

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

    mappable = cm.ScalarMappable(norm=norm, cmap="jet")
    mappable.set_array([vmin, vmax])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
    cbar.ax.tick_params(direction="in")
    cbar.ax.set_title("kg/m³", fontsize=12)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_title("")
    ax.view_init(elev=25, azim=225)
    x_len = x_range[1] - x_range[0]
    y_len = y_range[1] - y_range[0]
    z_len = max(z_extent, 1e-6)
    max_len = max(x_len, y_len, z_len)
    ax.set_box_aspect([x_len / max_len, y_len / max_len, z_len / max_len])

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_prediction_artifacts(
    case_dir: Path,
    case_name: str,
    pred_density: np.ndarray,
    dx: float,
    dz: float,
) -> None:
    ensure_dir(case_dir)
    np.save(case_dir / "pred_density.npy", pred_density)
    threshold = case_threshold(case_name)
    vmin, vmax = display_range_for_case()
    show_negative = case_shows_negative(case_name)
    z_extent = float(pred_density.shape[0]) * float(dz)
    x_range = (0.0, float(pred_density.shape[2]) * float(dx))
    y_range = (0.0, float(pred_density.shape[1]) * float(dx))
    plot_pred_isosurface(
        pred_density,
        x_range,
        y_range,
        z_extent,
        case_dir / "isosurface_pred.png",
        threshold,
        vmin,
        vmax,
        show_negative,
    )


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


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, V4Config]:
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

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  data_mode={cfg.data_mode}, grid_shape={cfg.grid_shape}, device={device}")
    print(f"  missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    return model, cfg


def resize_3d(data: np.ndarray, shape: Tuple[int, int, int], order: int = 1) -> np.ndarray:
    if tuple(data.shape) == tuple(shape):
        return data.copy()
    scale_factors = tuple(shape[i] / data.shape[i] for i in range(3))
    return zoom(data, scale_factors, order=order)


def load_vti_density(path: Path) -> np.ndarray:
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError as exc:
        raise RuntimeError("VTK is required to read .vti files. Install it with `pip install vtk`.") from exc

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image_data = reader.GetOutput()

    dims = image_data.GetDimensions()
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
        raise ValueError(f"No scalar array found in {path}")

    expected = nx * ny * nz
    if data_flat.size != expected:
        raise ValueError(f"Unexpected VTI data size in {path}: {data_flat.size} vs {expected}")

    try:
        density = data_flat.reshape((nz, ny, nx), order="F")
    except ValueError:
        density = data_flat.reshape((nz, ny, nx))

    return np.asarray(density, dtype=np.float32)


def build_model_inputs(
    density: np.ndarray,
    cfg: V4Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, ny, nx = cfg.grid_shape
    target_density = resize_3d(density, (nz, ny, nx), order=1).astype(np.float32)

    if cfg.data_mode == "joint":
        forward = FastGravityForward(cfg.grid_shape, dx=cfg.dx, dz=cfg.dz, mode="joint")
        gz, gzz = forward.forward(target_density)

        gz_norm = gz / (np.abs(gz).max() + 1e-8)
        gzz_norm = gzz / (np.abs(gzz).max() + 1e-8)
        gz_vol = np.repeat(gz_norm[None, :, :], nz, axis=0)
        gzz_vol = np.repeat(gzz_norm[None, :, :], nz, axis=0)
        z_indices = np.linspace(0, 1, nz, dtype=np.float32)
        z_map = np.tile(z_indices[:, None, None], (1, ny, nx))
        input_vol = np.stack([gz_vol, gzz_vol, z_map], axis=0).astype(np.float32)
        obs_gravity = gz.astype(np.float32)
    else:
        forward = FastGravityForward(cfg.grid_shape, dx=cfg.dx, dz=cfg.dz, mode="gz")
        gz = forward.forward(target_density)

        gz_norm = gz / (np.abs(gz).max() + 1e-8)
        gz_vol = np.repeat(gz_norm[None, :, :], nz, axis=0)
        z_indices = np.linspace(0, 1, nz, dtype=np.float32)
        z_map = np.tile(z_indices[:, None, None], (1, ny, nx))
        input_vol = np.stack([gz_vol, z_map], axis=0).astype(np.float32)
        obs_gravity = gz.astype(np.float32)

    return input_vol, target_density, obs_gravity


class FolderModelDataset(Dataset):
    def __init__(self, vti_files: List[Path], cfg: V4Config):
        self.vti_files = vti_files
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.vti_files)

    def __getitem__(self, idx: int):
        path = self.vti_files[idx]
        density = load_vti_density(path)
        input_vol, target_density, obs_gravity = build_model_inputs(density, self.cfg)

        return {
            "name": path.stem,
            "path": str(path),
            "inputs": torch.from_numpy(input_vol),
            "targets": torch.from_numpy(target_density).unsqueeze(0),
            "obs_gravity": torch.from_numpy(obs_gravity).unsqueeze(0),
        }


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = np.asarray(a, dtype=np.float64).ravel()
    b_flat = np.asarray(b, dtype=np.float64).ravel()
    if a_flat.size == 0 or b_flat.size == 0:
        return 0.0
    if np.allclose(a_flat.std(), 0.0) or np.allclose(b_flat.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def write_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@torch.no_grad()
def evaluate_folder_models(
    model: torch.nn.Module,
    cfg: V4Config,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    forward_op = DifferentiableForward(cfg.grid_shape, cfg.dx, cfg.dz).to(device)
    criterion = ComprehensiveLoss(cfg, forward_op)
    metrics = V4Metrics()

    model.eval()
    total_loss = 0.0
    case_losses: List[float] = []
    deep_ious: List[float] = []
    rel_errors: List[float] = []
    rows: List[Dict[str, Any]] = []
    case_index = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        obs_gravity = batch["obs_gravity"].to(device)
        names = batch["name"]
        paths = batch["path"]

        outputs = model(inputs)
        pred_gravity = forward_op(outputs)
        loss, _ = criterion(outputs, targets, obs_gravity)

        batch_size = outputs.shape[0]
        total_loss += loss.item()

        for i in range(batch_size):
            name = names[i]
            path = paths[i]
            sample_loss, _ = criterion(outputs[i:i + 1], targets[i:i + 1], obs_gravity[i:i + 1])

            pred_np = tensor_to_numpy(outputs[i, 0])
            true_np = tensor_to_numpy(targets[i, 0])
            obs_np = tensor_to_numpy(obs_gravity[i, 0])
            pred_grav_np = tensor_to_numpy(pred_gravity[i, 0])

            deep_iou = metrics.deep_anomaly_iou(outputs[i:i + 1], targets[i:i + 1])
            rel_error = metrics.relative_error(outputs[i:i + 1], targets[i:i + 1])
            density_rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))
            gravity_rmse = float(np.sqrt(np.mean((pred_grav_np - obs_np) ** 2)))
            density_corr = safe_corrcoef(pred_np, true_np)
            gravity_corr = safe_corrcoef(pred_grav_np, obs_np)

            deep_ious.append(deep_iou)
            rel_errors.append(rel_error)
            case_losses.append(float(sample_loss.item()))

            case_dir = output_dir / f"{case_index:02d}_{name.lower().replace(' ', '_')}"
            save_prediction_artifacts(case_dir, name, pred_np, cfg.dx, cfg.dz)

            row = {
                "report_type": "case",
                "case_name": name,
                "source": path,
                "loss": float(sample_loss.item()),
                "deep_iou": float(deep_iou),
                "relative_error": float(rel_error),
                "density_rmse": density_rmse,
                "gravity_rmse": gravity_rmse,
                "density_corr": density_corr,
                "gravity_corr": gravity_corr,
                "pred_min": float(pred_np.min()),
                "pred_max": float(pred_np.max()),
                "true_min": float(true_np.min()),
                "true_max": float(true_np.max()),
            }
            rows.append(row)
            case_index += 1

            print(
                f"[{name}] loss={row['loss']:.6f} "
                f"deep_iou={row['deep_iou']:.4f} "
                f"density_corr={row['density_corr']:.4f} "
                f"gravity_corr={row['gravity_corr']:.4f}"
            )

    summary = {
        "report_type": "summary",
        "mean_loss": float(total_loss / max(len(loader), 1)),
        "mean_case_loss": float(np.mean(case_losses)) if case_losses else 0.0,
        "mean_deep_iou": float(np.mean(deep_ious)) if deep_ious else 0.0,
        "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else 0.0,
        "num_cases": len(rows),
    }
    return rows, summary


def find_vti_files(models_dir: Path) -> List[Path]:
    return sorted(models_dir.rglob("*.vti"))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = args.checkpoint.resolve()
    models_dir = args.models_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {models_dir}")

    vti_files = find_vti_files(models_dir)
    if not vti_files:
        raise FileNotFoundError(f"No .vti files found under: {models_dir}")

    ensure_dir(output_dir)
    model, cfg = load_model(checkpoint, device)

    dataset = FolderModelDataset(vti_files, cfg)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
    )

    rows, summary = evaluate_folder_models(model, cfg, loader, device, output_dir)
    write_summary(output_dir, rows + [summary])

    print("\nValidation-style summary:")
    print(f"  num_cases={summary['num_cases']}")
    print(f"  mean_loss={summary['mean_loss']:.6f}")
    print(f"  mean_deep_iou={summary['mean_deep_iou']:.4f}")
    print(f"  mean_relative_error={summary['mean_relative_error']:.6f}")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
