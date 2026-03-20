from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.ndimage import zoom

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
    HighResVisualizer,
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


def save_case_arrays(case_dir: Path, outputs: np.ndarray, targets: np.ndarray, obs_gravity: np.ndarray, pred_gravity: np.ndarray) -> None:
    ensure_dir(case_dir)
    np.save(case_dir / "pred_density.npy", outputs)
    np.save(case_dir / "true_density.npy", targets)
    np.save(case_dir / "obs_gravity.npy", obs_gravity)
    np.save(case_dir / "pred_gravity.npy", pred_gravity)


def save_case_metrics(case_dir: Path, metrics: Dict[str, Any]) -> None:
    ensure_dir(case_dir)
    with open(case_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)


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
    visualizer = HighResVisualizer(str(output_dir), dpi=220, upsampling=2)

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
            save_case_arrays(case_dir, pred_np, true_np, obs_np, pred_grav_np)
            visualizer.save_epoch_result(case_index, obs_np, pred_grav_np, true_np, pred_np, cfg.dx, cfg.dz)

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
            save_case_metrics(case_dir, row)
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
