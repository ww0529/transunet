import torch
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class V4Config:
    exp_name: str = "Hybrid2D3D_CDL_TransNet"

    data_mode: str = 'joint'

    grid_shape: Tuple[int, int, int] = (16, 32, 32)
    dx: float = 100.0
    dz: float = 100.0

    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    decoder_channels: Tuple[int, ...] = (128, 64, 32)
    lifting_channels: int = 64
    num_transformer_layers: int = 6
    num_heads: int = 8
    use_position_encoding: bool = True
    use_depth_attention: bool = True

    lr: float = 5e-4
    batch_size: int = 8
    epochs: int = 400
    steps_per_epoch: int = 500

    w_depth: float = 1.0
    w_focus: float = 0.3
    w_gdl: float = 0.2
    w_physics: float = 0.3
    w_edge: float = 0.3
    w_l1: float = 0.001
    w_morph: float = 0.1
    w_adv: float = 0.0
    w_boundary: float = 0.3
    depth_beta: float = 2.0
    focus_beta: float = 30.0

    use_augmentation: bool = True
    elastic_alpha: float = 30.0
    elastic_sigma: float = 4.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "/home/jszxgx/ysw/deeplearn/checkpoints_v4_new"
    num_workers: int = 4
    resume_path: Optional[str] = None
