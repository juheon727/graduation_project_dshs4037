import os
import argparse
import yaml
import numpy as np
import cv2
import torch

from train.dataset import FSKeypointDataset, Collator


def _to_uint8_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    img_tensor: shape (3, H, W), dtype float or uint8-like, range assumed 0..255 if uint8 else 0..1/255
    returns: np.uint8 array (H, W, 3) in RGB order.
    """
    img = img_tensor.detach().cpu().float()
    # The dataset code returns raw images stacked from uint8 numpy arrays; after torch.tensor they
    # will be dtype=torch.uint8. Handle both gracefully and normalize to [0,255].
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.clamp(0, 255).byte()
    img_np = img.permute(1, 2, 0).contiguous().numpy()  # (H, W, 3) RGB
    return img_np


def _composite_heatmap(hm_tensor: torch.Tensor) -> np.ndarray:
    """
    hm_tensor: shape (K, H, W), float tensor with arbitrary positive values.
    returns: np.uint8 heatmap image (H, W) scaled 0..255.
    """
    if hm_tensor.numel() == 0:
        # No keypoints present for this sample; return zeros.
        h, w = hm_tensor.shape[-2], hm_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.uint8)

    hm = hm_tensor.detach().cpu().float()
    # Combine channels by max to highlight any keypoint response.
    hm = torch.max(hm, dim=0, keepdim=False).values  # (H, W)
    # Normalize robustly to [0, 255]
    hm_min = float(hm.min())
    hm_max = float(hm.max())
    if hm_max <= hm_min + 1e-6:
        scaled = torch.zeros_like(hm)
    else:
        scaled = (hm - hm_min) / (hm_max - hm_min)
    hm_uint8 = (scaled * 255.0).clamp(0, 255).byte().numpy()
    return hm_uint8


def _overlay(img_rgb_uint8: np.ndarray, hm_uint8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend RGB image with a colorized heatmap.
    """
    # OpenCV expects BGR for color maps; convert for colorizing, then convert back.
    color_hm_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    color_hm_rgb = cv2.cvtColor(color_hm_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb_uint8, 1.0 - alpha, color_hm_rgb, alpha, 0)
    return overlay


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_overlays_for_batch(
    batch,
    outdir: str,
    n_samples_support: int = 8,
    n_samples_query: int = 8,
) -> None:
    """
    Persists example overlays for support and query images from a single collated batch.
    The batch conforms to the dictionary format produced by Collator.__call__.
    """
    _ensure_dir(outdir)
    supp_imgs = batch["support_images"]  # shape (B*n_shot, 3, H, W)
    qry_imgs = batch["query_images"]     # shape (B*n_query, 3, H, W)
    supp_hms_list = batch["support_heatmaps"]  # list length B; each (n_shot, K_i, H, W)
    qry_hms_list = batch["query_heatmaps"]     # list length B; each (n_query, K_i, H, W)
    task_indices = batch["task_indices"]       # list length B

    # Determine counts per episode from heatmap tensors to index into the flat image tensors.
    episode_counts_support = [hms.shape[0] for hms in supp_hms_list]
    episode_counts_query = [hms.shape[0] for hms in qry_hms_list]

    # Running offsets into the flattened image tensors
    supp_offset = 0
    qry_offset = 0

    saved_s = 0
    saved_q = 0

    for epi_idx, task_id in enumerate(task_indices):
        # Support samples for this episode
        n_s = episode_counts_support[epi_idx]
        for s in range(n_s):
            if saved_s >= n_samples_support:
                break
            img = _to_uint8_rgb(supp_imgs[supp_offset + s])
            hm = _composite_heatmap(supp_hms_list[epi_idx][s])
            overlay = _overlay(img, hm, alpha=0.45)
            fname = os.path.join(outdir, f"support_task{task_id}_e{epi_idx}_s{s}.png")
            # cv2.imwrite expects BGR
            cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved_s += 1
        supp_offset += n_s

        # Query samples for this episode
        n_q = episode_counts_query[epi_idx]
        for q in range(n_q):
            if saved_q >= n_samples_query:
                break
            img = _to_uint8_rgb(qry_imgs[qry_offset + q])
            hm = _composite_heatmap(qry_hms_list[epi_idx][q])
            overlay = _overlay(img, hm, alpha=0.45)
            fname = os.path.join(outdir, f"query_task{task_id}_e{epi_idx}_q{q}.png")
            cv2.imwrite(fname, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved_q += 1
        qry_offset += n_q

        if saved_s >= n_samples_support and saved_q >= n_samples_query:
            break


def main():
    parser = argparse.ArgumentParser(description="Create sample heatmap overlays for a batch.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--outdir", type=str, default="./visualizations", help="Directory to save overlays")
    parser.add_argument("--num-support", type=int, default=8, help="Max support overlays to save")
    parser.add_argument("--num-query", type=int, default=8, help="Max query overlays to save")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Seed for reproducibility of the sampled task and dataloader shuffling.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load training config for dataset_dir override
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg_train = cfg.get("train", {})
    else:
        cfg_train = {}

    n_shot = cfg_train.get("n_shot", 10)
    n_query = cfg_train.get("n_query", 3)
    batch_size = cfg_train.get("batch_size", 16)
    use_keypoint_subsets = cfg_train.get("use_keypoint_subsets", -1)
    resolution = cfg_train.get("resolution", [448, 448])
    sigma = cfg_train.get("heatmap_sigma", 5.0)

    dataset_dir = cfg_train.get(
        "dataset_dir",
        "/home/juheon727/lets_fucking_graduate/dataset/datasetv1/",
    )

    w, h = int(resolution[0]), int(resolution[1])

    dataset = FSKeypointDataset(
        path=dataset_dir,
        epoch_length=1000,
        n_shot=n_shot,
        n_query=n_query,
        use_keypoint_subsets=use_keypoint_subsets,
        resolution=(w, h),
    )

    collate = Collator(
        resolution=(w, h),
        sigma=float(sigma),
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        collate_fn=collate,
        shuffle=True,
        num_workers=0,  # keep simple/portable; increase if desired
        pin_memory=False,
    )

    # Take a single batch and render overlays
    batch = next(iter(loader))
    os.makedirs(args.outdir, exist_ok=True)
    save_overlays_for_batch(
        batch=batch,
        outdir=args.outdir,
        n_samples_support=args.num_support,
        n_samples_query=args.num_query,
    )

    print(f"Saved sample overlays to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
