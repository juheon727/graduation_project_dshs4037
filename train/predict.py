import os
import argparse
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Import modules from your codebase
from train.model import LitModule, MultiRegHead
from train.dataset import FSKeypointDataset, Collator
from train.train import load_model
from train.visualize_dataset import _to_uint8_rgb, _composite_heatmap, _overlay, _ensure_dir

'''
def _get_keypoints_from_heatmaps(hm_tensor: torch.Tensor) -> np.ndarray:
    """
    Calculates the centroid (weighted average) for each heatmap in the stack.
    
    Args:
        hm_tensor: (K, H, W) tensor of heatmaps.
    Returns:
        np.ndarray of shape (K, 2) with (x, y) coordinates.
    """
    if hm_tensor.numel() == 0:
        return np.empty((0, 2), dtype=int)
        
    # Ensure tensor is on CPU and float for calculations
    hm_tensor = hm_tensor.detach().cpu().float()
    k, h, w = hm_tensor.shape
    
    # Create coordinate grids
    # y_coords: (1, H, 1)
    y_coords = torch.arange(h, dtype=hm_tensor.dtype, device=hm_tensor.device).view(1, h, 1)
    # x_coords: (1, 1, W)
    x_coords = torch.arange(w, dtype=hm_tensor.dtype, device=hm_tensor.device).view(1, 1, w)
    
    # Calculate total mass for each keypoint
    # total_mass: (K,)
    total_mass = hm_tensor.sum(dim=[1, 2])
    
    # Calculate weighted sums
    # (K, H, W) * (1, H, 1) -> sum over H, W -> (K,)
    centroid_y = (hm_tensor * y_coords).sum(dim=[1, 2])
    # (K, H, W) * (1, 1, W) -> sum over H, W -> (K,)
    centroid_x = (hm_tensor * x_coords).sum(dim=[1, 2])
    
    # Avoid division by zero for heatmaps with no mass
    safe_mass = total_mass + 1e-6
    
    centroid_y = (centroid_y / safe_mass)
    centroid_x = (centroid_x / safe_mass)
    
    # Stack to (K, 2) and convert to numpy int for drawing
    kps = torch.stack([centroid_x, centroid_y], dim=1).round().long().numpy()
    
    return kps
'''

def _get_keypoints_from_heatmaps(hm_tensor: torch.Tensor) -> np.ndarray:
    """
    Calculates the peak (argmax) for each heatmap in the stack.
    
    Args:
        hm_tensor: (K, H, W) tensor of heatmaps.
    Returns:
        np.ndarray of shape (K, 2) with (x, y) coordinates.
    """
    if hm_tensor.numel() == 0:
        return np.empty((0, 2), dtype=int)
        
    # Ensure tensor is on CPU
    hm_tensor = hm_tensor.detach().cpu()
    k, h, w = hm_tensor.shape
    
    # Flatten the H, W dimensions to find the argmax easily
    # hm_flat: (K, H*W)
    hm_flat = hm_tensor.view(k, -1)
    
    # Find the flat index of the maximum value for each keypoint
    # flat_indices: (K,)
    flat_indices = torch.argmax(hm_flat, dim=1)
    
    # Convert flat indices back to (y, x) coordinates
    y_coords = flat_indices // w
    x_coords = flat_indices % w
    
    # Stack to (K, 2) and convert to numpy int for drawing
    # Note the order: (x, y)
    kps = torch.stack([x_coords, y_coords], dim=1).long().numpy()
    
    return kps


# --- End of Visualization Helpers ---


def plot_pck_curve(all_distances: np.ndarray, save_path: str, max_threshold: int = 30):
    """
    Calculates and plots the PCK curve based on keypoint distances.
    
    Args:
        all_distances: A 1D numpy array of all keypoint distances (GT to Pred).
        save_path: Full path (including filename) to save the plot.
        max_threshold: The maximum pixel distance to plot on the x-axis.
    """
    if all_distances.size == 0:
        print("Warning: No distances found. Skipping PCK plot generation.")
        return

    thresholds = np.arange(0, max_threshold + 1, 1)
    pck_values = []

    for t in thresholds:
        correct_kps = np.sum(all_distances <= t)
        pck = (correct_kps / len(all_distances)) * 100.0
        pck_values.append(pck)
    
    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, pck_values, marker='o', linestyle='-')
    plt.title(f'Threshold-PCK Curve (Total Keypoints: {len(all_distances)})')
    plt.xlabel('Distance Threshold (pixels)')
    plt.ylabel('PCK (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 100)
    plt.xlim(0, max_threshold)
    plt.xticks(np.arange(0, max_threshold + 1, 2))
    plt.yticks(np.arange(0, 101, 10))
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()
    print(f"PCK curve saved to {save_path}")
    
    # Print some key PCK values
    pck_at_5 = pck_values[5] if len(pck_values) > 5 else pck_values[-1]
    pck_at_10 = pck_values[10] if len(pck_values) > 10 else pck_values[-1]
    print(f"Key Metrics: PCK@5px = {pck_at_5:.2f}%, PCK@10px = {pck_at_10:.2f}%")


def save_predictions(
    batch,
    query_preds_list: list,
    outdir: str,
    max_samples: int,
    save_visuals: bool = True # NEW: Flag to control saving images
) -> np.ndarray:
    """
    Calculates keypoint distances for ALL samples in the batch for PCK.
    If save_visuals is True, also saves side-by-side overlays for query images.
    
    Args:
        batch: The raw batch from the dataloader (CPU tensors).
        query_preds_list: List of prediction tensors from the model (GPU/CPU tensors).
        outdir: Directory to save images.
        max_samples: Maximum number of query images to save.
        save_visuals: If True, save visualization images.
        
    Returns:
        np.ndarray: A 1D array of all keypoint distances.
    """
    if save_visuals:
        _ensure_dir(outdir)
        
    # Get data from the original CPU batch
    qry_imgs_tensor = batch["query_images"]      # (B*n_query, 3, H, W)
    qry_hms_gt_list = batch["query_heatmaps"] # list[B] of (n_query, K_i, H, W)
    task_indices = batch["task_indices"]         # list[B]

    episode_counts_query = [hms.shape[0] for hms in qry_hms_gt_list]
    qry_offset = 0
    saved_q = 0
    all_distances = [] # List to store all keypoint distances

    if save_visuals:
        print(f"Calculating PCK and saving max {max_samples} prediction samples...")
    else:
        print(f"Calculating PCK (visuals disabled for this batch)...")


    for epi_idx, task_id in enumerate(task_indices):
        n_q = episode_counts_query[epi_idx]
        # Get the predictions for this specific episode
        query_preds_episode = query_preds_list[epi_idx] # (n_query, K_i, H, W)
        
        for q_idx in range(n_q):
            
            # --- 1. PCK Calculation (runs for all samples) ---
            hm_gt_tensor = qry_hms_gt_list[epi_idx][q_idx]
            hm_pred_tensor = query_preds_episode[q_idx] # May be on GPU

            # Get GT and Pred keypoints
            kps_gt = _get_keypoints_from_heatmaps(hm_gt_tensor)
            kps_pred = _get_keypoints_from_heatmaps(hm_pred_tensor)

            # Calculate distances if keypoints exist and shapes match
            if kps_gt.shape[0] > 0 and kps_gt.shape == kps_pred.shape:
                distances = np.linalg.norm(kps_gt - kps_pred, axis=1)
                all_distances.extend(distances)


            # --- 2. Visualization Saving (runs only if flag is True) ---
            if save_visuals and saved_q < max_samples:
                # 1. Get original image
                img_tensor = qry_imgs_tensor[qry_offset + q_idx]
                img_rgb = _to_uint8_rgb(img_tensor)

                # 2. Get Ground Truth heatmap and overlay
                hm_gt_vis = _composite_heatmap(hm_gt_tensor)
                overlay_gt = _overlay(img_rgb, hm_gt_vis)

                # 3. Get Predicted heatmap and overlay
                hm_pred_vis = _composite_heatmap(hm_pred_tensor)
                overlay_pred = _overlay(img_rgb, hm_pred_vis)

                # 4. Add text labels
                overlay_gt = cv2.putText(overlay_gt, 'Ground Truth', (10, 30), 
                                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                overlay_pred = cv2.putText(overlay_pred, 'Prediction', (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 5. Draw GT keypoints (white circles)
                # kps_gt already calculated
                for (x, y) in kps_gt:
                    cv2.circle(overlay_gt, (int(x), int(y)), radius=3, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

                # 6. Draw Pred keypoints (yellow circles)
                # kps_pred already calculated
                for (x, y) in kps_pred:
                    cv2.circle(overlay_pred, (int(x), int(y)), radius=3, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

                # 7. Combine side-by-side
                comparison_img = np.concatenate((overlay_gt, overlay_pred), axis=1) # (H, W*2, 3)

                # 8. Save
                fname = os.path.join(outdir, f"query_task{task_id}_e{epi_idx}_q{q_idx}.png")
                cv2.imwrite(fname, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))
                saved_q += 1
            
        qry_offset += n_q
    
    if save_visuals:
        print(f"Done. Saved {saved_q} visual samples.")
    else:
        print("Done. PCK calculated.")

    
    return np.array(all_distances)


def main():
    parser = argparse.ArgumentParser(description="Generate sample predictions from a trained model.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the .ckpt checkpoint file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--outdir", type=str, default="./visualizations", help="Directory to save prediction overlays")
    parser.add_argument("--dataset_split", type=str, default="val", choices=["train", "val"], help="Dataset split to use for samples ('train' or 'val')")
    parser.add_argument("--num_samples", type=int, default=8, help="Max query overlays to save (from the first batch only)")
    parser.add_argument("--pck_max_thresh", type=int, default=30, help="Maximum pixel threshold for PCK plot")
    # NEW ARGUMENT
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to accumulate PCK over.")
    args = parser.parse_args()

    # --- 1. Load Config ---
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)['train']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    print(f"Loading feature model: {cfg.get('model', 'N/A')}")
    # Must instantiate feature_model first, as it's ignored in save_hyperparameters
    feature_model = load_model(model=cfg.get('model', None), **cfg.get('model_kwargs', {}))
    feature_channels = cfg.get('feature_channels', 256) # Ensure this matches config

    print(f"Loading LitModule from checkpoint: {args.ckpt_path}")
    lightning_module = LitModule.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        # Provide the args that were ignored in save_hyperparameters
        feature_model=feature_model,
        feature_channels=feature_channels,
        # map_location ensures weights are loaded to the correct device
        map_location=device
    )
    lightning_module.to(device)
    lightning_module.eval() # Set to evaluation mode

    # --- 3. Load Data ---
    dataset_path = cfg.get(f'dataset_dir_{args.dataset_split}')
    if not dataset_path:
        print(f"Error: `dataset_dir_{args.dataset_split}` not found in config.")
        return

    print(f"Loading data from: {dataset_path}")
    dataset = FSKeypointDataset(
        path=dataset_path,
        epoch_length=cfg.get('batch_size', 1) * args.num_batches, # MODIFIED: Ensure enough samples for all batches
        n_shot=cfg.get('n_shot', 5),
        n_query=cfg.get('n_query', 5),
        use_keypoint_subsets=cfg.get('use_keypoint_subsets', -1),
        resolution=cfg.get('resolution', (224, 224)),
    )
    
    collator = Collator(
        resolution=cfg.get('resolution', (224, 224)),
        sigma=cfg.get('heatmap_sigma', 5.0)
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.get('batch_size', 1), # Use batch size from config
        collate_fn=collator,
        shuffle=True, # Shuffle to get different batches
        num_workers=0 # Simple for prediction
    )

    # --- 4. Run Prediction Loop ---
    
    # Get a data iterator
    data_iter = iter(loader)
    total_all_distances = [] # Accumulator for all distances

    print(f"\nRunning prediction and PCK accumulation for {args.num_batches} batches...")

    for batch_idx in range(args.num_batches):
        print(f"\n--- Processing Batch {batch_idx + 1} / {args.num_batches} ---")
        try:
            batch_cpu = next(data_iter)
        except StopIteration:
            print(f"Warning: DataLoader exhausted after {batch_idx} batches.")
            break # Stop if dataset runs out early

        # Move data to device for model
        support_images = batch_cpu['support_images'].to(device)
        query_images = batch_cpu['query_images'].to(device)
        support_heatmaps = [h.to(device) for h in batch_cpu['support_heatmaps']]
        query_heatmaps = [h.to(device) for h in batch_cpu['query_heatmaps']]
        n_keypoints = batch_cpu['n_keypoints']

        # Get the feature extractor from the loaded module
        feature_model = lightning_module.feature_model

        # A. Extract features (no_grad)
        with torch.no_grad():
            support_features = feature_model(support_images)
            query_features = feature_model(query_images)

        # B. Instantiate MultiRegHead
        multireg = MultiRegHead(
            in_channels=lightning_module.feature_channels,
            n_keypoints=n_keypoints
        ).to(device)

        # C. Adapt (enable_grad)
        with torch.enable_grad():
            multireg.adapt(
                support_features=support_features,
                support_heatmaps=support_heatmaps,
                optim_lr=lightning_module.hparams.adapt_lr,
                optim_steps=lightning_module.hparams.adapt_steps,
                optim_betas=lightning_module.hparams.adapt_betas,
            )

        # D. Get query predictions (no_grad)
        all_query_preds = []
        with torch.no_grad():
            query_features_split = list(torch.tensor_split(query_features, multireg.batch_size, dim=0))
            for feature, reg_head in zip(query_features_split, multireg.regressors):
                pred = reg_head(feature) # Shape: [n_query, K_i, H, W]
                n, k, h, w = pred.shape
                pred = pred.reshape(n*k, h*w)
                pred = torch.softmax(pred, dim=1)
                pred = pred.reshape(n, k, h, w)
                all_query_preds.append(pred) # List[B] of (n_query, K_i, H, W)

        # --- 5. Save Visualizations & Get PCK Data ---
        should_save_visuals = (batch_idx == 0) # Only save visuals for the first batch
        
        batch_distances = save_predictions(
            batch=batch_cpu, # Pass the original CPU batch
            query_preds_list=all_query_preds, # Pass the list of prediction tensors
            outdir=args.outdir,
            max_samples=args.num_samples,
            save_visuals=should_save_visuals # Pass the new flag
        )
        total_all_distances.extend(batch_distances)

        torch.cuda.empty_cache()

    print("\n--- All Batches Processed ---")
    
    # --- 6. Plot PCK Curve (uses accumulated data) ---
    pck_plot_path = os.path.join(args.outdir, "pck_curve.png")
    
    # Ensure output dir exists for the plot, even if no visuals were saved
    _ensure_dir(args.outdir)
    
    plot_pck_curve(
        all_distances=np.array(total_all_distances), # Use accumulated distances
        save_path=pck_plot_path,
        max_threshold=args.pck_max_thresh
    )
    
    print(f"\nSuccessfully generated predictions and PCK plot.")
    print(f"Output saved to: {os.path.abspath(args.outdir)}")


if __name__ == '__main__':
    main()