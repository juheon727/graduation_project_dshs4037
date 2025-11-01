import pytubefix
from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
import cv2
import tqdm
import subprocess
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import yaml
from ultralytics import YOLO, SAM
from typing import Any, List, Tuple

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
yolo_model = YOLO(model="yolov8s-world.pt")
yolo_model_person = YOLO(model="yolov8s-world.pt")
sam_model = SAM(model="sam_l.pt")

def compute_clip_similarity(img: np.ndarray, prompt: str = "center circle of soccer field") -> float:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    inputs = clip_processor(text=[prompt], images=img_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        similarity = (image_embeds @ text_embeds.T).item()
    return similarity

def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def find_goalposts(frame: np.ndarray) -> Any:
    results = yolo_model.predict(frame, conf=0.05, imgsz=960)
    return results[0]

def convert_to_av1(video_path: str, safe_title: str) -> str:
    converted_path = f"{safe_title}_converted.mp4"
    print(f"Converting AV1 video '{safe_title}' to MP4")
    subprocess.run([
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-b:v", "5M",
        "-c:a", "aac",
        "-movflags", "+faststart",
        converted_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(video_path)
    return converted_path

def clip_filter(cap: cv2.VideoCapture, frame_step: float, total_frames: int, clip_prompt: str, n: int, lap_var_thresh: float) -> Tuple[List, List]:
    sims = []
    frames = []
    for i in range(n):
        target_frame_index = min(round(i * frame_step), total_frames - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        ret, frame = cap.read()
        if ret:
            blur = blur_score(frame)
            if blur > lap_var_thresh:
                continue
            score = compute_clip_similarity(frame, prompt=clip_prompt)
            sims.append(score)
            frames.append(frame)
            
    return sims, frames

def hsv_threshold(img: np.ndarray, bbox: List[int], lower_bound: np.ndarray, upper_bound: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox[:4]
    roi = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    uniform_color = np.zeros_like(roi, dtype=np.uint8)
    uniform_color[:] = (0, 255, 0)
    colored_mask = cv2.bitwise_and(uniform_color, uniform_color, mask=mask)
    overlay = img.copy()
    overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0, colored_mask, 0.5, 0)
    return mask, overlay

def annotate(img: np.ndarray, n_points: int = 10) -> np.ndarray:
    """
    Annotates an image by:
    1. Finding goalposts with YOLO.
    2. Creating a positive (white) and negative (non-white) mask using HSV in the bounding box.
    3. Sampling N positive and N negative points from these masks.
    4. Feeding these points as prompts to SAM.
    5. Overlaying the resulting SAM mask and points on the image.
    """
    results = find_goalposts(frame=img).cpu().numpy()

    overlay_img = img.copy()

    cv2.putText(
        img=overlay_img,
        text=f"Blur: {blur_score(img):.2f}",
        org=(10, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.5,
        color=(0, 0, 255),
        thickness=3,
        lineType=cv2.LINE_AA
    )
    
    h, w = img.shape[:-1]
    
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = np.reshape(box.xyxy, (-1,)).astype(np.int32)
            dx, dy = x2 - x1, y2 - y1
            x0, y0 = max(x1 - dx // 8, 0), max(y1 - dy // 8, 0)
            x3, y3 = min(x2 + dx // 8, w - 1), min(y2 + dy // 8, h - 1)

            cv2.rectangle(overlay_img, (x0, y0), (x3, y3), (255, 0, 0), 2)

            lower_bound = np.array([0, 0, 200])
            upper_bound = np.array([255, 50, 255])

            positive_mask_roi, _ = hsv_threshold(img=img, bbox=[x0, y0, x3, y3], lower_bound=lower_bound, upper_bound=upper_bound)

            pos_coords_roi = np.argwhere(positive_mask_roi == 255)
            neg_coords_roi = np.argwhere(positive_mask_roi == 0)

            if len(pos_coords_roi) < n_points or len(neg_coords_roi) < n_points:
                print("Not enough points to sample from HSV mask, skipping SAM for this box.")
                continue

            pos_indices = np.random.choice(len(pos_coords_roi), n_points, replace=False)
            neg_indices = np.random.choice(len(neg_coords_roi), n_points, replace=False)
            
            sampled_pos_roi = pos_coords_roi[pos_indices] # (N, 2) -> (y, x)
            sampled_neg_roi = neg_coords_roi[neg_indices] # (N, 2) -> (y, x)

            sampled_pos_abs = sampled_pos_roi[:, ::-1] + [x0, y0] # (N, 2) -> (x, y)
            sampled_neg_abs = sampled_neg_roi[:, ::-1] + [x0, y0] # (N, 2) -> (x, y)

            points = np.concatenate((sampled_pos_abs, sampled_neg_abs), axis=0).tolist()
            labels = [1] * n_points + [0] * n_points

            sam_results = sam_model(img, points=[points], labels=[labels])[0].cpu()
            if sam_results.masks is not None and hasattr(sam_results.masks, 'xy'):
                masks = sam_results.masks.xy
                sam_mask_overlay = np.zeros_like(img, dtype=np.uint8)
                polygons = [np.round(poly).astype(np.int32).reshape(-1, 1, 2) for poly in masks]
                cv2.fillPoly(img=sam_mask_overlay, pts=polygons, color=(0, 255, 0), lineType=cv2.LINE_AA)

                overlay_img = cv2.addWeighted(overlay_img, 1.0, sam_mask_overlay, 0.5, 0)

            for (x, y) in sampled_pos_abs:
                cv2.circle(overlay_img, (x, y), 5, (0, 255, 0), -1) # Green for positive
            for (x, y) in sampled_neg_abs:
                cv2.circle(overlay_img, (x, y), 5, (0, 0, 255), -1) # Red for negative

    overlay_img = locate_field(img, overlay_img, n_points=40, k=5, q=25)

    return overlay_img

def locate_field(img: np.ndarray, base_overlay: np.ndarray, n_points: int = 10, k: int = 10, q: int = 0) -> np.ndarray:
    """
    Select k prompt points nearest to the person center in (x,y), then add q more points from the same grid
    that are closest (by L2 distance in RGB color space) to the mean color of the initial k points.
    """
    # 1) Person detection and center estimation (unchanged logic)
    results_yolo = yolo_model_person.predict(img, conf=0.1)[0].cpu().numpy()
    x_coords, y_coords = [], []
    for box in results_yolo.boxes:
        x1, y1, x2, y2 = np.reshape(box.xyxy, (-1,))
        x_coords.append((x1 + x2) / 2)
        y_coords.append(y2)
    x_c = np.array(x_coords).mean() if x_coords else img.shape[1] / 2.0
    y_c = np.array(y_coords).mean() if y_coords else img.shape[0] / 2.0

    # 2) Build a uniform grid over the image
    h, w = img.shape[:2]
    xs = np.linspace(0, w - 1, n_points, dtype=np.float32)
    ys = np.linspace(0, h - 1, n_points, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (n_points, n_points)
    grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (N, 2) with (x, y)
    N = grid_pts.shape[0]

    # 3) Initial k by geometric proximity to the detected center
    d2 = (grid_pts[:, 0] - x_c) ** 2 + (grid_pts[:, 1] - y_c) ** 2
    k_clamped = int(min(max(k, 0), N))
    idx_k = np.argpartition(d2, k_clamped - 1)[:k_clamped]
    idx_k = idx_k[np.argsort(d2[idx_k])]

    # 4) Compute mean color of the initial k points (nearest-neighbor sampling in RGB)
    #    Prepare per-grid-point colors up front for efficiency
    ys_i = np.clip(np.rint(grid_pts[:, 1]).astype(int), 0, h - 1)
    xs_i = np.clip(np.rint(grid_pts[:, 0]).astype(int), 0, w - 1)
    colors = img[ys_i, xs_i, :].astype(np.float32)  # (N, 3)

    mu_color = colors[idx_k].mean(axis=0) if k_clamped > 0 else colors.mean(axis=0)

    # 5) Among remaining grid points, pick q closest to mu_color by L2 distance in RGB
    mask_remaining = np.ones(N, dtype=bool)
    mask_remaining[idx_k] = False
    remaining_idxs = np.nonzero(mask_remaining)[0]
    q_clamped = int(min(max(q, 0), remaining_idxs.size))

    if q_clamped > 0:
        diff = colors[remaining_idxs] - mu_color[None, :]
        dcolor2 = np.einsum('ij,ij->i', diff, diff)  # squared L2 distance
        sel = np.argpartition(dcolor2, q_clamped - 1)[:q_clamped]
        idx_q = remaining_idxs[sel[np.argsort(dcolor2[sel])]]
    else:
        idx_q = np.array([], dtype=int)

    # 6) Concatenate prompts: initial k by geometry + q by color proximity to mean
    idx_all = np.concatenate([idx_k, idx_q], axis=0)
    prompt_points = grid_pts[idx_all]  # (k+q, 2)
    prompt_labels = np.ones(prompt_points.shape[0], dtype=np.int32)

    # 7) Run SAM
    results_sam = sam_model.predict(img, points=prompt_points, labels=prompt_labels)[0].cpu()

    # Initialize overlay safely whether masks exist or not
    overlay_img = base_overlay.copy()

    if getattr(results_sam, "masks", None) is not None and hasattr(results_sam.masks, "xy"):
        masks = results_sam.masks.xy
        sam_mask_overlay = np.zeros_like(img, dtype=np.uint8)
        polygons = [np.round(poly).astype(np.int32).reshape(-1, 1, 2) for poly in masks]
        cv2.fillPoly(img=sam_mask_overlay, pts=polygons, color=(255, 0, 0), lineType=cv2.LINE_AA)

        # Morphological close to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        sam_mask_overlay = cv2.morphologyEx(sam_mask_overlay, cv2.MORPH_CLOSE, kernel)

        overlay_img = cv2.addWeighted(base_overlay, 1.0, sam_mask_overlay, 0.5, 0)

    # 8) Draw visualization:
    #    - All grid points in red
    #    - Initial k in blue
    #    - Added q in green
    for x, y in grid_pts:
        cv2.circle(overlay_img, (int(x), int(y)), 5, (0, 0, 255), -1)

    for x, y in grid_pts[idx_k]:
        cv2.circle(overlay_img, (int(x), int(y)), 5, (255, 0, 0), -1)

    if idx_q.size > 0:
        for x, y in grid_pts[idx_q]:
            cv2.circle(overlay_img, (int(x), int(y)), 5, (0, 255, 0), -1)

    return overlay_img


def download_frames(url: str, n: int, dir: str, idx: int, lap_var_thresh: float, clip_prompt: str, n_points: int) -> None:
    yt = YouTube(url=url, on_progress_callback=on_progress)
    safe_title = "".join(c for c in yt.title if c.isalnum() or c in (' ', '_')).rstrip()
    temp_filename = f"{safe_title}_temp.mp4"
    ys = yt.streams.filter(progressive=False).order_by('resolution').last()
    assert type(ys) == pytubefix.Stream
    video_path = ys.download(filename=temp_filename)
    assert type(video_path) == str

    codec_info = ys.video_codec.lower() if ys.video_codec else ""
    if "av01" in codec_info or "av1" in codec_info:
        video_path = convert_to_av1(video_path=video_path, safe_title=safe_title)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames / n

    dir = os.path.join(dir, f'{idx:03d}')
    print(dir)
    if not os.path.exists(dir):
        os.mkdir(dir)

    sims, frames = clip_filter(cap=cap, frame_step=frame_step, total_frames=total_frames, clip_prompt=clip_prompt, n=n, lap_var_thresh=lap_var_thresh)
    thresh = np.percentile(np.array(sims), q=80)
    
    for i in range(len(sims)):
        if sims[i] > thresh:
            annotated_results = annotate(img=frames[i])
            cv2.imwrite(filename=os.path.join(dir, f'{i:03d}.jpg'), img=annotated_results)

    cap.release()
    os.remove(video_path)

if __name__ == '__main__':
    with open("./config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        cfg = cfg['pseudodataset']

    urls_dir = cfg["urls"]
    with open(urls_dir, "r") as f:
        urls = yaml.safe_load(f)

    yolo_model.set_classes(cfg.get("yolo_prompt", ["a white football goalpost with a net on a soccer field"]))
    yolo_model_person.set_classes(["person"])
    dir = cfg.get("output_dir", "/home/juheon727/lets_fucking_graduate/dataset/episodes2/")
    n_frames = cfg.get("num_frames", 200)
    lap_var_thresh = float(cfg.get("lap_var_thresh", 1000))
    clip_prompt = cfg.get("clip_prompt", "a scene of a football game")
    n_prompt_points = cfg.get("n_prompt_points", 5)

    for i, url in tqdm.tqdm(enumerate(urls), desc="Downloading Video Clips", total=len(urls)):
        try:
            download_frames(url=url, n=n_frames, dir=dir, idx=i, lap_var_thresh=lap_var_thresh, clip_prompt=clip_prompt, n_points=n_prompt_points)
        except Exception as e:
            print(f"Failed to process {i}th video. Error: {e}")
