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

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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

def download_frames(url: str, n: int, dir: str, idx: int):
    yt = YouTube(url=url, on_progress_callback=on_progress)
    safe_title = "".join(c for c in yt.title if c.isalnum() or c in (' ', '_')).rstrip()
    temp_filename = f"{safe_title}_temp.mp4"
    ys = yt.streams.filter(progressive=False).order_by('resolution').last()
    assert type(ys) == pytubefix.Stream
    video_path = ys.download(filename=temp_filename)
    assert type(video_path) == str

    codec_info = ys.video_codec.lower() if ys.video_codec else ""
    if "av01" in codec_info or "av1" in codec_info:
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
        video_path = converted_path

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames / n

    sampled_count = 0
    
    sims = []
    frames = []

    default_dir = os.path.join(dir, f'{idx:03d}')
    goal_dir = os.path.join(dir, f'{idx:03d}g')
    print(default_dir)
    print(goal_dir)
    if not os.path.exists(default_dir):
        os.mkdir(default_dir)
    if not os.path.exists(goal_dir):
        os.mkdir(goal_dir)

    for i in range(n):
        target_frame_index = min(round(i * frame_step), total_frames - 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        ret, frame = cap.read()
        if ret:
            '''blur = blur_score(frame)
            if blur < 150:
                continue'''
            score = compute_clip_similarity(frame, prompt="a football field with a goalpost")
            sims.append(score)
            frames.append(frame)
            
            sampled_count += 1

    thresh = np.percentile(np.array(sims), q=30)
    for i in range(len(sims)):
        if sims[i] > thresh:
            cv2.imwrite(filename=os.path.join(goal_dir, f'{i:03d}.jpg'), img=frames[i])
        else:
            cv2.imwrite(filename=os.path.join(default_dir, f'{i:03d}.jpg'), img=frames[i])

    cap.release()
    os.remove(video_path)

if __name__ == '__main__':
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
        config = config['dataset']

    urls = config["urls"]
    dir = config.get("output_dir", "/home/juheon727/lets_fucking_graduate/dataset/episodes2/")
    n_frames = config.get("num_frames", 200)

    for i, url in tqdm.tqdm(enumerate(urls), desc="Downloading Video Clips", total=len(urls)):
        try:
            download_frames(url=url, n=n_frames, dir=dir, idx=i)
        except Exception as e:
            print(f"Failed to process {i}th video. Error: {e}")
