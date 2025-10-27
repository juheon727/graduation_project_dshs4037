import os
import yaml
import re
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from pycocotools.coco import COCO
import sys
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import torch
from torch.utils.data import Dataset, DataLoader

class FewShotKeypointTask:
    def __init__(self,
                 dataset_path: str,
                 coco: COCO,
                 support_imgIds: List[int],
                 query_imgIds: List[int],
                 keypoint_subset: List[int],
                 resolution: Tuple[int, int],
                 task_idx: str = "-1") -> None:
        self.support_imgIds = support_imgIds
        self.query_imgIds = query_imgIds
        self.keypoint_subset = keypoint_subset
        self.dataset_path = dataset_path
        self.resolution = resolution
        self.coco = coco
        self.task_idx = task_idx

    def __str__(self) -> str:
        return (f"FewShotKeypointTask(\n"
                f"  Dataset Path: {self.dataset_path}\n"
                f"  Support Images: {len(self.support_imgIds)}\n"
                f"  Query Images: {len(self.query_imgIds)}\n"
                f"  Keypoints Used: {len(self.keypoint_subset)}\n"
                f"  Resolution: {self.resolution}\n"
                f")")

    def get_images(self, support: bool = True) -> List[np.ndarray]:
        imgIds = self.support_imgIds if support else self.query_imgIds
        img_metadatas = self.coco.loadImgs(imgIds)
        
        ret = []
        for metadata in img_metadatas:
            path = os.path.join(self.dataset_path, metadata['file_name'])
            img = cv2.imread(path)
            img = cv2.resize(img, self.resolution)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ret.append(img)
        
        return ret
    
    def get_keypoints(self, support: bool = True) -> List[Dict[int, np.ndarray]]:
        """
        Retrieves the filtered keypoints for images of either support or query.

        Args:
            support (bool): If True, returns support keypoints, else, returns query keypoints
        
        Returns:
            A list where each element corresponds to a support image.
            Each element is a dictionary mapping {category_id: keypoints}.
            'keypoints' is a np.ndarray of shape (K, 2), where K is the
            length of self.keypoint_subset. The columns are (x, y).
        """
        imgIds = self.support_imgIds if support else self.query_imgIds
        
        ret = []
        for img_id in imgIds:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            keypoints_for_this_image = {}
            for ann in anns:
                if ann['category_id'] not in self.keypoint_subset or 'keypoints' not in ann.keys():
                    continue
                
                img_id = ann['image_id']
                w_0 = self.coco.imgs[img_id]['width']
                h_0 = self.coco.imgs[img_id]['height']
                w_1 = self.resolution[0]
                h_1 = self.resolution[1]
                keypoints_for_this_image[ann['category_id']] = ann['keypoints'][:2] * np.array([w_1 / w_0, h_1 / h_0])
            
            ret.append(keypoints_for_this_image)
        
        return ret

class FSKeypointDatasetBase:
    def __init__(self, 
                path: str, 
                n_shot: int = 5, 
                n_query: int = 5,
                use_keypoint_subsets: int = -1,
                resolution: Tuple[int, int] = (1920, 1080)) -> None:
        self.path = path
        self.n_shot = n_shot
        self.n_query = n_query
        self.use_keypoint_subsets = use_keypoint_subsets
        self.resolution = resolution

        subdirectories = os.listdir(self.path)
        self.episode_numbers = [re.sub(r'^([^A-Za-z]*)[A-Za-z].*', r'\1', episode) for episode in subdirectories]
        self.episode_numbers = list(set(self.episode_numbers))

        self.episodes = {}
        for episode_number in self.episode_numbers:
            self.episodes[episode_number] = []
            for subdir in subdirectories:
                if episode_number in subdir:
                    self.episodes[episode_number].append(subdir)

    def __str__(self) -> str:
        return (f"DatasetBase(\n"
                f"  Root Path: {self.path}\n"
                f"  Episodes: {len(self.episodes)} ({', '.join(self.episode_numbers)})\n"
                f"  n_shot: {self.n_shot}, n_query: {self.n_query}\n"
                f"  Keypoint Subset Mode: {'All' if self.use_keypoint_subsets == -1 else f'Min {self.use_keypoint_subsets}'}\n"
                f")")

    def sample_random_task(self) -> FewShotKeypointTask:
        episode_number = np.random.choice(self.episode_numbers)

        subdirs = self.episodes[episode_number]
        chosen_subdir = np.random.choice(subdirs)
        dataset_path = os.path.join(self.path, chosen_subdir)

        with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
            coco = COCO(os.path.join(dataset_path, 'labels.json'))
        all_img_ids = coco.getImgIds()
        np.random.shuffle(all_img_ids)

        n_support = self.n_shot
        n_query = self.n_query
        support_imgIds = all_img_ids[:n_support]
        query_imgIds = all_img_ids[n_support:n_support + n_query]

        all_keypoints = list(range(1, len(coco.loadCats(coco.getCatIds())) + 1))
        if self.use_keypoint_subsets == -1:
            keypoint_subset = all_keypoints
        else:
            subset_size = np.random.randint(min(self.use_keypoint_subsets, len(all_keypoints)), len(all_keypoints) + 1)
            keypoint_subset = sorted(np.random.choice(all_keypoints, subset_size, replace=False))

        with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
            coco = COCO(os.path.join(dataset_path, 'labels.json'))

        return FewShotKeypointTask(
            dataset_path=dataset_path,
            coco=coco,
            support_imgIds=support_imgIds,
            query_imgIds=query_imgIds,
            keypoint_subset=keypoint_subset,
            resolution=self.resolution,
            task_idx=episode_number,
        )
    
class FSKeypointDataset(FSKeypointDatasetBase, Dataset):
    def __init__(self,
                 path: str,
                 epoch_length: int = 1000,
                 n_shot: int = 5,
                 n_query: int = 5,
                 use_keypoint_subsets: int = -1,
                 resolution: Tuple[int, int] = (1920, 1080)) -> None:
        
        super().__init__(
            path=path,
            n_shot=n_shot,
            n_query=n_query,
            use_keypoint_subsets=use_keypoint_subsets,
            resolution=resolution
        )

        self.epoch_length = epoch_length

    def __len__(self) -> int:
        return self.epoch_length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns one complete few-shot task.
        
        Note: The 'idx' is ignored, as a new random task is sampled every time.
        
        Returns:
            A dictionary containing the task data:
            {
                'support_images': List[np.ndarray],
                'support_keypoints': List[Dict[int, np.ndarray]],
                'query_images': List[np.ndarray],
                'query_keypoints': List[Dict[int, np.ndarray]],
                'keypoint_subset': List[int] (List of category IDs used),
                'task_idx': str,
            }
        """
        task = self.sample_random_task()

        support_images = task.get_images(support=True)
        support_keypoints = task.get_keypoints(support=True)
        query_images = task.get_images(support=False)
        query_keypoints = task.get_keypoints(support=False)
        
        return {
            'support_images': support_images,
            'support_keypoints': support_keypoints,
            'query_images': query_images,
            'query_keypoints': query_keypoints,
            'keypoint_subset': task.keypoint_subset,
            'task_idx': task.task_idx
        }
    
class Collator:
    def __init__(self, resolution: Tuple[int, int], sigma: float) -> None:
        """
        Initializer for the Collator class.
        Args:
            resolution (Tuple[int, int]): Desired image resolution as (width, height).
            sigma (float): Standard deviation for Gaussian heatmap generation.

        Returns:
            None
        """
        self.resolution = resolution
        self.sigma = sigma

    def heatmap_from_coords(self, keypoints: Dict[int, np.ndarray]) -> np.ndarray:
        w, h = self.resolution
        #print(keypoints)
        channels = []
        for idx, coords in keypoints.items():
            gauss_x = np.arange(w).reshape(1, w) - coords[0]
            gauss_x = np.exp(-(gauss_x ** 2) / (2 * self.sigma ** 2))

            gauss_y = np.arange(h).reshape(h, 1) - coords[1]
            gauss_y = np.exp(-(gauss_y ** 2) / (2 * self.sigma ** 2))

            heatmap = gauss_x * gauss_y
            channels.append(heatmap)

        return np.stack(channels, axis=0)

    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collation function for batching few-shot keypoint data. Used for Pytorch DataLoader.

        Args:
            data (List[Dict[str, Any]]): A list of few-shot task dictionaries as returned by FSKeypointDataset.__getitem__.

        Returns:
            A dictionary containing batched tensors:
            {
                'support_images': Tensor of shape (batch_size * n_shot, 3, H, W),
                'support_heatmaps': List of tensors of shape (n_shot, K_i, H, W),
                'query_images': Tensor of shape (batch_size * n_query, 3, H, W),
                'query_heatmaps': List of tensors of shape (n_query, K_i, H, W),
                'n_keypoints': List of number of keypoints for each task,
                'task_indices': List of task indices,
            }
        """
        support_images = []
        support_heatmaps = []
        query_images = []
        query_heatmaps = []
        n_keypoints = []
        task_indices = []
        for episode in data:
            #print(episode['task_idx'])
            support_images_sample = episode['support_images'] # List of np.ndarray of shape (H, W, 3)
            support_images.extend(support_images_sample)

            query_images_sample = episode['query_images'] # List of np.ndarray of shape (H, W, 3)
            query_images.extend(query_images_sample)

            support_heatmaps_episode = []
            for support_kp in episode['support_keypoints']:
                heatmap = self.heatmap_from_coords(keypoints=support_kp)
                support_heatmaps_episode.append(heatmap)
                #print(heatmap.shape)
            support_heatmaps.append(torch.tensor(np.stack(support_heatmaps_episode, axis=0)))

            query_heatmaps_episode = []
            for query_kp in episode['query_keypoints']:
                heatmap = self.heatmap_from_coords(keypoints=query_kp)
                query_heatmaps_episode.append(heatmap)
            query_heatmaps.append(torch.tensor(np.stack(query_heatmaps_episode, axis=0)))

            n_keypoints.append(len(episode['keypoint_subset']))

            task_indices.append(episode['task_idx'])

        support_images = torch.tensor(np.stack(support_images, axis=0)).permute(0, 3, 1, 2)  # (B*n_shot, 3, H, W)
        support_images = support_images / 255.0
        query_images = torch.tensor(np.stack(query_images, axis=0)).permute(0, 3, 1, 2)      # (B*n_query, 3, H, W)
        query_images = query_images / 255.0

        return {
            'support_images': support_images,
            'support_heatmaps': support_heatmaps,
            'query_images': query_images,
            'query_heatmaps': query_heatmaps,
            'n_keypoints': n_keypoints,
            'task_indices': task_indices,
        }

if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
        config = config['train']

    dataset = FSKeypointDataset(
        path=config.get(
            'dataset_dir', 
            '/home/juheon727/lets_fucking_graduate/dataset/datasetv1/'
        ),
        n_shot=10,
        n_query=3,
        use_keypoint_subsets=-1,
        resolution=(448, 448),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        collate_fn=Collator(
            resolution=(448, 448),
            sigma=8.0
        ),
        shuffle=True,
    )

    for idx, batch in enumerate(dataloader):
        print(f"Batch {idx}:")
        print(f"  Task Indices: {batch['task_indices']}")
        print(f"  Number of Tasks: {len(batch['task_indices'])}")
        print(f"  Support Images Shape: {batch['support_images'].shape}")
        print(f"  Query Images Shape: {batch['query_images'].shape}")
        print(f"  Support Heatmaps Length: {len(batch['support_heatmaps'])}")
        print(f"  Query Heatmaps Length: {len(batch['query_heatmaps'])}")
        
        break