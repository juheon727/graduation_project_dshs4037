import os
import yaml
import re
import numpy as np
import cv2
from typing import List, Dict
from pycocotools.coco import COCO

class FewShotKeypointTask:
    def __init__(self,
                 dataset_path: str,
                 support_imgIds: List[int],
                 query_imgIds: List[int],
                 keypoint_subset: List[int]) -> None:
        self.support_imgIds = support_imgIds
        self.query_imgIds = query_imgIds
        self.keypoint_subset = keypoint_subset
        self.dataset_path = dataset_path
        self.coco = COCO(os.path.join(dataset_path, 'labels.json'))

    def __str__(self) -> str:
        return (f"FewShotKeypointTask(\n"
                f"  Dataset Path: {self.dataset_path}\n"
                f"  Support Images: {len(self.support_imgIds)}\n"
                f"  Query Images: {len(self.query_imgIds)}\n"
                f"  Keypoints Used: {len(self.keypoint_subset)}\n"
                f")")

    def get_images(self, support: bool = True) -> List[np.ndarray]:
        imgIds = self.support_imgIds if support else self.query_imgIds
        img_metadatas = self.coco.loadImgs(imgIds)
        
        ret = []
        for metadata in img_metadatas:
            path = os.path.join(self.dataset_path, metadata['file_name'])
            img = cv2.imread(path)
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
                
                keypoints_for_this_image[ann['category_id']] = ann['keypoints'][:2]
            
            ret.append(keypoints_for_this_image)
        
        return ret

class FSKeypointDatasetBase:
    def __init__(self, 
                path: str, 
                n_shot: int = 5, 
                n_query: int = 5,
                use_keypoint_subsets: int = -1) -> None:
        self.path = path
        self.n_shot = n_shot
        self.n_query = n_query
        self.use_keypoint_subsets = use_keypoint_subsets

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

        return FewShotKeypointTask(
            dataset_path=dataset_path,
            support_imgIds=support_imgIds,
            query_imgIds=query_imgIds,
            keypoint_subset=keypoint_subset
        )

if __name__ == '__main__':
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)
        config = config['train']
    
    dataset_base = FSKeypointDatasetBase(
        path=config.get(
            'dataset_dir', 
            '/home/juheon727/lets_fucking_graduate/dataset/datasetv1/'
        ),
        n_shot=10,
        n_query=5,
        use_keypoint_subsets=8
    )

    task = dataset_base.sample_random_task()

    print(dataset_base)
    print(task)