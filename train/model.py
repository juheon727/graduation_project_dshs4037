import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, Optimizer
from torch import Tensor
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from typing import Any, Tuple, List
import yaml
from train.dataset import FSKeypointDataset, Collator

def kl_loss(pred: Tensor, gt: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Calculates the KL divergence loss between prediction and ground truth (gt) heatmaps.
    Args:
        pred (Tensor): Predicted raw heatmaps by the model of shape [N, K, H, W].
        gt (Tensor): Ground truth heatmaps of shape [N, K, H, W].
        eps (float): A small float value for numerical stability.

    Returns:
        loss (Tensor): A scalar loss value.
    """
    n, k, h, w = pred.shape
    pred = pred.reshape(n*k, h*w)
    gt = gt.reshape(n*k, h*w)
    pred = F.log_softmax(pred, dim=1)
    gt = gt / gt.sum(dim=1, keepdim=True).clamp(min=eps)
    loss = F.kl_div(pred, gt, reduction='batchmean')
    return loss

class HeatmapRegressor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Last stage module, a heatmap regression head.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
    
    def adapt(self, 
              support_features: Tensor, 
              support_heatmaps: Tensor,
              optim_lr: float,
              optim_steps: int,
              optim_betas: Tuple[float, float]) -> None:
            """
            Performs adaptation on the support set by fine-tuning the regression head.
            Args:
                support_features (Tensor): Input feature maps from support images [S, C, H, W]
                support_heatmaps (Tensor): Ground truth heatmaps [S, K, H, W]
                optim_lr (float): Learning rate for adaptation
                optim_steps (int): Number of gradient update steps
                optim_betas (Tuple[float, float]): Adam optimizer betas
            """
            optimizer = Adam(params=self.parameters(), lr=optim_lr, betas=optim_betas)
            self.train()
            for _ in range(optim_steps):
                optimizer.zero_grad()
                support_pred = self(support_features)
                loss = kl_loss(support_pred, support_heatmaps)
                loss.backward(retain_graph=True)
                optimizer.step()

class MultiRegHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 n_keypoints: List[int]) -> None:
        super().__init__()
        self.batch_size = len(n_keypoints)
        self.n_keypoints = n_keypoints
        self.regressors = nn.ModuleList([HeatmapRegressor(
            in_channels=in_channels,
            out_channels=out_channels,
        ) for out_channels in n_keypoints])

    def adapt(self,
              support_features: Tensor,
              support_heatmaps: List[Tensor],
              optim_lr: float,
              optim_steps: int,
              optim_betas: Tuple[float, float]) -> None:
        """
        Performs adaptation on the support set by fine-tuning each regression head on its corresponding support set.
        Args:
            support_features (Tensor): Input feature maps from support images [N*S, C, H, W]
            support_heatmaps (List[Tensor]): Length N list of ground truth heatmaps [S, K_i, H, W]
            optim_lr (float): Learning rate for adaptation
            optim_steps (int): Number of gradient update steps
            optim_betas (Tuple[float, float]): Adam optimizer betas
        """
        support_features_split = list(torch.tensor_split(support_features, self.batch_size, dim=0))
        for feature, heatmap, reg_head in zip(support_features_split, support_heatmaps, self.regressors):
            reg_head.adapt(
                support_features=feature,
                support_heatmaps=heatmap,
                optim_lr=optim_lr,
                optim_steps=optim_steps,
                optim_betas=optim_betas
            )

    def calculate_loss(self,
                       query_features: Tensor,
                       query_heatmaps: List[Tensor]) -> Tensor:
        """
        Calculates the total loss across all regression heads on the query set.
        Args:
            query_features (Tensor): Input feature maps from query images [N*Q, C, H, W]
            query_heatmaps (List[Tensor]): Length N list of ground truth heatmaps [Q, K_i, H, W]

        Returns:
            total_loss (Tensor): A scalar loss value.
        """
        total_loss = 0.0
        query_features_split = list(torch.tensor_split(query_features, self.batch_size, dim=0))
        for feature, heatmap, reg_head in zip(query_features_split, query_heatmaps, self.regressors):
            pred = reg_head(feature)
            loss = kl_loss(pred, heatmap)
            total_loss += loss
        return total_loss

class LitModule(L.LightningModule):
    def __init__(self, 
                 feature_model: nn.Module,
                 feature_channels: int,
                 main_lr: float,
                 main_betas: Tuple[float, float],
                 adapt_lr: float,
                 adapt_steps: int,
                 adapt_betas: Tuple[float, float],
                 accelerator: str = 'auto') -> None:
        super().__init__()

        self.feature_model = feature_model
        self.feature_channels = feature_channels

        self.main_lr = main_lr
        self.main_betas = main_betas
        self.adapt_lr = adapt_lr
        self.adapt_steps = adapt_steps
        self.adapt_betas = adapt_betas

        self.save_hyperparameters(ignore=['feature_model'])

        self.accelerator = accelerator

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.feature_model.parameters(), lr=self.main_lr, betas=self.main_betas)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """
        Args:
            batch (Any): A dictionary containing batched tensors:
                {
                    'support_images': Tensor of shape (batch_size * n_shot, 3, H, W),
                    'support_heatmaps': List of tensors of shape (n_shot, K_i, H, W),
                    'query_images': Tensor of shape (batch_size * n_query, 3, H, W),
                    'query_heatmaps': List of tensors of shape (n_query, K_i, H, W),
                    'n_keypoints': List of number of keypoints for each task,
                    'task_indices': List of task indices,
                }
            
            batch_idx (int): Current index of the batch

        Returns:
            loss (Tensor): Loss function value for the query set.
        """
        support_images = batch['support_images']
        support_heatmaps = batch['support_heatmaps']
        query_images = batch['query_images']
        query_heatmaps = batch['query_heatmaps']
        n_keypoints = batch['n_keypoints']

        support_features = self.feature_model(support_images)
        query_features = self.feature_model(query_images)

        multireg = MultiRegHead(in_channels=self.feature_channels, n_keypoints=n_keypoints)
        if (self.accelerator == 'auto' and torch.cuda.is_available()) or self.accelerator == 'gpu':
            multireg.cuda()
        multireg.adapt(
            support_features=support_features,
            support_heatmaps=support_heatmaps,
            optim_lr=self.adapt_lr,
            optim_steps=self.adapt_steps,
            optim_betas=self.adapt_betas,
        )

        loss = multireg.calculate_loss(
            query_features=query_features,
            query_heatmaps=query_heatmaps,
        )

        self.log('train/loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss
    
class PipelineTestModel(nn.Module):
    def __init__(self) -> None:
        """
        A very bad model, only for pipeline checking the integrity of the pipeline.
        Args:
            None
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return torch.relu(x)

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['train']

    dataset = FSKeypointDataset(
        path=cfg.get('dataset_dir', None),
        epoch_length=cfg.get('main_steps', 10),
        n_shot=cfg.get('n_shot', 5),
        n_query=cfg.get('n_query', 5),
        use_keypoint_subsets=cfg.get('use_keypoint_subsets', -1),
        resolution=cfg.get('resolution', (224, 224)),
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=cfg.get('num_workers', 1),
        collate_fn=Collator(
            resolution=cfg.get('resolution', (224, 224)), 
            sigma=cfg.get('heatmap_sigma', 5.0)
        ),
        drop_last=True,
    )

    print(cfg)

    feature_model = PipelineTestModel()
    feature_channels = 32

    lightning_module = LitModule(
        feature_model=feature_model,
        feature_channels=feature_channels,
        main_lr=cfg.get('main_lr', 1e-4),
        main_betas=tuple(cfg.get('main_betas', [0.9, 0.999])),
        adapt_lr=cfg.get('adapt_lr', 1e-3),
        adapt_steps=cfg.get('adapt_steps', 5),
        adapt_betas=tuple(cfg.get('adapt_betas', [0.9, 0.999]))
    )

    wandb_logger = WandbLogger(project="graduation_project", log_model="all")
    trainer = Trainer(
        max_epochs=1,
        accelerator=cfg.get('accelerator', 'auto'),
        devices=cfg.get('devices', 1),
        logger=wandb_logger,
        log_every_n_steps=1,
    )

    trainer.fit(model=lightning_module, train_dataloaders=dataloader)