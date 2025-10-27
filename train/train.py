from train.model import PipelineTestModel, LitModule
from train.dataset import FSKeypointDataset, Collator
from typing import Any, Tuple, List
import yaml
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
import torch
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)['train']

    dataset = FSKeypointDataset(
        path=cfg.get('dataset_dir_train', None),
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

    val_dataset = FSKeypointDataset(
        path=cfg.get('dataset_dir_train', None),
        epoch_length=cfg.get('val_steps', 1),
        n_shot=cfg.get('n_shot', 5),
        n_query=cfg.get('n_query', 5),
        use_keypoint_subsets=cfg.get('use_keypoint_subsets', -1),
        resolution=cfg.get('resolution', (224, 224)),
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=1,
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
        precision=cfg.get('precision', 'bf16-mixed'),
        check_val_every_n_epoch=None,
        val_check_interval=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=lightning_module, 
                train_dataloaders=dataloader, 
                val_dataloaders=[val_dataloader])