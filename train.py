import os.path as osp
import warnings
from argparse import ArgumentParser

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from save_phase import SaveTopKEncoderCallback, SaveTopKModelCallback

from dataloaders import DATAMODULES
from models import MODELS


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config_name', type=str, help='The name of configuration file.')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--pseudo_weight', type=float, default=None, help='Override pseudo label weight')
    parser.add_argument('--save_dir', type=str, default=None, help='Override work_dir for saving checkpoints and logs')
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")

    args = parse_args()
    cfg_path = osp.join('configs', f'{args.config_name}.yaml')
    cfg = Config.fromfile(cfg_path)
    
    if args.pseudo_weight is not None:
        print(f"Overriding pseudo_weight in config: {cfg.loss.pseudo_weight} -> {args.pseudo_weight}")
        cfg.loss.pseudo_weight = args.pseudo_weight
    
    if args.save_dir is not None:
        print(f"Overriding work_dir: {cfg.training.work_dir} -> {args.save_dir}")
        cfg.training.work_dir = args.save_dir    

    # Set random seed
    print("Seed from config:", cfg.training.seed)
    seed_everything(cfg.training.seed)

    # Build datamodule (Phase1: only train set will be loaded)
    data_module = DATAMODULES.build({
        'type': cfg.dataset.name,
        'cfg': cfg
    })

    # Build model (make sure your model supports Phase1)
    model = MODELS.build({
        'type': cfg.model.type,
        'cfg': cfg
    })

    print(f'Training {type(model).__name__}')

    # Resume checkpoint path
    resume_path = cfg.training.get('resume_from', None)
    if resume_path:
        print(f'Resuming training from: {resume_path}')
    else:
        print('Starting training from scratch.')

    # Logging and checkpoint saving
    work_dir = osp.join(cfg.training.work_dir, args.config_name)
    
    callbacks = []
    if cfg.dataset.phase == 1:
        callbacks.append(SaveTopKEncoderCallback(
            save_dir=work_dir,
            monitor='train/loss' if cfg.dataset.phase == 1 else 'train_phase2/loss',
            mode='min',
            top_k=cfg.training.get('save_top_k', 3)
        ))
    elif cfg.dataset.phase == 2:
        callbacks.append(SaveTopKModelCallback(
            save_dir=work_dir,
            monitor='train/loss' if cfg.dataset.phase == 1 else 'train_phase2/loss',
            mode='min',
            top_k=cfg.training.get('save_top_k', 3)
        ))
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=work_dir,
            every_n_epochs=cfg.evaluation.every_n_epochs,
            monitor='val/rms', 
            mode='min',
            save_weights_only=False,
            save_top_k=cfg.training.get('save_top_k', 3)
        )
        callbacks.append(checkpoint_callback)      

    logger = TensorBoardLogger(save_dir=work_dir, name="tcdepth_logs")

    # Setup trainer
    trainer = Trainer(
        logger=logger,
        precision=cfg.training.precision,
        accelerator='gpu',
        devices=args.gpus,
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=cfg.evaluation.every_n_epochs if cfg.dataset.phase != 1 else 1,
        #callbacks=[checkpoint_callback],
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=cfg.training.find_unused_parameters, static_graph=False),
        sync_batchnorm=(args.gpus > 1),
        num_nodes=1,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )

    # Start training
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)
    print(f"[Checkpoint path] {work_dir}")
    print(f"[Monitoring metric]")
    print('Training completed.')


if __name__ == '__main__':
    main()

