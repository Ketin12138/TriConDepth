import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.YHDepth import YHDepth
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from networks.projectionhead import ProjectionHead2
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

from utils import post_process_depth, flip_lr, compute_errors_pth, colormap, inv_normalize, colormap_magma
from .registry import MODELS
from .utils import SmoothRegularity


class SILogLossInstance(nn.Module):
    def __init__(self, variance_focus: float, patch_size: int = 8, min_valid_pixels: int = 4, square_root: bool = True):
        super().__init__()

        assert 0 <= variance_focus <= 1.
        self.variance_focus = variance_focus

        self.patch_size = patch_size
        self.min_valid_pixels = min_valid_pixels
        self.square_root = square_root
        self.register_buffer(
            '_weight',
            torch.ones(1, 1, self.patch_size, self.patch_size, dtype=torch.float32)
        )

    def forward(self, depth_log: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, gt_log_space: bool = False, **kwargs):
        """
        Compute the silog loss
        :param depth_log: depth prediction in log space
        :param depth_gt: depth ground truth in metric
        :param mask: valid mask, binary
        :return:
        """
        mask = mask.float()
        assert depth_log.shape == mask.shape

        # filter mask
        if self.min_valid_pixels > 0:
            patch_mask = F.conv2d(mask, self._weight, stride=self.patch_size)
            patch_mask = (patch_mask >= self.min_valid_pixels).float()
            patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=-1).repeat_interleave(self.patch_size, dim=-2)
            mask = mask * patch_mask

        B, _, H, W = depth_log.shape
        # convert gt to log space
        
        if not gt_log_space:
            depth_gt = torch.log(depth_gt.clamp_min(1.0e-3))
        # flatten
        depth_log = depth_log.flatten(1)
        depth_gt = depth_gt.flatten(1)
        mask = mask.flatten(1)
        # compute difference
        diff = (depth_log - depth_gt) * mask
        # compute silog loss for each sample
        num = torch.clamp(mask.sum(1), min=1.0)
        loss = diff.square().sum(1) / num - self.variance_focus * (diff.sum(1) / num).square()  # (B,)
        if self.square_root:
            loss = loss.sqrt()
        loss = 10. * loss
        # compute weight
        loss = loss.mean()

        return loss

class FeatureSwap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape
        mask = torch.zeros((B, 1, H, W), device=feat_a.device)
        mask[:, :, :H // 2, :] = 1  
        mixed_a = mask * feat_a + (1 - mask) * feat_b
        mixed_b = mask * feat_b + (1 - mask) * feat_a
        return mixed_a, mixed_b
    
@MODELS.register_module()
class Stage4Model(LightningModule):
    """
    Bisection depth model
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.patch_size = 8
        self.max_depth = self.cfg.dataset.max_depth
        self.min_depth = self.cfg.dataset.min_depth

        # output space, (metric or log)
        self.output_space = self.cfg.model.output_space
        assert self.output_space in ['metric', 'log']

        # model
        self.model = YHDepth(
            backbone_name= self.cfg.model.encoder,
            #self.cfg.model.pretrain,
            scale=(math.log(self.max_depth) if self.output_space == 'log' else self.max_depth),
            img_size=(self.cfg.dataset.input_height, self.cfg.dataset.input_width),
            #ape=self.cfg.model.ape,
            drop_path_rate=self.cfg.model.drop_path_rate,
            drop_path_rate_crf=self.cfg.model.drop_path_rate_crf,
            seq_dropout_rate=self.cfg.model.seq_dropout_rate
        )
        
        is_training = cfg.get("mode", "train") == "train"
        pretrained_path = cfg.model.get('pretrained_model_path', None)
        
        if is_training and pretrained_path is not None:
            print(f"[Load] Loading model from: {pretrained_path}")
            state = torch.load(pretrained_path, map_location='cpu')
            
            if isinstance(state, dict) and 'state_dict' in state:
                print("[Load] Detected Lightning .ckpt format.")
                model_state_dict = {
                    k.replace('model.', ''): v
                    for k, v in state['state_dict'].items()
                    if k.startswith('model.')
                }
   
            elif isinstance(state, dict):
                print("[Load] Detected raw .pth state_dict format.")
                model_state_dict = {
                    k.replace('model.', ''): v
                    for k, v in state.items()
                    if k.startswith('model.')
                }

            else:
                raise ValueError(f"Unsupported pretrained format: {type(state)}")

            # 加载
            missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
            loaded_keys = set(model_state_dict.keys()) - set(unexpected_keys)
            print(f"[Load] ✅ Loaded {len(loaded_keys)} keys.")

        # loss
        self.si_log = SILogLossInstance(self.cfg.loss.variance_focus, self.patch_size,
                                        self.cfg.loss.min_valid_pixels, self.cfg.loss.square_root)

        self.smooth_regularity = SmoothRegularity()
        self.feature_swap = FeatureSwap()

        self.beta = self.cfg.loss.beta
        if self.beta is not None:
            assert 0.5 <= self.beta <= 1.5

        self.total_steps = None
        self.register_buffer("global_step_buffer", torch.tensor(0, dtype=torch.long))

        print(f'Output Space={self.output_space}.')
        

    def output2metric(self, out: torch.Tensor):
        """
        Convert output in metric or log space to metric depth
        :param out:
        :return:
        """
        if self.output_space == 'log':
            return out.exp()
        elif self.output_space == 'metric':
            return out
        else:
            raise NotImplementedError

    def output2log(self, out: torch.Tensor):
        """
        Convert output in metric or log space to log space
        :param out:
        :return:
        """
        if self.output_space == 'log':
            return out
        elif self.output_space == 'metric':
            return out.clamp_min(1.0e-4).log()
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        self.total_steps = self.trainer.estimated_stepping_batches

        optimizer = AdamW(
            [
                {
                    'params': self.model.parameters_5x(),
                    'lr': self.cfg.optimization.max_lr,
                    'weight_decay': 0.
                },
                {
                    'params': self.model.parameters_1x(),
                    'lr': self.cfg.optimization.max_lr / self.cfg.optimization.lr_ratio,
                    'weight_decay': self.cfg.optimization.weight_decay
                },
            ]
        )
        # scheduler
        lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lrs, self.total_steps, div_factor=self.cfg.optimization.div_factor,
            final_div_factor=self.cfg.optimization.final_div_factor, pct_start=self.cfg.optimization.pct_start,
            anneal_strategy=self.cfg.optimization.anneal_strategy
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @torch.no_grad()
    def log_images(self, image, image_aug, depth_preds, depth_gt):
        writer = self.logger.experiment

        depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e-3, depth_gt)
        global_step = self.global_step

        # visualize rgb
        writer.add_image(f'train/image', inv_normalize(image[0, :, :, :]), global_step)
        writer.add_image(f'train/image_aug', inv_normalize(image_aug[0, :, :, :]), global_step)

        # visualize depth
        n_pred = len(depth_preds)
        if self.cfg.dataset.name in ['nyu', 'tofdc']:
            writer.add_image(f'train/depth_gt', colormap(depth_gt[0, :, :, :]), global_step)

            for idx in range(n_pred):
                depth_pred = self.output2metric(depth_preds[idx].detach()[0])
                writer.add_image(f'train/depth_pred_{idx}', colormap(depth_pred), global_step)

        else:
            writer.add_image(f'train/depth_gt', colormap_magma(torch.log10(depth_gt[0, :, :, :])), global_step)

            for idx in range(n_pred):
                depth_pred = self.output2metric(depth_preds[idx].detach()[0])
                writer.add_image(f'train/depth_pred_{idx}', colormap_magma(torch.log10(depth_pred)), global_step)

    def get_exponential_weights(self, n: int):
        xs = np.arange(n)
        ys = self.beta ** xs
        ys = ys / ys.sum()
        weights = ys.tolist()
        return list(reversed(weights))

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()
        
    #def augment_image(self, images):
        #images_aug = []
        #for img in images:
            #img_pil = T.ToPILImage()(img.cpu())

            #if random.random() < 0.5:
                #brightness_factor = random.uniform(0.98, 1.02)  
                #img_pil = TF.adjust_brightness(img_pil, brightness_factor)

            #if random.random() < 0.5:
                #contrast_factor = random.uniform(0.98, 1.02)  
                #img_pil = TF.adjust_contrast(img_pil, contrast_factor)

            #if random.random() < 0.5:
                #saturation_factor = random.uniform(0.98, 1.02)  
                #img_pil = TF.adjust_saturation(img_pil, saturation_factor)

            #if random.random() < 0.3:
                #hue_factor = random.uniform(-0.02, 0.02)  
                #img_pil = TF.adjust_hue(img_pil, hue_factor)

            #if random.random() < 0.1:
                #blur = T.GaussianBlur(kernel_size=3, sigma=(0.05, 0.1))  
                #img_pil = blur(img_pil)

            #img_tensor = T.ToTensor()(img_pil).to(img.device)
            #images_aug.append(img_tensor)

        #return torch.stack(images_aug) 
    
    def augment_image(self, image, strong=False):
        transform_info = {}

        if strong:
            image = self.horizontal_flip(image)
            transform_info['flip'] = True
        else:
            transform_info['flip'] = False  

        return image, transform_info
    
    def horizontal_flip(self, image):
        return torch.flip(image, dims=[3])  
    
    def training_step(self, batch, batch_idx):
        self.global_step_buffer += 1
        ramp_weight = min(1.0, self.global_step_buffer.item() / self.cfg.loss.pseudo_ramp_up_iters)
        labeled_sample, unlabeled_sample = batch
        image_l = labeled_sample['image']
        depth_gt = labeled_sample['depth']
        image_u = unlabeled_sample['image']
        
        loss_total = 0.
        log_dict = {}
        
        if depth_gt is not None:
            depths, freq_regs = self.model(image_l, mode='train_stage3')
            mask = depth_gt >= self.min_depth
            weight_func = self.get_exponential_weights
            for idx, (depth_log, freq_reg, weight) in enumerate(zip(depths, freq_regs, weight_func(len(depths)))):
                depth_log = self.output2log(depth_log)

                si_log = self.si_log(depth_log, depth_gt, mask)
                if idx > 3:
                    smooth_reg = self.smooth_regularity(depth_log, image_l)
                else:
                    smooth_reg = torch.zeros(1, dtype=si_log.dtype, device=si_log.device)

                log_dict[f'loss/si_log_{idx}'] = si_log.item()
                log_dict[f'loss/freq_reg_{idx}'] = freq_reg.item()
                log_dict[f'loss/smooth_reg_{idx}'] = smooth_reg.item()

                loss_total += weight * (si_log + self.cfg.loss.freq_reg_weight * freq_reg +
                                    self.cfg.loss.smooth_reg_weight * smooth_reg)  
                      
            #image1 = self.augment_image(image_u)
            #image2 = self.augment_image(image_u)
            
            weak_aug, weak_info = self.augment_image(image_u, strong=False)
            strong_aug, strong_info = self.augment_image(image_u, strong=True)
        
            feats1 = self.model.backbone(weak_aug)
            feats2 = self.model.backbone(strong_aug)

            encoder_consistency = torch.tensor(0.0, device=feats1[0].device)
            for f1, f2 in zip(feats1, feats2):
                if strong_info['flip']:
                    f2 = torch.flip(f2, dims=[3])
                diff = (f1 - f2).abs()
                encoder_consistency += diff.mean()
        
            feat1_layers, context1, feats0_1 = self.model.extract_feature(weak_aug)
            feat2_layers, context2, feats0_2 = self.model.extract_feature(strong_aug)
        
            feat1_layers_swapped = []
            feat2_layers_swapped = []
            for f1, f2 in zip(feat1_layers, feat2_layers):
                f1s, f2s = self.feature_swap(f1, f2)
                feat1_layers_swapped.append(f1s)
                feat2_layers_swapped.append(f2s)
            
            embedding1_swapped = self.model.fuse_features(feat1_layers_swapped, context1, feats0_1)
            embedding2_swapped = self.model.fuse_features(feat2_layers_swapped, context2, feats0_2)  
                  
            depth_swapped1 = self.model.decode_from_feature(embedding1_swapped)
            depth_swapped2 = self.model.decode_from_feature(embedding2_swapped)
            
            if strong_info['flip']:
                depth_swapped2[0][-1] = torch.flip(depth_swapped2[0][-1], dims=[3])

            log_dict['loss/encoder_consistency'] = encoder_consistency.item()
            loss_total += self.cfg.loss.encoder_consistency_weight * encoder_consistency
        
            valid_mask = (depth_swapped1[0][-1] > self.min_depth) & \
                         (depth_swapped1[0][-1] < self.max_depth) & \
                         (depth_swapped2[0][-1] > self.min_depth) & \
                         (depth_swapped2[0][-1] < self.max_depth)
            if valid_mask.sum() == 0:
                out_consistency = torch.tensor(0., device=depth_swapped1[0][-1].device)
            else:
                out_consistency = F.l1_loss(depth_swapped1[0][-1][valid_mask],depth_swapped2[0][-1][valid_mask])

            log_dict['loss/output_consistency'] = out_consistency.item()
            loss_total += self.cfg.loss.consistency_l1_weight * out_consistency

            for k, v in log_dict.items():
                if k in ['loss/pseudo1','loss/pseudo2','loss/output_consistency', 'loss/encoder_consistency']:
                    self.log(k, v, on_step=True, prog_bar=True)
                else:
                    self.log(k, v, on_step=True, prog_bar=False)

        return loss_total

    def evaluate_depth(self, batch, batch_idx):
        post_process = True

        # fetch data
        image = batch['image']
        gt_depth = batch['depth']
        has_valid_depth = batch['has_valid_depth']

        if not has_valid_depth:
            # print('Has no valid depth.')
            return

        depths = self.model(image)
        depth = self.output2metric(depths[-1])
        if post_process:
            image_flipped = flip_lr(image)
            depth_flipped = self.output2metric(self.model(image_flipped)[-1])
            pred_depth = post_process_depth(depth, depth_flipped)
        else:
            pred_depth = depth

        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()

        if self.cfg.evaluation.do_kb_crop:
            assert self.cfg.dataset.name in ['kitti_eigen', 'kitti_official']
            height, width = pred_depth.shape
            top_margin = 352 - height
            left_margin = (1216 - width) // 2
            pred_depth_uncropped = torch.zeros(352, 1216).type_as(pred_depth)
            pred_depth_uncropped[top_margin:, left_margin: left_margin + width] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < self.min_depth] = self.min_depth
        pred_depth[pred_depth > self.max_depth] = self.max_depth
        pred_depth[torch.isinf(pred_depth)] = self.max_depth
        pred_depth[torch.isnan(pred_depth)] = self.min_depth

        valid_mask = torch.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)

        if self.cfg.evaluation.garg_crop or self.cfg.evaluation.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros_like(valid_mask)

            if self.cfg.evaluation.garg_crop:
                eval_mask[int(0.40810811 * gt_height): int(0.99189189 * gt_height),
                int(0.03594771 * gt_width): int(0.96405229 * gt_width)] = 1

            elif self.cfg.evaluation.eigen_crop:
                if self.cfg.dataset.name == 'kitti':
                    eval_mask[int(0.3324324 * gt_height): int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width): int(0.96405229 * gt_width)] = 1
                elif self.cfg.dataset.name == 'nyu':
                    eval_mask[45: 471, 41: 601] = 1
                else:
                    raise NotImplementedError

            valid_mask = torch.logical_and(valid_mask, eval_mask)

        # compute metrics
        measures = compute_errors_pth(gt_depth[valid_mask], pred_depth[valid_mask])

        # log
        for metric_name, metric in measures.items():
            self.log(f'val/{metric_name}', metric, sync_dist=True)

        # log image
        if batch_idx < 8:
            writer = self.logger.experiment

            writer.add_image(f'val/image_{batch_idx}', inv_normalize(image[0]), global_step=self.global_step)

            # plot error map
            error_map = ((pred_depth - gt_depth).abs() * valid_mask).unsqueeze(0)
            writer.add_image(f'val/error_map_{batch_idx}', colormap(error_map), global_step=self.global_step)

            # pred_depth = remove_border(pred_depth, 4)
            if self.cfg.dataset.name in ['nyu', 'tofdc']:
                writer.add_image(f'val/pred_{batch_idx}', colormap(pred_depth.unsqueeze(0)),
                                 global_step=self.global_step)
                writer.add_image(f'val/gt_{batch_idx}', colormap(gt_depth.unsqueeze(0)),
                                 global_step=self.global_step)
            else:
                writer.add_image(f'val/pred_{batch_idx}', colormap_magma(pred_depth.unsqueeze(0).log10()),
                                 global_step=self.global_step)
                writer.add_image(f'val/gt_{batch_idx}', colormap_magma(gt_depth.clamp_min(1.0e-3).unsqueeze(0).log10()),
                                 global_step=self.global_step)

    def validation_step(self, batch, batch_idx):
        self.evaluate_depth(batch, batch_idx)
