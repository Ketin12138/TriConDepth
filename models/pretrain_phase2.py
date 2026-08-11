import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.YHDepth import YHDepth
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from networks.projectionhead import ProjectionHead2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from networks.encoder import get_backbone

from .registry import MODELS

class DepthSupervisedWindowContrastive(nn.Module):
    def __init__(self, window_size=5, feature_dim=256, depth_thresh=None, self_sup_threshold=0.3,
                 self_sup_ratio=0.2, self_sup_weight=1.0,sigma=0.05, device='cuda'):
        super().__init__()
        self.window_size = window_size
        self.depth_thresh = depth_thresh
        self.sigma = sigma
        self.device = device
        self.self_sup_threshold = self_sup_threshold  
        self.self_sup_ratio = self_sup_ratio          
        self.self_sup_weight = self_sup_weight        

        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(feature_dim, feature_dim)

    def forward(self, features, depth_labels):
        
        B, C, H, W = features.shape

        # Padding makes H and W evenly divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        features = F.pad(features, (0, pad_w, 0, pad_h))  # (B, C, Hp, Wp)
        Hp, Wp = features.shape[2], features.shape[3]
        if depth_labels.ndim == 3:
            depth_labels = depth_labels.unsqueeze(1)
        depth_labels = depth_labels.float()
        depth_labels = F.interpolate(depth_labels, size=(Hp, Wp), mode='nearest')

        win_area = self.window_size ** 2

        # 将 (B, C, Hp, Wp) -> (B, num_windows, win_area, C)
        features_windows = features.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)

        # features_windows shape: (B, C, num_win_h, num_win_w, wh, ww)
        B, C, num_win_h, num_win_w, wh, ww = features_windows.shape
        features_windows = features_windows.contiguous().view(B, C, num_win_h * num_win_w, win_area).permute(0, 2, 3, 1)

        # Perform the same window expansion on depth_labels
        depth_windows = depth_labels.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
        B, C, num_win_h, num_win_w, wh, ww = depth_windows.shape
        depth_windows = depth_windows.contiguous().view(B, C, num_win_h * num_win_w, wh * ww)
        depth_windows = depth_windows.squeeze(1)  # (B, num_windows, win_area)

        # Q, K projection
        Q = self.to_q(features_windows)  # (B, num_windows, win_area, C)
        K = self.to_k(features_windows)  # (B, num_windows, win_area, C)

        # Feature normalization
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # Calculate cosine similarity
        sim_matrix = torch.matmul(Q, K.transpose(-2, -1))  # (B, num_windows, win_area, win_area)
        sim_matrix = torch.exp(sim_matrix / 0.1) 

        # Depth difference matrix
        depth_diff = torch.abs(depth_windows.unsqueeze(-1) - depth_windows.unsqueeze(-2))  # (B, num_windows, win_area, win_area)

        # Constructing positive sample weights
        if self.depth_thresh is not None:
            depth_weight = (depth_diff < self.depth_thresh).float()
        else:
            depth_weight = torch.exp(-(depth_diff ** 2) / (2 * self.sigma ** 2))
            
        valid_depth_mask = (depth_windows > 0).float()
        valid_pair_mask = valid_depth_mask.unsqueeze(-1) * valid_depth_mask.unsqueeze(-2)
        
        eye_mask = torch.eye(win_area, device=features.device).unsqueeze(0).unsqueeze(0)  # (1, 1, win_area, win_area)
        valid_mask = (1.0 - eye_mask) * valid_pair_mask  

        pos_weight = depth_weight * valid_mask
        neg_weight = (1.0 - depth_weight) * valid_mask
        
        pos_sum = pos_weight.sum(dim=-1)  # (B, num_windows, win_area)
        neg_sum = neg_weight.sum(dim=-1)  # (B, num_windows, win_area)

        valid_pos = (pos_sum > 0)
        valid_neg = (neg_sum > 0)
        valid_samples_mask = valid_pos & valid_neg  

        valid_depth_sum = valid_depth_mask.sum(dim=-1)  # (B, num_windows)
        valid_window_mask = valid_depth_sum > 0  

        valid_pixel_mask = valid_samples_mask.any(dim=-1)  # (B, num_windows)

        final_valid_window_mask = valid_window_mask & valid_pixel_mask  # (B, num_windows)
        
        pos_sim = (sim_matrix * pos_weight).sum(dim=-1)  # (B, num_windows, win_area)
        neg_sim = (sim_matrix * neg_weight).sum(dim=-1)  # (B, num_windows, win_area)
        
        loss = torch.zeros_like(pos_sim)
        
        valid_loss_idx = (final_valid_window_mask.unsqueeze(-1) & valid_samples_mask).nonzero(as_tuple=True)

        if valid_loss_idx[0].numel() > 0:
            valid_pos_sim = pos_sim[valid_loss_idx]
            valid_neg_sim = neg_sim[valid_loss_idx]
            loss_vals = -torch.log(valid_pos_sim / (valid_pos_sim + valid_neg_sim + 1e-6) + 1e-6)
            loss[valid_loss_idx] = loss_vals

        label_loss = loss.mean()
    
        invalid_ratio = 1.0 - (valid_depth_mask.sum(dim=-1) / win_area)
        
        self_sup_mask = (
            (~final_valid_window_mask) |  
            (invalid_ratio >= self.self_sup_threshold)  
        )
        self_sup_mask = self_sup_mask.unsqueeze(-1).expand(-1, -1, win_area)
        
        sim_matrix_adjusted = sim_matrix * (1 - eye_mask)
        
        pos_k = max(1, int(self.self_sup_ratio * win_area))
        neg_k = max(1, int(self.self_sup_ratio * win_area))
        
        topk_pos, _ = torch.topk(sim_matrix_adjusted, k=pos_k, dim=-1)
        pos_sum = topk_pos.sum(dim=-1)
        
        bottomk_neg, _ = torch.topk(sim_matrix_adjusted * -1, k=neg_k, dim=-1)
        neg_sum = bottomk_neg.sum(dim=-1) * -1
        
        self_loss_per_anchor = -torch.log(
            pos_sum / (pos_sum + neg_sum + 1e-6) + 1e-6
        )
        
        if self_sup_mask.sum() > 0:
            self_loss = (self_loss_per_anchor * self_sup_mask).sum() / self_sup_mask.sum()
        else:
            self_loss = torch.tensor(0.0).to(features.device)
        
        total_loss = label_loss + self.self_sup_weight * self_loss
        
        return total_loss

class CrossViewDepthSupervisedWindowContrastive(nn.Module):
    def __init__(self, window_size=5, feature_dim=256, depth_thresh=None, sigma=0.05, device='cuda'):
        super().__init__()
        self.window_size = window_size
        self.depth_thresh = depth_thresh
        self.sigma = sigma
        self.device = device

        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(feature_dim, feature_dim)

    def forward(self, feat1, feat2, depth1, depth2):
        B, C, H, W = feat1.shape

        # padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        feat1 = F.pad(feat1, (0, pad_w, 0, pad_h))
        feat2 = F.pad(feat2, (0, pad_w, 0, pad_h))
        depth1 = F.interpolate(depth1.float(), size=feat1.shape[2:], mode='nearest')
        depth2 = F.interpolate(depth2.float(), size=feat2.shape[2:], mode='nearest')

        Hp, Wp = feat1.shape[2:]

        def unfold(x):
            x = x.unfold(2, self.window_size, self.window_size).unfold(3, self.window_size, self.window_size)
            B, C, nH, nW, wh, ww = x.shape
            return x.contiguous().view(B, C, nH * nW, self.window_size**2).permute(0, 2, 3, 1)  # (B, num_win, win_area, C)

        q_feat = F.normalize(self.to_q(unfold(feat1)), dim=-1)  # view1: anchor
        k_feat = F.normalize(self.to_k(unfold(feat2)), dim=-1)  # view2: sample
        
        depth1_win = unfold(depth1).squeeze(1).squeeze(-1)  # (B, num_win, win_area)
        depth2_win = unfold(depth2).squeeze(1).squeeze(-1)  # (B, num_win, win_area)

        sim = torch.matmul(q_feat, k_feat.transpose(-2, -1))  # (B, num_win, win_area, win_area)
        sim = torch.exp(sim / 0.1)  # optional temperature

        # cross-view depth difference: anchor is depth1[i], sample is depth2[j]
        depth_diff = torch.abs(depth1_win.unsqueeze(-1) - depth2_win.unsqueeze(-2))

        if self.depth_thresh is not None:
            depth_weight = (depth_diff < self.depth_thresh).float()
        else:
            depth_weight = torch.exp(-(depth_diff ** 2) / (2 * self.sigma ** 2))

        valid1 = (depth1_win > 0).float()
        valid2 = (depth2_win > 0).float()
        valid_mask = valid1.unsqueeze(-1) * valid2.unsqueeze(-2)  # (B, num_win, win_area, win_area)

        eye_mask = torch.eye(self.window_size**2, device=feat1.device).unsqueeze(0).unsqueeze(0)
        valid_mask = valid_mask * (1 - eye_mask)  # ignore (i, i) pairs

        pos_weight = depth_weight * valid_mask
        neg_weight = (1 - depth_weight) * valid_mask

        pos_sim = (sim * pos_weight).sum(-1)
        neg_sim = (sim * neg_weight).sum(-1)

        pos_sum = pos_weight.sum(-1)
        neg_sum = neg_weight.sum(-1)

        valid_pos = pos_sum > 0
        valid_neg = neg_sum > 0
        valid_pixel = valid_pos & valid_neg
        valid_window = valid_pixel.any(-1)

        final_mask = valid_window.unsqueeze(-1) & valid_pixel

        loss = torch.zeros_like(pos_sim)
        idx = final_mask.nonzero(as_tuple=True)

        if idx[0].numel() > 0:
            pos_val = pos_sim[idx]
            neg_val = neg_sim[idx]
            loss_val = -torch.log(pos_val / (pos_val + neg_val + 1e-6) + 1e-6)
            loss[idx] = loss_val

        return loss.mean()


@MODELS.register_module()
class Stage2Model(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.encoder = get_backbone(cfg.model.encoder, pretrained=False)
        
        pretrained_path = cfg.model.get('pretrained_encoder_path', None)      
        if pretrained_path is not None:
            self._load_encoder_weights(pretrained_path)
                   
        self.head = ProjectionHead2(num_class=256)

        self.model = YHDepth(
            backbone_name=cfg.model.encoder,
            img_size=(cfg.dataset.input_height, cfg.dataset.input_width),
            drop_path_rate=cfg.model.drop_path_rate,
            drop_path_rate_crf=cfg.model.drop_path_rate_crf,
            seq_dropout_rate=cfg.model.seq_dropout_rate
        )
        
        self.model.backbone = self.encoder
              
        self.window_contrastive = DepthSupervisedWindowContrastive(
            window_size=cfg.model.window_size,
            feature_dim=cfg.model.feat_dim,  
            self_sup_threshold=cfg.model.self_sup_threshold,
            self_sup_ratio=cfg.model.self_sup_ratio, 
            self_sup_weight=cfg.model.self_sup_weight,
            sigma=cfg.model.sigma,
            device=self.device,
        )
        
        self.cross_contrastive = CrossViewDepthSupervisedWindowContrastive(
            window_size=cfg.model.window_size,
            feature_dim=cfg.model.feat_dim,  
            #depth_thresh=cfg.model.depth_thresh,
            sigma=cfg.model.sigma,
            device=self.device
        )
    
    def _load_encoder_weights(self, path):
        print(f"[Load] Loading encoder from: {path}")
        state = torch.load(path, map_location='cpu')

        if isinstance(state, dict) and 'state_dict' in state:
            print("[Load] Detected Lightning .ckpt format.")
            state_dict = {
                k.replace('encoder.', ''): v
                for k, v in state['state_dict'].items()
                if k.startswith('encoder.')
            }
        elif isinstance(state, dict):
            print("[Load] Detected raw .pth state_dict format.")
            state_dict = state
        else:
            raise ValueError(f"Unsupported format: {type(state)}")

        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
        loaded_keys = set(state_dict.keys()) - set(unexpected_keys)
        print(f"[Load] Successfully loaded {len(loaded_keys)} encoder parameters.")
        print(f"  ➤ Missing keys: {missing_keys}")
        print(f"  ➤ Unexpected keys: {unexpected_keys}")    

        

    def forward(self, x):
        return self.model(x)

    def _augment(self, img_batch, depth_batch):
        view1_img, view2_img = [], []
        view1_depth, view2_depth = [], []

        for img, depth in zip(img_batch, depth_batch):
            if img.shape[0] == 4:
                img = img[:3]

            img = TF.to_pil_image(img)
            depth = TF.to_pil_image(depth.squeeze(0))  # shape: (H, W)

            def apply_photometric(x):
                if torch.rand(1) < 0.3:
                    x = TF.adjust_brightness(x, 1 + float(torch.empty(1).uniform_(-0.1, 0.1)))
                    
                if torch.rand(1) < 0.3:
                    x = TF.adjust_contrast(x, 1 + float(torch.empty(1).uniform_(-0.1, 0.1)))
            
                if torch.rand(1) < 0.3:
                    x = TF.adjust_saturation(x, 1 + float(torch.empty(1).uniform_(-0.1, 0.1)))
                return x

            def apply_noise(tensor_img):
            # PIL to Tensor is done outside
                if torch.rand(1) < 0.3:
                    noise = torch.randn_like(tensor_img) * 0.01  
                    tensor_img = torch.clamp(tensor_img + noise, 0.0, 1.0)
                return tensor_img

            img1 = apply_photometric(img)
            img2 = apply_photometric(img)

            tensor_img1 = TF.to_tensor(img1)
            tensor_img2 = TF.to_tensor(img2)

            tensor_img1 = apply_noise(tensor_img1).to(self.device)
            tensor_img2 = apply_noise(tensor_img2).to(self.device)

            tensor_depth = TF.to_tensor(depth).to(self.device)  

            view1_img.append(tensor_img1)
            view2_img.append(tensor_img2)
            view1_depth.append(tensor_depth)
            view2_depth.append(tensor_depth.clone())

        return (
            torch.stack(view1_img), torch.stack(view2_img),
            torch.stack(view1_depth), torch.stack(view2_depth)
        )


    def training_step(self, batch, batch_idx):
        img = batch['image']      # (B, 3, H, W)
        depth = batch['depth']    # (B, 1, H, W)

        view1, view2, depth1, depth2 = self._augment(img, depth)

        _, _, e1_view1 = self.model(view1, mode='train_stage2')
        _, _, e1_view2 = self.model(view2, mode='train_stage2')
        
        proj1 = self.head(e1_view1)
        proj2 = self.head(e1_view2)

        loss1 = self.window_contrastive(proj1, depth1)
        loss2 = self.window_contrastive(proj2, depth2)

        loss_cross1 = self.cross_contrastive(proj1, proj2, depth1, depth2)
        loss_cross2 = self.cross_contrastive(proj2, proj1, depth2, depth1)


        loss = loss1 + loss2 + loss_cross1 + loss_cross2

        self.log("train_phase2/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        self.total_steps = self.trainer.estimated_stepping_batches  

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.model.parameters(),  
                    'lr': self.cfg.optimization.max_lr / self.cfg.optimization.lr_ratio,
                    'weight_decay': self.cfg.optimization.weight_decay  
                },
                {
                    'params': self.head.parameters(),  
                    'lr': self.cfg.optimization.max_lr,  
                    'weight_decay': 0.0
                }
            ]
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [group['lr'] for group in optimizer.param_groups],
            total_steps=self.total_steps,
            div_factor=self.cfg.optimization.div_factor,
            final_div_factor=self.cfg.optimization.div_factor,
            pct_start=self.cfg.optimization.pct_start,
            anneal_strategy=self.cfg.optimization.anneal_strategy  
        )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

