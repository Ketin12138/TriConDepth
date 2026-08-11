import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pytorch_lightning import LightningModule
from networks.YHDepth import YHDepth
from networks.DCDepth import DCDepth
from networks.projectionhead import ProjectionHead1
from .registry import MODELS
from torch.optim import AdamW

class IHPC_Loss(nn.Module):
    def __init__(self, beta=0.1, tau=0.5):
        super().__init__()
        self.beta = beta
        self.tau = tau
        
    def forward(self, anchor, positive, negatives):
        
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=1)
        
        # Calculate anchor-positive sample similarity
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        weights = torch.exp(-self.beta * pos_sim)  
        
        # Calculate the positive sample
        pos_term = torch.exp(pos_sim / self.tau) * weights
        
        # Calculate the negative sample
        neg_term = torch.sum(torch.exp(F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1) / self.tau), dim=1)
        
        # Calculate Contrastive Loss
        loss = -torch.log(pos_term / (pos_term + neg_term))
        return loss.mean()
    
class MoCoMemoryBank:
    def __init__(self, size=4096, feat_dim=256, device='cuda'):
        self.size = size
        self.device = torch.device(device)
        self.feats = torch.randn(size, feat_dim, device=self.device)
        self.ptr = 0

    def update(self, new_feats):
        new_feats = new_feats.detach()
        n = new_feats.size(0)
        for i in range(n):
            self.feats[self.ptr] = new_feats[i]
            self.ptr = (self.ptr + 1) % self.size 
           
    def sample_negatives(self, num_samples, batch_feats=None, recent_window=512, skip_last_n=64, avoid_oldest=512):
        max_index = min(self.ptr, self.size)
        if max_index <= num_samples + skip_last_n + avoid_oldest:
            if batch_feats is not None:
                batch_size = batch_feats.size(0)
                indices = torch.randint(0, batch_size, (num_samples,))
                return batch_feats[indices]
            else:
                return torch.randn(num_samples, self.feats.size(1), device=self.device)

        # Define the area
        start = max(avoid_oldest, max_index - recent_window - skip_last_n)
        end = max(start + 1, max_index - skip_last_n)  # Ensure at least one valid sampling interval

        indices = torch.randint(start, end, (num_samples,))
        return self.feats[indices]
        
       
@MODELS.register_module()
class Stage1Model(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg  
        self.warmup_steps = cfg.model.get('warmup_steps', 2000)

        self.model = YHDepth(
            backbone_name=self.cfg.model.encoder,
            img_size=(self.cfg.dataset.input_height, self.cfg.dataset.input_width),
            drop_path_rate=self.cfg.model.drop_path_rate,
            drop_path_rate_crf=self.cfg.model.drop_path_rate_crf,
            seq_dropout_rate=self.cfg.model.seq_dropout_rate
        )
        
        
        self.encoder = self.model.backbone
        self.head = ProjectionHead1(num_class=256)       
        self.memory = MoCoMemoryBank(size=cfg.model.mem_size, feat_dim=cfg.model.feat_dim, device=self.device)
        self.criterion = IHPC_Loss(beta=cfg.model.beta, tau=cfg.model.tau)
        self.lr = cfg.optimization.lr

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
        ])

    def _augment(self, img_batch):
        views1, views2 = [], []
        for img in img_batch:
            if img.shape[0] == 4:
                img = img[:3]
            img = TF.to_pil_image(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            views1.append(TF.to_tensor(img1).to(self.device))
            views2.append(TF.to_tensor(img2).to(self.device))
        return torch.stack(views1), torch.stack(views2)
    
    def forward(self, x):
        feats = self.encoder(x)
        return feats[-1]
    

    def training_step(self, batch, batch_idx):
        img = batch['image']
        feats = self.forward(img)
        orig_proj = self.head(feats)

        if self.global_step < self.warmup_steps:
            self.memory.update(orig_proj.detach())
            dummy_loss = 0.0 * feats[0].sum()
            return dummy_loss

        view1, view2 = self._augment(img)
        feat1 = self.forward(view1)
        feat2 = self.forward(view2)
        proj1 = self.head(feat1)
        proj2 = self.head(feat2)

        neg_feats = self.memory.sample_negatives(64, batch_feats=orig_proj,recent_window=512,skip_last_n=100,
                                                 avoid_oldest=512).to(self.device)
        loss = self.criterion(proj1, proj2, neg_feats)

        self.memory.update(orig_proj.detach())
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        self.total_steps = self.trainer.estimated_stepping_batches  

        optimizer = torch.optim.AdamW(
            [
                {
                    'params': self.encoder.parameters(),  
                    'lr': self.lr,
                    'weight_decay': 0.01  
                },
                {
                    'params': self.head.parameters(),  
                    'lr': self.lr * 5,  
                    'weight_decay': 0.0
                }
            ]
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            [group['lr'] for group in optimizer.param_groups],
            total_steps=self.total_steps,
            div_factor=10.0,
            final_div_factor=100.0,
            pct_start=0.3,
            anneal_strategy='cos' 
        )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]