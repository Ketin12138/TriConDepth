import os
import os.path as osp
import torch
from pytorch_lightning.callbacks import Callback
import torch.distributed as dist

def is_global_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class SaveTopKEncoderCallback(Callback):
    def __init__(self, save_dir, monitor='train/loss', mode='min', top_k=3):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.best_k_models = []  
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if not is_global_zero():
            return  

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            print(f"[SaveTopK] Epoch {trainer.current_epoch}: No metric {self.monitor} found.")
            return

        score = current_score.item()
        epoch = trainer.current_epoch

        print(f"[SaveTopK] Epoch {epoch} {self.monitor}: {score:.4f}")
        print(f"[SaveTopK] Current saved encoder scores: {[x[0] for x in self.best_k_models]}")

        filename = f"encoder_{self.monitor.replace('/', '_')}_epoch{epoch:03d}_score{score:.4f}.pth"
        save_path = osp.join(self.save_dir, filename)

        should_add = False

        if len(self.best_k_models) < self.top_k:
            should_add = True
        else:
            worst_score, _ = max(self.best_k_models, key=lambda x: x[0]) if self.mode == 'min' else min(self.best_k_models, key=lambda x: x[0])
            if (self.mode == 'min' and score < worst_score) or (self.mode == 'max' and score > worst_score):
                should_add = True

        if should_add:
            torch.save(pl_module.encoder.state_dict(), save_path)
            print(f"[SaveTopK] Saved encoder to {save_path} with {self.monitor}={score:.4f}")

            self.best_k_models.append((score, save_path))

            if len(self.best_k_models) > self.top_k:
                if self.mode == 'min':
                    worst = max(self.best_k_models, key=lambda x: x[0])
                else:
                    worst = min(self.best_k_models, key=lambda x: x[0])

                try:
                    os.remove(worst[1])
                    print(f"[SaveTopK] Removed worst encoder: {worst[1]}")
                except Exception as e:
                    print(f"[SaveTopK] Failed to remove: {worst[1]}, error: {e}")
                self.best_k_models.remove(worst)

class SaveTopKModelCallback(Callback):
    def __init__(self, save_dir, monitor='train/loss', mode='min', top_k=3):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k
        self.best_k_models = []  
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if not is_global_zero():
            return 

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            print(f"[SaveTopK] Epoch {trainer.current_epoch}: No metric {self.monitor} found.")
            return

        score = current_score.item()
        epoch = trainer.current_epoch

        print(f"[SaveTopK] Epoch {epoch} {self.monitor}: {score:.4f}")
        print(f"[SaveTopK] Current saved models scores: {[x[0] for x in self.best_k_models]}")

        filename = f"model_{self.monitor.replace('/', '_')}_epoch{epoch:03d}_score{score:.4f}.pth"
        save_path = osp.join(self.save_dir, filename)

        should_add = False

        if len(self.best_k_models) < self.top_k:
            should_add = True
        else:
            worst_score, _ = max(self.best_k_models, key=lambda x: x[0]) if self.mode == 'min' else min(self.best_k_models, key=lambda x: x[0])
            if (self.mode == 'min' and score < worst_score) or (self.mode == 'max' and score > worst_score):
                should_add = True

        if should_add:
            torch.save(pl_module.state_dict(), save_path)
            print(f"[SaveTopK] Saved full model to {save_path} with {self.monitor}={score:.4f}")

            self.best_k_models.append((score, save_path))

            if len(self.best_k_models) > self.top_k:
                if self.mode == 'min':
                    worst = max(self.best_k_models, key=lambda x: x[0])
                else:
                    worst = min(self.best_k_models, key=lambda x: x[0])

                try:
                    os.remove(worst[1])
                    print(f"[SaveTopK] Removed worst model: {worst[1]}")
                except Exception as e:
                    print(f"[SaveTopK] Failed to remove: {worst[1]}, error: {e}")
                self.best_k_models.remove(worst)


    
