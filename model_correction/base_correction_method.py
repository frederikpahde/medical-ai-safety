import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from utils.layer_names import get_lnames_sorted
from utils.training_utils import get_optimizer, get_loss
from utils.metrics import get_accuracy, get_f1, get_auc

class LitClassifier(pl.LightningModule):
    def __init__(self, model, config, **kwargs):
        super().__init__()
        self.loss = None
        self.optim = None
        self.model = model
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def default_step(self, x, y, stage):
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict(
            {f"{stage}_loss": loss,
             f"{stage}_acc": self.get_accuracy(y_hat, y),
             f"{stage}_auc": self.get_auc(y_hat, y),
             f"{stage}_f1": self.get_f1(y_hat, y),
             },
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.default_step(x, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="valid")

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.default_step(x, y, stage="test")

    def set_optimizer(self, optim_name, params, lr, ckpt_path):
        self.optim = get_optimizer(optim_name, params, lr, ckpt_path)

    def set_loss(self, loss_name, weights=None):
        self.loss = get_loss(loss_name, weights)

    def configure_optimizers(self):
        sche = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=[80, 120], gamma=0.1)
        scheduler = {
            "scheduler": sche,
            "name": "lr_history",
        }

        return [self.optim], [scheduler]

    @staticmethod
    def get_accuracy(y_hat, y):
        return get_accuracy(y_hat, y)

    @staticmethod
    def get_f1(y_hat, y):
        return get_f1(y_hat, y)

    @staticmethod
    def get_auc(y_hat, y):
        return get_auc(y_hat, y)

    def state_dict(self, **kwargs):
        return self.model.state_dict()


class Vanilla(LitClassifier):
    def __init__(self, model, config):
        super().__init__(model, config)

    def configure_callbacks(self):
        return [
            Freeze(stop_at_layer=self.config['layer_name'])
        ]

class Freeze(Callback):
    def __init__(self, stop_at_layer=None, stop_before=False):
        super().__init__()
        self.stop_at_layer = stop_at_layer
        self.stop_before = stop_before

        self.freeze_types = [
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.Conv1d,
            torch.nn.Conv2d,
            torch.nn.Linear,
            torch.nn.LayerNorm,
            torch.nn.MultiheadAttention
        ]

    def check_freeze_layer(self, layer_type):
        for freeze_type in self.freeze_types:
            if isinstance(layer_type, freeze_type):
                return True
        return False

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"Freezing conv+bn layers. Up to {self.stop_at_layer}")
        lnames_sorted = get_lnames_sorted(pl_module.model)
        for n, m in pl_module.model.named_modules():
            freeze_layer = lnames_sorted.index(self.stop_at_layer) >= lnames_sorted.index(n)
            if freeze_layer and self.check_freeze_layer(m):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                print(f"Freeze {n}")

        ## Freeze extra ViT layers
        params_blacklist = [
            # ViT
            "class_token", "encoder.pos_embedding",
            "cls_token", "pos_embed",
            ]
        for n, p in pl_module.model.named_parameters():
            if any([p in n for p in params_blacklist]):
                p.requires_grad = False

        layers_to_optimize = [n for n, m in pl_module.model.named_parameters() if m.requires_grad]
        print(f"Done. Optimizing {layers_to_optimize}")