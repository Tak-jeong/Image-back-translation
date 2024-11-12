import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric, MaxMetric
from typing import Any, Tuple

class CapMixLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        num_classes: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.criterion = nn.CrossEntropyLoss()

        # Accuracy metrics
        self.train_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='micro')
        self.train_acc_top2 = Accuracy(task="multiclass", num_classes=num_classes, top_k=2, average='micro')

        self.val_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='micro')
        self.val_acc_top2 = Accuracy(task="multiclass", num_classes=num_classes, top_k=2, average='micro')

        self.test_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='micro')
        self.test_acc_top2 = Accuracy(task="multiclass", num_classes=num_classes, top_k=2, average='micro')

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc_top1.reset()
        self.val_acc_top2.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, (label1, label2, lam) = batch
        logits = self.forward(x)
        
        loss = (lam * self.criterion(logits, label1) + (1 - lam) * self.criterion(logits, label2)).mean()
        
        return loss, logits, label1, label2, lam
    
    def weighted_accuracy(self, logits, label1, label2, lam):
        acc1 = (logits.argmax(dim=1) == label1).float()
        acc2 = (logits.argmax(dim=1) == label2).float()
        return lam * acc1 + (1 - lam) * acc2

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, label1, label2, lam = self.model_step(batch)

        # Calculate weighted accuracy for all data
        weighted_acc = self.weighted_accuracy(logits, label1, label2, lam)
        self.log("train/weighted_acc", weighted_acc.mean(), on_step=False, on_epoch=True, prog_bar=True)

        # Calculate top1 and top2 accuracy using label1 (original label for non-mixcut data)
        self.train_acc_top1(logits, label1)
        self.train_acc_top2(logits, label1)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_top1", self.train_acc_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_top2", self.train_acc_top2, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, label1, label2, lam = self.model_step(batch)
        
        self.val_acc_top1(logits, label1)
        self.val_acc_top2(logits, label1)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top1", self.val_acc_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top2", self.val_acc_top2, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        acc = self.val_acc_top1.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, logits, label1, label2, lam = self.model_step(batch)

        if label2 is not None:
            self.test_acc_top1(logits, label1)
            self.test_acc_top1(logits, label2)
            self.test_acc_top2(logits, label1)
            self.test_acc_top2(logits, label2)
        else:
            self.test_acc_top1(logits, label1)
            self.test_acc_top2(logits, label1)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_top1", self.test_acc_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_top2", self.test_acc_top2, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


