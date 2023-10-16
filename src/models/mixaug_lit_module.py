from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MixAugLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_classes: int,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = torch.nn.BCEWithLogitsLoss()

        # self.train_acc = Accuracy(task="multilabel", num_labels=num_classes, average='micro')
        # self.val_acc = Accuracy(task="multilabel", num_labels=num_classes, average='micro')
        # self.test_acc = Accuracy(task="multilabel", num_labels=num_classes, average='micro')

        self.train_acc_top1 = Accuracy(task="multilabel", num_labels=num_classes, top_k=1, average='micro')
        self.train_acc_top2 = Accuracy(task="multilabel", num_labels=num_classes, top_k=2, average='micro')

        self.val_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='micro')
        self.val_acc_top2 = Accuracy(task="multiclass", num_classes=num_classes, top_k=2, average='micro')

        self.test_acc_top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average='micro')
        self.test_acc_top2 = Accuracy(task="multiclass", num_classes=num_classes, top_k=2, average='micro')
        print(f"num_classes: {num_classes}")
        



        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc_top1.reset()
        self.val_acc_top2.reset()
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        
        # For multi-label classification
        preds = torch.sigmoid(logits)
        
        return loss, preds, y


    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # Convert soft labels to hard labels for accuracy calculation
        hard_targets = torch.zeros_like(targets)
        hard_targets[torch.arange(targets.size(0)), targets.argmax(1)] = 1

        # update and log metrics
        self.train_loss(loss)

        self.train_acc_top1(preds, hard_targets)
        self.train_acc_top2(preds, hard_targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_top1", self.train_acc_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_top2", self.train_acc_top2, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        _, targets_indices = torch.max(targets, dim=1)
        self.val_acc_top1(preds, targets_indices)
        self.val_acc_top2(preds, targets_indices)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top1", self.val_acc_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top2", self.val_acc_top2, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc_top1.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)

        _, preds_indices = torch.max(preds, dim=1)
        _, targets_indices = torch.max(targets, dim=1)


        self.test_acc_top1(preds_indices, targets_indices)
        self.test_acc_top2(preds_indices, targets_indices)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_top1", self.test_acc_top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_top2", self.test_acc_top2, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
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


if __name__ == "__main__":
    _ = MixAugLitModule(None, None, None, None)
