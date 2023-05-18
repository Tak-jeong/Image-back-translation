import torch
import torchvision.models as models

class ConvNeXt(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False, num_classes: int = 1000) -> None:
        super().__init__()
        if model_name in models.__dict__:
            self.model = models.__dict__[model_name](pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError(f"Model '{model_name}' not found in torchvision.models")

    def forward(self, x):
        return self.model(x)
