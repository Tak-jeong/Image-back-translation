import torch
from timm import create_model

class VisionTransformer(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False, num_classes: int = 1000) -> None:
        super().__init__()
        self.model_name = model_name
        if pretrained:
            self.model = create_model(model_name, pretrained=True, num_classes=num_classes)
        else:
            self.model = create_model(model_name, pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
