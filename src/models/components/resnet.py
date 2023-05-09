import torch
import torchvision.models as models

class ResNet(torch.nn.Module):
    def __init__(self, model_name: str, pretrained: bool = False, num_classes: int = 1000) -> None:
        super().__init__()
        self.model = getattr(models, model_name)(pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# def ResNet18D(pretrained: bool = False, num_classes: int = 10, **kwargs):
#     model = torchvision.models.resnet18(pretrained=pretrained, **kwargs)
#     model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#     return model
