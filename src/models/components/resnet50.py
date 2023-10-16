import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ClassificationModel(nn.Module):
    
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        
        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        self.image_processor  = None

        if backbone == "resnet50":
        
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)

        elif backbone == "deit":

            self.base_model = DeiTModel.from_pretrained(
                "facebook/deit-base-distilled-patch16-224")
            self.out = nn.Linear(768, num_classes)
        
    def forward(self, image):
        
        x = image

        if self.backbone == "resnet50":
            
            with torch.no_grad():

                x = self.base_model.conv1(x)
                x = self.base_model.bn1(x)
                x = self.base_model.relu(x)
                x = self.base_model.maxpool(x)

                x = self.base_model.layer1(x)
                x = self.base_model.layer2(x)
                x = self.base_model.layer3(x)
                x = self.base_model.layer4(x)

                x = self.base_model.avgpool(x)
                x = torch.flatten(x, 1)

        elif self.backbone == "deit":
            
            with torch.no_grad():

                x = self.base_model(x)[0][:, 0, :]
            
        return self.out(x)