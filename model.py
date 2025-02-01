import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class FruitDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(FruitDetectionModel, self).__init__()
        # Use EfficientNet-B0 as base model (better than ResNet18 for this task)
        self.model = models.efficientnet_b0(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False
            
        # Replace classifier
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def get_transform(is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
