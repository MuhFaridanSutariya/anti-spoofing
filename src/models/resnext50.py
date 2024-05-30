import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

class SEResNeXT50(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.resnext = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

        for param in self.resnext.parameters():
            param.requires_grad = False

        in_features = self.resnext.fc.in_features

        self.resnext.fc = nn.Identity()

        # Linear 1
        self.fc1 = nn.Linear(in_features=in_features, out_features=512)

        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        # Linear 2
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        # Classifier
        self.classifier = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        out = self.resnext(x)
        out = out.view(out.size(0), -1)

        # Linear 1
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Linear 2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Classifier
        out = self.classifier(out)
        return out

# # Download the model
# model = SEResNeXT50(input_shape=(3,224,224), num_classes=2)
# torch.save(model.state_dict(), 'SEResNeXT50_v0.ckpt')
