import torch
import torch.nn as nn
import torchvision

## pretrained vgg model for feature extraction
class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      vgg19 = torchvision.models.vgg19(True)
      self.features = vgg19.features
      self.fc = nn.Sequential(*list(vgg19.classifier[0:-2]))  #RELU

  def forward(self, x):
      x = self.features(x)
      x = x.view(x.size(0),-1)
      x = self.fc(x)
      return x
