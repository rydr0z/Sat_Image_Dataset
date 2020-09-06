import torch.nn as nn
import torch

class Neural_Network(nn.Module):
    """16 weight layers"""
    def __init__(self, config, batch_norm=False, init_weights=True, mode="cla-16", rgb=False):
        super(Neural_Network, self).__init__()
        
        self.cfgs = {'7': [64, 'M', 128, 'M', 256, 256, 'M'],
                     '9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
                    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

        self.config = config
        self.batch_norm = batch_norm
        self.mode = mode
        self.rgb = rgb
        self.in_channels = 7
        self.cfg = self.cfgs[self.config]
        self.flatten_channels = 512*7*7

        if self.mode == "cla-16":
          self.out = 16
        if self.mode == "cla-6":
          self.out = 6
        if self.mode == 'reg':
          self.out = 1
        if self.rgb:
          self.in_channels = 3
        if (self.config == "7"):
          self.flatten_channels = 256*7*7
        
        self.convs = self._make_layers(self.cfg, self.in_channels, self.batch_norm)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.fclayers = nn.Sequential(nn.Linear(self.flatten_channels, 4096),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(4096, 4096),
                                      nn.ReLU(),
                                      nn.Dropout(),
                                      nn.Linear(4096, self.out))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 5 convolutional / pooling layers
        x = self.convs(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fclayers(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_channels, batch_norm=False):
      layers = []
      in_channels = in_channels
      for v in cfg:
          if v == 'M':
              layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
          else:
              conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
              if batch_norm:
                  layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
              else:
                  layers += [conv2d, nn.ReLU(inplace=True)]
              in_channels = v
      return nn.Sequential(*layers)