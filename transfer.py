import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Transfer(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        mdeep = models.segmentation.deeplabv3_resnet101(pretrained=True)
        backbone = mdeep.backbone
        l4 = backbone.pop('layer4')
        l3 = backbone.pop('layer3')
        # Layer two ouputs appropriate size
        
        
        for param in backbone.parameters():
            # False implies no retraining
            param.requires_grad=False
        
        self.encoder = backbone
        
        self.decoder = nn.ModuleList([
        
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),),
        
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ),
            
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ),
            
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ),
        
            
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )
        ])
        
        # Final output layer to have a 30 output classes
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
        

        
        

    def forward(self, x):
        
        
        out_encoder = self.encoder(x)
        
        out_decoder = out_encoder
        
        for layer in self.decoder:
            out_decoder = layer(out_decoder)
        
        
        score = self.classifier(out_decoder)                   

        return score  