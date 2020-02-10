import torch.nn as nn
import torch.nn.functional as F
import torch

class Unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.encoder = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Consider removing padding
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                ),
            
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                ),
            
            nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256,256, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                ),
            
            
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                ),
            nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                ),
            ])
        
        
        self.upConvs = nn.ModuleList([
            
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                
                ),
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
            
            ])
        self.decoder = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(1024,512, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(512,512, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                ),
            nn.Sequential(
                nn.Conv2d(512,256, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256,256, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                ),
            nn.Sequential(
                nn.Conv2d(256,128, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                ),
            nn.Sequential(
                nn.Conv2d(128,64, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                ),
        ])
        
        # Final output layer to have a 30 output classes
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)


        
        

    def forward(self, x):
        
        
        out_encoder = x
        skipConnections = []
        for layer in self.encoder:
            # Can use this to append to layers on reverse operations
            out_encoder = layer(out_encoder)
            skipConnections.append(out_encoder)
            out_encoder = nn.MaxPool2d(2,2)(out_encoder)
            
        
        # First up scaling which will need to be concatenated with last pooled
        skipConnections.reverse()
        out_decoder = out_encoder
        
        for i, layer in enumerate(self.decoder):
            # First upscale
            out_decoder = self.upConvs[i](out_decoder)
            # Merge with corresponding output
            out_decoder = torch.cat([skipConnections[i], out_decoder], axis=1)
            # Convolutions
            out_decoder = layer(out_decoder)
        
        
        score = self.classifier(out_decoder)                   

        return score  