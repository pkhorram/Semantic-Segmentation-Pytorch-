import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.encoder = nn.ModuleList([ 
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(32),
                self.relu,
                ),
            
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(64),
                self.relu,
                ),
            
            
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(128),
                self.relu,
                ),
            
            nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(256),
                self.relu,
                ),
            
            
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1),
                nn.BatchNorm2d(512),
                self.relu,
                ),
        ])
        
        self.decoder = nn.ModuleList([
        
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(512),
                self.relu,
                ),
        
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(256),
                self.relu,
                ),
            
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(128),
                self.relu,
                ),
            
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(64),
                self.relu,
                ),
        
            
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                nn.BatchNorm2d(32),
                self.relu,
                )
        ])
        
        # Final output layer to have a 30 output classes
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

        

        
        

    def forward(self, x):
        
        
        for layer in self.encoder:
            # Can use this to append to layers on reverse operations
            x = layer(x)
            #out_encoder = nn.MaxPool2d(2)(out_encoder)
            
            
        for layer in self.decoder:
            x = layer(x)
            
        del layer  
        pred = self.classifier(x)
        del x
        
        return pred