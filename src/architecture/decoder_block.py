import torch.nn as nn
from loguru import logger

class Decoder_Block(nn.Module):
    def __init__ (self,num_layers=2,initial_filter=1024):
        super(Decoder_Block,self).__init__()
        self.relu = nn.ReLU()
        self.layers = []
        out_channel = initial_filter//2
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(initial_filter,out_channel,kernel_size=3,padding=1))
            self.layers.append(nn.ReLU())
            initial_filter = out_channel
        self.block = nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.block(x)
        return x 