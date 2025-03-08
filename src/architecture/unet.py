import torch
import torch.nn as nn
from .encoder_block import Encoder_Block
from .decoder_block import Decoder_Block
from loguru import logger

class UNet(nn.Module):
    def __init__(self,model_depth=5,low_dim=64,high_dim=1024,in_channel=3,n_class=2,device="cpu"):
        super(UNet,self).__init__()
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.upsample_blocks = []
        self.model_depth = model_depth
        self.device = device
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.in_channel = in_channel
        self.encoder_output = []
        self.dec_out = None
        self.build_encoder()
        self.build_decoder()

        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.out_conv = nn.Conv2d(self.low_dim,n_class,kernel_size=1)

    def build_encoder(self):
        final_channel = self.high_dim
        temp = self.in_channel 
        for i in reversed(range(self.model_depth)):
            initial_filter = final_channel//(2**(i))
            self.encoder_blocks.append(Encoder_Block(in_channel=temp,initial_filter=initial_filter,num_layers=2).to(self.device))         
            temp=initial_filter

    def build_decoder(self):
        self.decoder_blocks = []
        for i in reversed(range(1,self.model_depth)):
            initial_filter = self.low_dim*(2**(i))
            self.upsample_blocks.append(nn.ConvTranspose2d(initial_filter,initial_filter//2,2,2,padding=1,output_padding=1).to(self.device))
            self.decoder_blocks.append(Decoder_Block(initial_filter=initial_filter,num_layers=2).to(self.device))

    def forward(self,x):
        for i in range(self.model_depth):
            x = self.encoder_blocks[i](x)
            self.encoder_output.append(x)
            if self.model_depth-i != 1:
                x = self.downsample(x)
        
        
        for i in range(self.model_depth-1):
            if i ==0:
                temp = self.upsample_blocks[i](self.encoder_output[i-1])
            elif i != self.model_depth-1:
                temp = self.upsample_blocks[i](self.dec_out)
            shape1 = self.encoder_output[-i-2].shape[2]
            shape2 = temp.shape[2]
            start = (shape1 - shape2) // 2  
            end = start + shape2  
            cropped_tensor = self.encoder_output[-i-2][:, :, start:end, start:end]  
            cat_tensor = torch.cat([cropped_tensor,temp],dim=1)
            self.dec_out = self.decoder_blocks[i](cat_tensor)

        final_out = self.out_conv(self.dec_out)
        return final_out

if __name__ == "__main__" :
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device : {device}")
    test_tensor = torch.randn(1, 3, 572, 572).to(device)
    model  = UNet()
    model.to(device)
    output = model(test_tensor)
    print(output.shape,output.device)
