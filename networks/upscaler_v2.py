import torch
from torch import nn
import torch.nn.functional as f

class EncoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels, internal_channels, kernel_size=3):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, internal_channels, kernel_size=kernel_size, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(internal_channels + input_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.MaxPool2d(2, 2, padding=0),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        first_encoder_output = self.encoder1.forward(x)
        second_encoder_input = torch.cat([first_encoder_output, x], dim=-3)
        y = self.encoder2.forward(second_encoder_input)
        return y
    
class DecoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels, internal_channels, up_kernel_size=2, kernel_size=3):
        super().__init__()
        self.enlarger = nn.Sequential(
            nn.ConvTranspose2d(input_channels, internal_channels, up_kernel_size, up_kernel_size),
            nn.LeakyReLU()
        )
        
        self.res_block=nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(2 * internal_channels, out_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        enlarged = self.enlarger.forward(x)
        #upscaled = f.interpolate(x, scale_factor=2)
        res_block_output = self.res_block.forward(enlarged)

        #print(res_block_output.shape)
        #print(enlarged.shape)
        
        final_input = torch.cat([enlarged, res_block_output], dim=-3)
        y = self.final.forward(final_input)
        return y


class Upscaler2NN(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder1 = EncoderBlock(3, hp["n1"], hp["n1"])
        self.encoder2 = EncoderBlock(hp["n1"], hp["n2"], hp["n2"])
        self.encoder3 = EncoderBlock(hp["n2"], hp["n3"], hp["n3"])
        self.encoder4 = EncoderBlock(hp["n3"], hp["n3"], hp["n4"])
        self.encoder5 = EncoderBlock(hp["n4"], hp["n5"], hp["n5"])
        
        self.decoder5 = DecoderBlock(hp["n5"], hp["u4"], hp["u4"])
        self.decoder4 = DecoderBlock(hp["u4"] + hp["n4"], hp["u3"], hp["u3"])
        self.decoder3 = DecoderBlock(hp["u3"] + hp["n3"], hp["u2"], hp["u2"])
        self.decoder2 = DecoderBlock(hp["u2"] + hp["n2"], hp["u1"], hp["u1"])    
        self.decoder1 = DecoderBlock(hp["u1"] + hp["n1"], hp["interface"], hp["interface"])
    
        
        self.upscaler1 = nn.Sequential(
            nn.ConvTranspose2d(hp["interface"], hp["nupscaler"], 2, 2),
            nn.LeakyReLU(),


        )
        
        self.upscaler2 = nn.Sequential(
            nn.Conv2d(hp["nupscaler"], hp["nupscaler"], 1, 1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(hp["nupscaler"], hp["nupscaler"], 1, 1, padding = "same"),
            nn.LeakyReLU(),
        )
        
        self.upscaler3 = nn.Sequential(
            nn.Conv2d(2 * hp["nupscaler"], 3, 1, 1)
        )

        
    def forward(self, x):
        encoding1 = self.encoder1.forward(x)
        encoding2 = self.encoder2.forward(encoding1)
        encoding3 = self.encoder3.forward(encoding2)
        encoding4 = self.encoder4.forward(encoding3)
        encoding5 = self.encoder5.forward(encoding4)
        #print(encoding5.shape)
        decoding5 = self.decoder5.forward(encoding5)
        #print(decoding5.shape)
        
        decoding4 = self.decoder4.forward(torch.cat([decoding5, encoding4], dim=-3))
        
        decoding3 = self.decoder3.forward(torch.cat([encoding3, decoding4], dim=-3))
        #print(decoding3.shape)
        #print(encoding2.shape)
        decoder2_input = torch.cat([decoding3, encoding2], dim=-3)
        #print(decoder2_input.shape)
        decoding2 = self.decoder2.forward(decoder2_input)
        decoding1 = self.decoder1.forward(torch.cat([decoding2, encoding1], dim=-3))
        
        upscaler1_output = self.upscaler1.forward(decoding1)
        upscaler2_output = self.upscaler2.forward(upscaler1_output)
        #print(upscaler1_output.shape)
        #print(upscaler2_output.shape)
        y = self.upscaler3.forward(torch.cat([upscaler1_output, upscaler2_output], dim = -3))
        
        #print(upscaler1_output.shape)
        #print(decoding1.shape)
        
        

        
        return y