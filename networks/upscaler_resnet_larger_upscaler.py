from networks.upscaler_v2 import EncoderBlock, DecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import ResNet34_Weights, resnet34

class UNetBackbone(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder1 = EncoderBlock(3, hp["n1"], hp["n1"])
        self.encoder2 = EncoderBlock(hp["n1"], hp["n2"], hp["n2"])
        self.encoder3 = EncoderBlock(hp["n2"], hp["n3"], hp["n3"])
        self.encoder4 = EncoderBlock(hp["n3"], hp["n4"], hp["n4"])

        self.decoder4 = DecoderBlock(hp["n4"], hp["u3"], hp["u3"])
        self.decoder3 = DecoderBlock(hp["u3"] + hp["n3"], hp["u2"], hp["u2"])
        self.decoder2 = DecoderBlock(hp["u2"] + hp["n2"], hp["u1"], hp["u1"])    
        self.decoder1 = DecoderBlock(hp["u1"] + hp["n1"], hp["interface"], hp["interface"])
        
    def forward(self, x):
        encoding1 = self.encoder1.forward(x)
        encoding2 = self.encoder2.forward(encoding1)
        encoding3 = self.encoder3.forward(encoding2)
        encoding4 = self.encoder4.forward(encoding3)

        #print(decoding5.shape)
        
        decoding4 = self.decoder4.forward(encoding4)
        
        decoding3 = self.decoder3.forward(torch.cat([encoding3, decoding4], dim=-3))
        #print(decoding3.shape)
        #print(encoding2.shape)
        decoder2_input = torch.cat([decoding3, encoding2], dim=-3)
        #print(decoder2_input.shape)
        decoding2 = self.decoder2.forward(decoder2_input)
        decoding1 = self.decoder1.forward(torch.cat([decoding2, encoding1], dim=-3))
        return decoding1

class UpscalerResNetLarge(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.pretrained_classifier_backbone = torch.nn.Sequential(*(list(resnet34(pretrained=True).children())[:-1]))
        self.unet_backbone = UNetBackbone(hp)
        weights = ResNet34_Weights.DEFAULT
        self.preprocess = weights.transforms()
        
        
        
        self.upscaler = nn.Sequential(
            nn.ConvTranspose2d(hp["interface"] + 256, hp["nupscaler"], 2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(hp["nupscaler"], hp["nupscaler"], 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(hp["nupscaler"], hp["nupscaler"], 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(hp["nupscaler"], 3, 1)
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        unet_output = self.unet_backbone.forward(x)
        #print(unet_output.shape)
        classifier_output = self.unet_backbone.forward(self.preprocess(x))
        #print(classifier_output.shape)

            
        ushape = unet_output.shape
        
        upscaled_classifier = f.interpolate(classifier_output, (ushape[-2], ushape[-1]), mode="bilinear")
        
        
        
        upscaler_input = torch.cat([upscaled_classifier, unet_output], dim=-3)
        #print(upscaler_input.shape)
        y = self.upscaler.forward(upscaler_input)
        return y