import torch
from torch import nn
import torch.nn.functional as f


class OverfitNN(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, hp["n1"], 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(hp["n1"], hp["n1"], 3, padding='same'),
            nn.MaxPool2d(2, 2, padding=0),
            nn.LeakyReLU(),
        )
        self.encoder2 = nn.Sequential(    
            nn.Conv2d(hp["n1"], hp["n2"], 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(hp["n2"], hp["n2"], 3, padding='same'),
            nn.MaxPool2d(2, 2, padding=0),
            nn.LeakyReLU(),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(hp["n2"], hp["n3"], 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(hp["n3"], hp["n3"], 3, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU()            
        )
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(hp["n3"], hp["n4"], 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(hp["n4"], hp["n4"], 3, padding="same"),
            nn.MaxPool2d(2,2),
            nn.LeakyReLU()
        )
        
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(hp["n4"], hp["u3"], 2, 2),
            nn.LeakyReLU()
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(hp["n3"] + hp["u3"], hp["u2"], 2, 2),
            nn.LeakyReLU()
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(hp["u2"] + hp["n2"], hp["u1"], 2, 2),
            nn.LeakyReLU()
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(hp["u1"] + hp["n1"], hp["interface"], 2, 2)
        )
        
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
        
        decoding4 = self.decoder4.forward(encoding4)
        
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