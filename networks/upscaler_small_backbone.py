import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models import ResNet34_Weights, resnet34


class ResBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, padding="same")
        )
        
    def forward(self, x):
        output = self.network.forward(x)
        y = output + x
        return y

    

class ResNet(nn.Module):
    def __init__(self, hp) -> None:
        super().__init__()
        
        self.num_blocks = hp["num_blocks"]
        dimension = hp["backbone_dimension"]
        '''
        self.transform_dimension = nn.Sequential(
            nn.Conv2d(3, dimension, 1, padding = "same"),
            nn.LeakyReLU()
        )'''
        self.blocks = [ResBlock(dimension) for i in range(self.num_blocks)]
        for block, i in zip(self.blocks, range(len(self.blocks))):
            self.add_module("bl_" + str(i), block)
        
    def forward(self, x):
        output = x
        for block in self.blocks:
            output = block.forward(output)
            
        return output



   
def get_layer_dims(layer_list):
    input = torch.zeros((1, 3, 256, 256))
    out_dims = []
    for layer in layer_list:
        layer.eval()
        #print(input.shape)
        input = layer.forward(input)
        
        out_dims.append(input.shape)
        
    return out_dims


class UpscalerResNet(nn.Module):
    def __init__(self, hp):
        super().__init__()
               
        weights = ResNet34_Weights.DEFAULT
        self.resnet_preprocess = weights.transforms()
        layer_dim = hp["layer_dim"]
        
        classifier_list = list(resnet34(pretrained=True).children())[:hp["depth"]]
        
        for classifier in classifier_list:
            for param in classifier.parameters():
                param.requires_grad = False
        
        self.pretrained_classifier_layers = [classifier_list[i] for i in range(0, len(classifier_list)-1)]
        for cl, i in zip(self.pretrained_classifier_layers, range(len(self.pretrained_classifier_layers))):
            self.add_module("cl_" + str(i), cl)
        #print(len(self.pretrained_classifier_layers))
        
        self.out_dims = get_layer_dims(self.pretrained_classifier_layers)
        self.out_channels = [dim[-3] for dim in self.out_dims]
        #print(self.out_dims)
        
        self.reshape_filters = [nn.Conv2d(layer_dim[i + 1] + self.out_channels[i], layer_dim[i], 1) for i in range(1, len(self.out_dims) - 1)] + [nn.Conv2d(self.out_channels[-1], layer_dim[-1], 1)]
        for rf, i in zip(self.reshape_filters, range(len(self.reshape_filters))):
            self.add_module("rf_" + str(i), rf)
        
        self.decoder_blocks = [ResBlock(layer_dim[i]) for i in range(len(self.out_dims))]
        for db, i in zip(self.decoder_blocks, range(len(self.decoder_blocks))):
            self.add_module("db_" + str(i), db)
        
        self.upscaler = nn.Sequential(nn.Conv2d(layer_dim[0] + 3, hp["backbone_dimension"], 1), nn.LeakyReLU(), ResNet({"backbone_dimension" : hp["backbone_dimension"], "num_blocks" : 1}), nn.Conv2d(hp["backbone_dimension"], 3, 1))
        self.add_module("up", self.upscaler)
        
    def parameter_report(self):
        reshape_parameters = sum(sum(p.numel() for p in filter.parameters()) for filter in self.reshape_filters)
        decoder_parameters = sum(sum(p.numel() for p in block.parameters()) for block in self.decoder_blocks)
        resnet_parameters = sum(p.numel() for p in self.upscaler.parameters())
        
        print("{} parameters in reshape filters".format(reshape_parameters))
        print("{} parameters in decoder blocks".format(decoder_parameters))
        print("{} parameters in upscaler".format(resnet_parameters))
        
        
    def forward(self, x):
        classifier_outputs = []
        input = x
        for layer in self.pretrained_classifier_layers:
            input = layer.forward(input)
            classifier_outputs.append(input)
            
        previous_output = self.decoder_blocks[-1].forward(self.reshape_filters[-1].forward(classifier_outputs[-1]))
        #print(previous_output.shape)
        
        for i in range(2, len(classifier_outputs)):
            current_shape = classifier_outputs[-i].shape
            reshaped_previous_output = f.interpolate(previous_output, (current_shape[-2], current_shape[-1]), mode="bilinear")
            concatenation = torch.cat([reshaped_previous_output, classifier_outputs[-i]], dim=-3)
            reshape = self.reshape_filters[-i].forward(concatenation)
            decoding = self.decoder_blocks[-i].forward(reshape)
            
            previous_output = decoding
            #print(previous_output.shape)
            
        target_size = (x.shape[-2] * 2, x.shape[-1] * 2)
        upscaled_output = f.interpolate(previous_output, target_size, mode="bilinear")
        upscaled_input = f.interpolate(x, target_size, mode="bilinear")
        
        upscaler_input = torch.cat([upscaled_output, upscaled_input], dim=-3)
        
        y = self.upscaler.forward(upscaler_input)
        return y
        
    def parameters(self):
        parameter_list = list(self.upscaler.parameters())
        for block in self.decoder_blocks:
            parameter_list = parameter_list + list(block.parameters())
            
        for filter in self.reshape_filters:
            parameter_list = parameter_list + list(filter.parameters())
            
        return parameter_list 
            
        
        