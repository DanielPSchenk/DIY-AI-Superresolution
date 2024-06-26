import gc
from tqdm import tqdm
import torch
from torch.nn import L1Loss
import numpy as np
from networks.perceptual_loss import FeatureReconstructionLoss, StyleReconstructionLoss

#optimizer.lr=1e-3
def train(model, optimizer, epochs, train_loader, val_loader, train_losses, val_losses, device="cuda", loss_func = L1Loss()):
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    
    for epoch in range(epochs):
        bar = tqdm(zip(range(len(train_loader)), train_loader), total=len(train_loader), desc="train {:0>5}".format(epoch + 1), ncols=100)
        train_loss_sum = 0
        val_loss_sum = 0
        model.to(device)
        for i, batch in bar:
            down_image, target = batch
            gc.collect()
            torch.cuda.empty_cache()
        
            #down_image = f.interpolate(gpu_batch, scale_factor=(.5, .5), mode="bilinear", antialias=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                model.train()
                down_image = down_image.to(device)
                model.zero_grad()
                loss_func.zero_grad()
                prediction = model.forward(down_image)
        

                #target = (gpu_batch - f.interpolate(down_image, scale_factor=(2,2), mode="bilinear"))*2
                del down_image
                target = target.to(device)
                loss = loss_func(prediction, target)

                del target
                del prediction
                loss.backward()
                train_loss_sum += loss.item()
        
                optimizer.step()
            bar.set_postfix(loss = "{:.8f}".format(train_loss_sum / (i + 1)))

        

        
        train_losses.append(train_loss_sum / np.max([len(train_loader), 1]))
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        bar = tqdm(zip(range(len(val_loader)), val_loader), total=len(val_loader), desc="val {:0>5}".format(epoch + 1), ncols=100)
        for i, batch in bar:
            down_image, target = batch
            #down_image = f.interpolate(gpu_batch, scale_factor=(.5, .5), mode="bilinear", antialias=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                down_image, target = down_image.to(device), target.to(device)
                prediction = model.forward(down_image)
        
                #target = (gpu_batch - f.interpolate(down_image, scale_factor=(2,2), mode="bilinear"))*2
                del down_image
                loss = loss_func(prediction, target)
                val_loss_sum += loss.item()
        
                bar.set_postfix(loss = "{:.8f}".format(val_loss_sum / (i+ 1)))
        
        
                #del gpu_batch
                del target
                del prediction
        val_losses.append(val_loss_sum /np.max([len(val_loader), 1]))
        #optimizer.lr = optimizer.lr * .98
    
    #model.eval()
    #prediction = model.forward(val_down_image)
    #loss = loss_func(prediction, val_target)
    #val_losses.append(loss.item())
   
   
def train_feature_reconstruction(model, optimizer, epochs, train_loader, val_loader, train_losses, val_losses, train_loss_interval, val_loss_interval, device="cuda"):
    model.to(device)
    gc.collect()
    torch.cuda.empty_cache()
    loss_func = StyleReconstructionLoss()
    loss_func = loss_func.to(device)
    
    for epoch in range(epochs):
        bar = tqdm(zip(range(len(train_loader)), train_loader), total=len(train_loader), desc="train {:0>5}".format(epoch + 1), ncols=100)
        train_loss_sum = 0
        val_loss_sum = 0
        model.to(device)
        for i, batch in bar:
            down_image, target = batch
            gc.collect()
            torch.cuda.empty_cache()
        
            #down_image = f.interpolate(gpu_batch, scale_factor=(.5, .5), mode="bilinear", antialias=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                model.train()
                down_image = down_image.to(device)
                model.zero_grad()
                prediction = model.forward(down_image)
        

                #target = (gpu_batch - f.interpolate(down_image, scale_factor=(2,2), mode="bilinear"))*2
                
                target = target.to(device)
                loss = loss_func(prediction, target, down_image)
                del down_image
                del target
                del prediction
                loss.backward()
                train_loss_sum += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .0001)
                optimizer.step()
            bar.set_postfix(loss = "{:.8f}".format(train_loss_sum / (i + 1)))
            if int(i + 1) % train_loss_interval == 0:
                train_losses.append(train_loss_sum / (i + 1))

        

        
        #train_losses.append(train_loss_sum / np.max([len(train_loader), 1]))
        gc.collect()
        torch.cuda.empty_cache()
        model.eval()
        bar = tqdm(zip(range(len(val_loader)), val_loader), total=len(val_loader), desc="val {:0>5}".format(epoch + 1), ncols=100)
        for i, batch in bar:
            gc.collect()
            torch.cuda.empty_cache()
            down_image, target = batch
            #down_image = f.interpolate(gpu_batch, scale_factor=(.5, .5), mode="bilinear", antialias=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                down_image, target = down_image.to(device), target.to(device)
                prediction = model.forward(down_image)
        
                #target = (gpu_batch - f.interpolate(down_image, scale_factor=(2,2), mode="bilinear"))*2
                
                loss = loss_func(prediction, target, down_image)
                del down_image
                val_loss_sum += loss.item()
        
                bar.set_postfix(loss = "{:.8f}".format(val_loss_sum / (i+ 1)))
                if int(i + 1) % val_loss_interval == 0:
                    val_losses.append(val_loss_sum / (i+ 1))
        
                #del gpu_batch
                del target
                del prediction
        gc.collect()
        torch.cuda.empty_cache()
        #val_losses.append(val_loss_sum /np.max([len(val_loader), 1]))