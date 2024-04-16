import gc
from tqdm import tqdm
import torch
from torch.nn import L1Loss

#optimizer.lr=1e-3
def train(model, optimizer, epochs, train_loader, val_loader, train_losses, val_losses, device="cuda", loss_func = L1Loss()):
    model.to(device)
    
    
    for epoch in range(epochs):
        bar = tqdm(zip(range(len(train_loader)), train_loader), total=len(train_loader), desc="train {:0>5}".format(epoch + 1), ncols=100)
        train_loss_sum = 0
        val_loss_sum = 0
        for i, batch in bar:
            down_image, target = batch
            
        
            #down_image = f.interpolate(gpu_batch, scale_factor=(.5, .5), mode="bilinear", antialias=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                model.train()
                down_image = down_image.to(device)
                model.zero_grad()
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
   