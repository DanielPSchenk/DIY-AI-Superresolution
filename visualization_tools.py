import random
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as f

def area_half_size(area, scale = 2):
    u, v, w, x = area
    return (int(u / scale), int(v / scale), int(w / scale), int(x / scale))

def random_image_slice(image, size=100):
    #print(image.shape)
    first = random.randint(0, image.shape[-2] - size)  # Generate a random integer for the first number
    second = first + size  # Calculate the second number based on the offset
    third = random.randint(0, image.shape[-1] - size)  # Generate a random integer for the third number
    fourth = third + size  # Calculate the fourth number based on the offset
    return (first, second, third, fourth)

def crop(image, area):
    return image[area[0]:area[1], area[2]:area[3], :]

def crop_tensor(image, area):
    return image[:, area[2]:area[3], area[0]:area[1]]

def plot_upscaling(pair, model):
    down_image, target = pair
    model.to("cpu")
    rows = 6
    fig, axes = plt.subplots(rows, 6, figsize=(12, 3))
    fig.set_size_inches(15, 15)
    down_image = down_image.unsqueeze(0)
    up_image = f.interpolate(down_image, scale_factor=(2,2), mode="bilinear").squeeze(0)
    down_image = down_image.squeeze(0)
    image = (up_image + .5 * target).squeeze(0)
    prediction_unprocessed = model.forward(down_image).detach()
    if len(prediction_unprocessed.shape) == 4:
        prediction_unprocessed = prediction_unprocessed.squeeze(0)
    
    prediction = up_image + prediction_unprocessed / 2


    for i in range(rows):
        area = random_image_slice(image)
        area_l = area_half_size(area)
    
        image1 = crop_tensor(down_image, area_l).cpu().permute(1,2,0)
        image2 = crop_tensor(up_image, area).cpu().permute(1,2,0)
        image3 = crop_tensor(prediction, area).cpu().permute(1,2,0)
        image4 = crop_tensor(image, area).cpu().permute(1,2,0)
        image5 = crop_tensor(prediction_unprocessed, area).cpu().permute(1,2,0) + .5
        image6 = crop_tensor(target, area).cpu().permute(1,2,0) + .5
    
        axes[i, 0].imshow(image1)
        axes[0, 0].set_title('')

        axes[i, 1].imshow(image2)
        axes[0, 1].set_title('blurry upscale')

        axes[i, 2].imshow(image3)
        axes[0, 2].set_title('prediction')

        axes[i, 3].imshow(image4)
        axes[0, 3].set_title('Ground truth')

        axes[i, 4].imshow(image5)
        axes[0, 4].set_title('Raw output')

        axes[i, 5].imshow(image6)
        axes[0, 5].set_title('Raw target')

    plt.show()
    del image
    del down_image
    del up_image
    del target
    del prediction_unprocessed
    del prediction


