import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def imread(img_name, device):
    """
    loads an image as torch.tensor on the selected device
    """ 
    np_img = plt.imread(img_name)
    tens_img = torch.tensor(np_img, dtype=torch.float, device=device)
    if torch.max(tens_img) > 1:
        tens_img/=255
    if len(tens_img.shape) < 3:
        tens_img = tens_img.unsqueeze(2)
    if tens_img.shape[2] > 3:
        tens_img = tens_img[:,:,:3]
    tens_img = tens_img.permute(2,0,1)
    return tens_img.unsqueeze(0)

def imshow(tens_img):
    """
    shows a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
        ax = plt.imshow(np_img)
        ax.set_cmap('gray')
    else:
        ax = plt.imshow(np_img)
    plt.axis('off')
    return plt.show()

def imsave(save_name, tens_img):
    """
    save a tensor image
    """ 
    np_img = np.clip(tens_img.squeeze(0).permute(1,2,0).data.cpu().numpy(), 0,1)
    if np_img.shape[2] < 3:
        np_img = np_img[:,:,0]
    plt.imsave(save_name, np_img)
    return 

def im2pat(img, patch_size):
    """
    extract patches from an image
    """ 
    myunfold = nn.Unfold(kernel_size=patch_size)
    return myunfold(img).squeeze(0).transpose(1,0)

def pat2im(patches, img_shape, patch_size):
    """
    uniform aggregation
    """ 
    y = patches.transpose(0,1).unsqueeze(0)
    uns = torch.ones(y.shape, device=y.device)
    myfold = nn.Fold(output_size=img_shape, kernel_size=patch_size)
    return myfold(y)/myfold(uns)

def psnr(img1, img2):
    """
    compute PSNR between two images
    """
    MSE = torch.mean((img1-img2)**2)
    return 10*torch.log10(1**2/MSE)