import torch
import os
import numpy as np
import torch.nn as nn
from torch import Tensor, from_numpy, randn, full
from torch.autograd.variable import Variable
from random import uniform

from monai.transforms import (
    RandFlip,
    RandRotate,
    RandAdjustContrast,
    RandSpatialCrop,
    Resize
)

def augmentation(scan):
    
    dim=(scan.shape[0],scan.shape[1],scan.shape[2])

    rand_flip = RandFlip(prob=1,spatial_axis=2)

    rand_rotate = RandRotate(prob=1, range_x=[-np.pi/8,np.pi/8])

    rand_contrast = RandAdjustContrast(prob=0.5, gamma = [0.9,1.2])

    min_crop_scale=0.5
    crop_scale=uniform(min_crop_scale,1)
    rand_crop = RandSpatialCrop(
        roi_size=(round(dim[0]*crop_scale),round(dim[1]*crop_scale),round(dim[2]*crop_scale)), 
        max_roi_size=None, 
        random_center=True, 
        random_size=False,
    )

    resize = Resize(spatial_size=dim)


    x=scan[None,:]
    x = rand_flip(x)
    x = rand_rotate(x)
    x = rand_contrast(x)
    x = rand_crop(x)
    x = resize(x)

    image = torch.squeeze(x)

    return image

def custom_grad_hook(module, grad_input, grad_output):
    """
    This function creates a hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # get stored mask
    custom_change = 0 # leave empty unless we want to do something for the grad
    # multiply with mask
    new_grad = grad_input[0] + 0.2 * (grad_input[0] * custom_change) #0.2 is a hyperparameter
    return (new_grad, )

def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)






def images_to_vectors(images: Tensor) -> Tensor:
    """ converts (Nx28x28) tensor to (Nx784) torch tensor """
    return images.view(images.size(0), 32 * 32)


def images_to_vectors_numpy(images: np.array) -> Tensor:
    """ converts (Nx28x28) np array to (Nx784) torch tensor """
    images = images.reshape(images.shape[0], images.shape[1]*images.shape[2], images.shape[3])
    return from_numpy(images[:, :, 0])


def images_to_vectors_numpy_multiclass(images: np.array) -> Tensor:
    """ converts (Nx28x28) numpy array to (Nx784) tensor in multiclass setting"""
    images = images.reshape(images.shape[0], images.shape[2]*images.shape[3], images.shape[1])
    return from_numpy(images[:, :, 0])

def vectors_to_images(vectors,w,h):
    return vectors.view(vectors.size(0), 3, w,h)


def values_target(size: tuple, value: float, cuda: False) -> Variable:
    """ returns tensor filled with value of given size """
    result = Variable(full(size=size, fill_value=value))
    if cuda:
        result = result.cuda()
    return result


def weights_init(m):
    """ initialize convolutional and batch norm layers in generator and discriminator """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
