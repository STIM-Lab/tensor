import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean
import scipy as sp
from skimage.color import rgb2gray
import sys
import tensorvote as tv

def structure2d(I, sigma=3, deriv=1, noise=0):

    if(len(I.shape) > 2):
        img = I[:, :, 0]
    else:
        img = I

    # calculate the image gradient
    dIdy = np.gradient(img, axis=0, edge_order=2)
    #dIdy = np.gradient(dIdy, axis=0, edge_order=2)
    dIdx = np.gradient(img, axis=1, edge_order=2)
    #dIdx = np.gradient(dIdx, axis=1, edge_order=2)
    print("Max dIdy: " + str(np.max(dIdy)))
    
    

    # create the structure tensor
    T = np.zeros((img.shape[0], img.shape[1], 2, 2))
    
    T[:, :, 0, 0] = dIdx * dIdx
    if noise > 0:
        T[:, :, 0, 0] = T[:, :, 0, 0] + np.abs(np.random.normal(0.0, noise, T[:, :, 0, 0].shape))
        
    T[:, :, 1, 1] = dIdy * dIdy
    if noise > 0:
        T[:, :, 1, 1] = T[:, :, 1, 1] + np.abs(np.random.normal(0.0, noise, T[:, :, 1, 1].shape))
        
    T[:, :, 0, 1] = dIdx * dIdy
    if noise > 0:
        T[:, :, 0, 1] = T[:, :, 0, 1] + np.random.normal(0.0, noise, T[:, :, 0, 1].shape)
        
    T[:, :, 1, 0] = T[:, :, 0, 1]

    #if the sigma value is 0, don't do any blurring or resampling
    if sigma > 0:
        window = np.ones((sigma, sigma, 1, 1))
        T = sp.signal.convolve(T, window, mode="same") 
    
        

    return T

def structure2d_nz(I, w=3):
    
    if(len(I.shape) > 2):
        img = I[:, :, 0]
    else:
        img = I
        
    T = structure2d(I, w)
    
    nz = img == 0
    T[nz, :, :] = 0
    
    return T

# calculate the N-D structure tensor
def structen(I, sigma):
    
    # calculate the gradient of the image
    D = np.gradient(I)
    
    dims = len(D)
    
    # allocate the structure tensor
    T_shape = I.shape + (dims, dims)
    T = np.zeros(T_shape)
    
    for d1 in range(dims):
        for d2 in range(dims):
            T[..., d1, d2] = D[d1] * D[d2]
            
    if sigma is not None:
        
        if np.isscalar(sigma):
            s = tuple([sigma] * dims) + (0, 0)
        elif len(sigma) == 1:
            s = ((sigma[0]) * dims) + (0, 0)
        elif len(sigma) == dims:
            s = sigma + (0, 0)
        else:
            raise Exception("Invalid sigma (must be a single number or one number for each dimension)")
        
        return sp.ndimage.gaussian_filter(T, s)
        
    return T
    

def hessian(I, sigma=0):
    
    if(len(I.shape) > 2):
        img = I[:, :, 0]
    else:
        img = I
        
    Dy, Dx = np.gradient(img)
    
    D2xy, D2x2 = np.gradient(Dx)
    D2y2, D2yx = np.gradient(Dy)   

    
    # create the Hessian tensor
    T = np.zeros((img.shape[0], img.shape[1], 2, 2))
    T[:, :, 0, 0] = D2x2
    T[:, :, 1, 1] = D2y2
    T[:, :, 0, 1] = D2xy
    T[:, :, 1, 0] = T[:, :, 0, 1]
    
    if sigma != 0:
        T = sp.ndimage.gaussian_filter(T, (sigma, sigma, 0, 0))
    
    return T

# split a tensor field into tensors with negative and positive eigenvalues
def signsplit(T):
    
    L, V = tv.eigmag(T)
    
    positiveIdx = L > 0
    negativeIdx = L < 0
    
    Lpos = np.zeros_like(L)
    Lpos[positiveIdx] = L[positiveIdx]
    
    Lneg = np.zeros_like(L)
    Lneg[negativeIdx] = L[negativeIdx]
    
    LMpos = np.zeros_like(T)
    LMpos[..., 0, 0] = Lpos[..., 0]
    LMpos[..., 1, 1] = Lpos[..., 1]
    
    Vt = np.moveaxis(V, -1, -2)
    Tpos = V @ LMpos @Vt
    
    LMneg = np.zeros_like(T)
    LMneg[..., 0, 0] = Lneg[..., 0]
    LMneg[..., 1, 1] = Lneg[..., 1]
    Tneg = V @ LMneg @Vt
    
    return Tpos, Tneg
    
    
    
    
        