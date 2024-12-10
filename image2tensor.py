import os
import sys
import cv2
import numpy as np
import scipy as sp
import tensorvote as tv
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean


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


def structure3d(T, sigma, noise=0):
    """
    This function takes a 3D grayscale volume and returns the structure tensor
    for visualization in tensorview3D.

    Parameters:
    T (uint8): the input grayscale volume (ZxYxX)

    Returns:
    float32: the structure tensor (ZxYxXx3x3)
    """
    # create the structure tensor
    dVdz, dVdy, dVdx = np.gradient(T, edge_order=2)
    T = np.zeros((dVdx.shape[0], dVdx.shape[1], dVdx.shape[2], 3, 3))
    T[:, :, :, 0, 0] = dVdx * dVdx
    if noise > 0:
        T[:, :, :, 0, 0] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 0, 0].shape))
    T[:, :, :, 1, 1] = dVdy * dVdy
    if noise > 0:
        T[:, :, :, 1, 1] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 1, 1].shape))
    T[:, :, :, 2, 2] = dVdz * dVdz
    if noise > 0:
        T[:, :, :, 2, 2] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 2, 2].shape))
    T[:, :, :, 0, 1] = dVdx * dVdy
    if noise > 0:
        T[:, :, :, 0, 1] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 0, 1].shape))
    T[:, :, :, 1, 0] = T[:, :, :, 0, 1]
    T[:, :, :, 0, 2] = dVdx * dVdz
    if noise > 0:
        T[:, :, :, 0, 2] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 0, 2].shape))
    T[:, :, :, 2, 0] = T[:, :, :, 0, 2]
    T[:, :, :, 1, 2] = dVdy * dVdz
    if noise > 0:
        T[:, :, :, 1, 2] += np.abs(np.random.normal(0.0, noise, T[:, :, :, 1, 2].shape))
    T[:, :, :, 2, 1] = T[:, :, :, 1, 2]

    # if sigma is zero, no blurring
    if sigma > 0:
        kernel = np.ones((sigma, sigma, sigma, 1, 1))
        T = sp.signal.convolve(T, kernel, mode="same") 
        #T = sp.ndimage.gaussian_filter(T, [sigma, sigma, sigma, 0, 0])         # my version
    
    return T.astype(np.float32)


def structure2d_nz(I, w=3):
    
    if(len(I.shape) > 2):
        img = I[:, :, 0]
    else:
        img = I
        
    T = structure2d(I, w)
    
    nz = img == 0
    T[nz, :, :] = 0
    
    return T


def structure3d_nz(V, sigma=3):
    
    T = structure3d(V, sigma)
    nz = V == 0
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
    
    
def images2volume(folder_path, grayscale=True):
    """
    The function loads all the images in the folder path and stack
    them into a 3D volume

    Parameters:
    folder_path (str): directory to the images

    Returns:
    uint8: the 3D volume
    """
    
    images_list = [f for f in os.listdir(folder_path) if f.endswith('.bmp') or
                   f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.png')]
        
    images = []
    for name in images_list:
        img = cv2.imread(os.path.join(folder_path, name), (cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
        images.append(img)

    return np.stack(images, axis=0)


# adds Gaussian noise to a symmetric second-order tensor field in 3D
def addGaussian3T(T, sigma1, sigma0=None):
    if sigma0 is None:
        sigma0 = sigma1

    # Generate random orientations in 3D
    PHI = np.random.uniform(0, 2 * np.pi, T.shape[:3])      # Azimuthal angle
    THETA = np.random.uniform(0, np.pi, T.shape[:3])        # Polar angle

    # Convert spherical to Cartesian coordinates
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)

    # Generate random eigenvalues for principal axes
    lambda_1 = np.abs(np.random.normal(0.0, sigma1, T.shape[:3]))
    lambda_0 = np.abs(np.random.normal(0.0, sigma0, T.shape[:3]))

    # Create noise tensors for lambda_1
    ETA1 = np.zeros_like(T)
    ETA1[..., 0, 0] = X * X * lambda_1
    ETA1[..., 0, 1] = X * Y * lambda_1
    ETA1[..., 0, 2] = X * Z * lambda_1
    ETA1[..., 1, 0] = ETA1[..., 0, 1]
    ETA1[..., 1, 1] = Y * Y * lambda_1
    ETA1[..., 1, 2] = Y * Z * lambda_1
    ETA1[..., 2, 0] = ETA1[..., 0, 2]
    ETA1[..., 2, 1] = ETA1[..., 1, 2]
    ETA1[..., 2, 2] = Z * Z * lambda_1

    # Create orthogonal vectors for lambda_0 noise
    X0, Y0, Z0 = -Y, X, np.zeros_like(Z)  # Perpendicular in the XY plane
    valid_mask = (X != 0) | (Y != 0)
    Z0[~valid_mask] = 1  # Avoid zero vectors; assign arbitrary orthogonal vectors
    mag = np.sqrt(X0**2 + Y0**2 + Z0**2)
    X0, Y0, Z0 = X0 / mag, Y0 / mag, Z0 / mag

    ETA0 = np.zeros_like(T)
    ETA0[..., 0, 0] = X0 * X0 * lambda_0
    ETA0[..., 0, 1] = X0 * Y0 * lambda_0
    ETA0[..., 0, 2] = X0 * Z0 * lambda_0
    ETA0[..., 1, 0] = ETA0[..., 0, 1]
    ETA0[..., 1, 1] = Y0 * Y0 * lambda_0
    ETA0[..., 1, 2] = Y0 * Z0 * lambda_0
    ETA0[..., 2, 0] = ETA0[..., 0, 2]
    ETA0[..., 2, 1] = ETA0[..., 1, 2]
    ETA0[..., 2, 2] = Z0 * Z0 * lambda_0

    return T + ETA1 + ETA0