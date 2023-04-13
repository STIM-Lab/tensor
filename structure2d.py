import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean
import scipy as sp
from skimage.color import rgb2gray
import sys

def structure2d(I, sigma):

    if(len(I.shape) > 2):
        img = rgb2gray(I)
    else:
        img = I

    # calculate the image gradient
    dIdy, dIdx = np.gradient(img)
    
    

    # create the structure tensor
    T = np.zeros((img.shape[0], img.shape[1], 2, 2))
    T[:, :, 0, 0] = dIdx * dIdx
    T[:, :, 1, 1] = dIdy * dIdy
    T[:, :, 0, 1] = dIdx * dIdy
    T[:, :, 1, 0] = T[:, :, 0, 1]

    #if the sigma value is 0, don't do any blurring or resampling
    if sigma == 0:
        return T
        
    # otherwise blur and resample the image
    else:    
        # blur the structure tensor
        T_blur = sp.ndimage.gaussian_filter(T, [sigma, sigma, 0, 0])
        
        # resample the structure tensor
        T_resampled = downscale_local_mean(T_blur, (sigma, sigma, 1, 1))

    return T_resampled

if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print("Provide an image file name for processing")
        exit()
    else:
        filename = sys.argv[1]
        print("processing file " + filename)
        I = imread(filename)
        T = structure2d(I, 0)
    
    if(len(sys.argv) > 2):
        # the user provided an output file name
        outfilename = sys.argv[2]
    else:
        outfilename = "out.npy"
        
    print("saving tensor field as " + outfilename)
    np.save(outfilename, T.astype(np.float32))
        