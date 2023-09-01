import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean
import scipy as sp
from skimage.color import rgb2gray
import sys

def structure2d(I, w=3):

    if(len(I.shape) > 2):
        img = I[:, :, 0]
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
    if w == 0:
        return T
        
    # otherwise blur the image
    else:    
        window = np.ones((w, w, 1, 1))
        T_blur = sp.signal.convolve(T, window, mode="same")

    return T_blur

def structure2d_bin(I, w=3):
    

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
        