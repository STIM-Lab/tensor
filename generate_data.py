import numpy as np
import matplotlib.pyplot as plt
import structen as st
import tensorvote as tv
import skimage as ski
import os


# generates an NxN axis-aligned grid with b boxes
# thickness is the width of each grid line
def genBoxGrid2(N, b, linewidth, noise=0):
    
    if b < 2:
        raise ValueError("An axis grid requries at least two boxes")
        
    # calculate the number of grid lines
    nlines = b - 1
    
    # calculate the total number of pixels taken up by grid lines
    linepixels = int(nlines * linewidth)
    
    # calculate the size of each box
    boxwidth = int((N - linepixels) / b)
    
    I = np.zeros((N, N))
    for i in range(nlines):
        startpixel = boxwidth + (linewidth + boxwidth) * i
        endpixel = startpixel + linewidth
        
        I[:, startpixel:endpixel] = 1
        I[startpixel:endpixel, :] = 1
        
    if noise != 0:
        I = np.random.normal(0, noise, I.shape) + I
    I[I<0] = 0
        
    return np.floor(((I - np.min(I)) / (np.max(I) - np.min(I))) * 255)

def genBoxGrid2T(N, b, linewidth, noise):
    
    I = genBoxGrid2(N, b, linewidth, 0).astype(np.float32) / 255
    
    T = st.structure2d(I)
    if noise != 0:
        T = np.random.normal(0, noise, T.shape) + T
        
    return T
    

def genCircleGrid2(N, circles, linewidth, noise=0):
    
    x = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, x)
    width = 1/N * linewidth
    
    diameter = 1.0 / circles
    radius = diameter / 2.0
    
    R = np.ones((N, N))
    for cxi in range(circles):
        for cyi in range(circles):
            cx = radius + diameter * cxi
            cy = radius + diameter * cyi
            DX = X - cx
            DY = Y - cy
            D = np.sqrt(DX**2 + DY**2)
            R = np.minimum(R, D)
    
    I = np.logical_and(R >= (radius - width), R <= radius ).astype(np.float32)
    if noise != 0:
        I = np.random.normal(0, noise, I.shape) + I
    I[I<0] = 0
            
    return np.floor(((I - np.min(I)) / (np.max(I) - np.min(I))) * 255)

def genCircleGrid2T(N, b, linewidth, noise):
    
    I = genCircleGrid2(N, b, linewidth, 0).astype(np.float32) / 255
    
    T = st.structure2d(I)
    if noise != 0:
        T = np.random.normal(0, noise, T.shape) + T
        
    return T           

# generates an NxNxN axis-aligned grid with b boxes along each dimension
# thickness is the width of each grid line
def axis_grid_3d(N, b, linewidth, noise=0):
    
    # calculate the number of grid lines
    nlines = b - 1
    
    # calculate the total number of pixels taken up by grid lines
    linepixels = int(nlines * linewidth)
    
    # calculate the size of each box
    boxwidth = int((N - linepixels) / b)
    
    I = np.zeros((N, N, N))
    for i in range(nlines):
        startpixel = boxwidth + (linewidth + boxwidth) * i
        endpixel = startpixel + linewidth
        
        
        I[startpixel:endpixel, :, :] = 1
        I[:, startpixel:endpixel, :] = 1
        I[:, :, startpixel:endpixel] = 1
        
    if noise != 0:
        I = np.random.normal(0, noise, I.shape) + I        
        
    return (I - np.min(I)) / (np.max(I) - np.min(I))

# saves a stack of images
def savestack(filename, I):
    
    # get the number of images to save
    nz = I.shape[2]
    digits = len(str(nz))
    
    # uint conversion
    I8 = (I * 255).astype(np.uint8)
    
    for zi in range(nz):
        filestring = filename + "%0" + str(digits) + "d.bmp"
        ski.io.imsave(filestring %zi, I8[:, :, zi])
    
if __name__ == "__main__":    
    N = 1000
    boxes = 5
    width = 1
    max_noise = 1.0
    n_noise = 5
    noise = np.linspace(0.0, 1.0, n_noise)
    
    if not os.path.exists("data"):
        os.mkdir("data")
    
    for ni in range(len(noise)):
    
        I = genBoxGrid2(N, boxes, width, noise[ni])
        ski.io.imsave("data/boxgrid2_" + str(noise[ni]*10) + ".png", I.astype(np.uint8))
        
        I = genCircleGrid2(N, boxes, width, noise[ni])
        ski.io.imsave("data/circlegrid2_"+ str(noise[ni]*10) + ".png", I.astype(np.uint8))
        
        T = genBoxGrid2T(N, boxes, width, noise[ni])
        np.save("data/boxgrid2t_" + str(noise[ni]*10) + ".npy", T)
        
        T = genCircleGrid2T(N, boxes, width, noise[ni])
        np.save("data/circlegrid2t_" + str(noise[ni]*10) + ".npy", T)
    
    
    #T = genCircleGrid2T(100, boxes, width, 0.2)
    #tv.visualize(T)
    #T2 = st.structen(I, 1)
    
    #plt.figure()
    #plt.imshow(I)
    #plt.title("Original Image")
    
    
    #plt.figure()
    #tv.visualize(T2)
    #plt.title("Structure Tensor Field")
    
    #ski.io.imsave("grid2D.bmp", np.uint8(I * 255))
    #np.save("grid2D.npy", T2.astype(np.float32))
    
    #grid3D = axis_grid_3d(N, boxes, width, noise)
    #T3 = st.structen(grid3D, 1)
    #np.save("grid3D.npy", T3.astype(np.float32))
    #savestack("grid3D/grid3D", grid3D)

