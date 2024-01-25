import numpy as np
import matplotlib.pyplot as plt
import structen as st
import tensorvote as tv
import skimage as ski


# generates an NxN axis-aligned grid with b boxes
# thickness is the width of each grid line
def axis_grid_2d(N, b, linewidth, noise=0):
    
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
        
    return (I - np.min(I)) / (np.max(I) - np.min(I))

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
    
    
N = 200
boxes = 5
width = 2
noise = 0.1

grid2D = axis_grid_2d(N, boxes, width, noise)


T2 = st.structen(grid2D, 1)

plt.figure()
plt.imshow(grid2D)
plt.title("Original Image")


plt.figure()
tv.visualize(T2)
plt.title("Structure Tensor Field")

ski.io.imsave("grid2D.bmp", np.uint8(grid2D * 255))
np.save("grid2D.npy", T2.astype(np.float32))

grid3D = axis_grid_3d(N, boxes, width, noise)
T3 = st.structen(grid3D, 1)
np.save("grid3D.npy", T3.astype(np.float32))
savestack("grid3D/grid3D", grid3D)

