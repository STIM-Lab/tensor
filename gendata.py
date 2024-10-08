import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import image2tensor as st
import tensorvote as tv
import skimage as ski
import os

# adds Gaussian noise to a symmetric second order tensor field        
def addGaussianT(T, sigma1, sigma0 = None):
    
    if sigma0 is None:
        sigma0 = sigma1
        
    THETA = np.random.uniform(0, np.pi * 2, (T.shape[0], T.shape[1]))
    X = np.cos(THETA)
    Y = np.sin(THETA)
    
    lambda_0 = np.abs(np.random.normal(0.0, sigma0, (T.shape[0], T.shape[1])))
    lambda_1 = np.abs(np.random.normal(0.0, sigma1, (T.shape[0], T.shape[1])))
    
    ETA1 = np.zeros_like(T)
    ETA1[:, :, 0, 0] = X * X * lambda_1
    ETA1[:, :, 0, 1] = X * Y * lambda_1
    ETA1[:, :, 1, 0] = ETA1[:, :, 0, 1]
    ETA1[:, :, 1, 1] = Y * Y * lambda_1
    
    ETA0 = np.zeros_like(T)
    X0 = Y
    Y0 = -X
    ETA0[:, :, 0, 0] = X0 * X0 * lambda_0
    ETA0[:, :, 0, 1] = X0 * Y0 * lambda_0
    ETA0[:, :, 1, 0] = ETA0[:, :, 0, 1]
    ETA0[:, :, 1, 1] = Y0 * Y0 * lambda_0
    
    return T + ETA1 + ETA0

def addShiftT(T, sigmax, dropout=0.5):

        
    x = np.array(range(0, T.shape[0]))
    y = np.array(range(0, T.shape[1]))
    X, Y = np.meshgrid(x, y)
    etaX = np.random.normal(0.0, sigma, size=(T.shape[0], T.shape[1])).astype(int)
    etaY = np.random.normal(0.0, sigma, size=(T.shape[0], T.shape[1])).astype(int)
    delete = np.random.uniform(0.0, 1.0, size=(T.shape[0], T.shape[1]))
    
    iX = X + etaX
    iY = Y + etaY
    iX[iX >= T.shape[0]] = 0
    iY[iY >= T.shape[1]] = 0
    iX[iX <= 0] = 0
    iY[iY <= 0] = 0
    T_out = T[iX, iY, :, :]
    T_out[delete < dropout, :, :] = 0
    return T_out

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

# generates an NxN axis-aligned grid tensor field with b boxes
def genBoxGrid2T(N, b, linewidth, noise):
    
    I = genBoxGrid2(N, b, linewidth, 0).astype(np.float32) / 255
    
    T = st.structure2d(I)
    if noise != 0:
        T = addGaussianT(T, noise, 0)
        
    return T
    

def genCircleGrid2(N, circles, linewidth, noise=0):
    
    x = np.linspace(0, N, N)                    # create a meshgrid of coordinates
    X, Y = np.meshgrid(x, x)
    #width = 1/N * linewidth
    
    diameter = N / circles                    # calculate the radius and diameter of the circles
    radius = diameter / 2.0
    
    R = np.ones((N, N)) * 100000
    for cxi in range(circles):                  # for each circle along x and y
        for cyi in range(circles):
            cx = radius + diameter * cxi
            cy = radius + diameter * cyi
            DX = X - cx
            DY = Y - cy
            D = np.sqrt(DX**2 + DY**2)
            R = np.minimum(R, D)
            
    I = R
    I[R<radius] = 0.0
    I[R>=radius] = 1.0
    
    #I = np.logical_and(R >= (radius - width), R <= radius ).astype(np.float32)
    if noise != 0:
        I = np.random.normal(0, noise, I.shape) + I
    #I[I<0] = 0
            
    return np.floor(((I - np.min(I)) / (np.max(I) - np.min(I))) * 255)

def genCircleGrid2T(N, b, linewidth, noise):
    
    I = genCircleGrid2(N, b, linewidth, 0).astype(np.float32) / 255
    
    T = st.structure2d(I)
    if noise != 0:
        T = addGaussianT(T, noise, 0)
        
    return T

def genSpiral3(N, d):
    
    I = np.zeros((N, N, N)).astype(np.float32)
    
    dt = 0.00001
    t = 0
    
    while True:
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = 2 * t
        
        xp = (x / (2 * d) + 0.5) * N
        yp = (y / (2 * d) + 0.5) * N
        zp = (z / (2 * d)) * N
        
        xi = int(xp)
        yi = int(yp)
        zi = int(zp)
        if(xi < 0 or yi < 0 or zi < 0 or xi >= N or yi >= N or zi >= N):
            return I
        
        I[zi, yi, xi] = 1.0
        
        t = t + dt
        

def genSpiral2T(N, d, noise = 0):
    
    T = np.zeros((N, N, 2, 2)).astype(np.float32)
    
    dt = 0.001
    
    t = 0
    while True:
        x = t * np.cos(t)
        y = t * np.sin(t)
        
        
        # calculate the direction of the spiral (as the derivative)
        dx_dt = np.cos(t) - t * np.sin(t)
        dy_dt = t * np.cos(t) + np.sin(t)
        
        xp = (x / (2 * d) + 0.5) * N
        yp = (y / (2 * d) + 0.5) * N
        
        
        xi = int(xp)
        yi = int(yp)
        if(xi < 0 or yi < 0 or xi >= N or yi >= N):
            break
        
        
        
        l = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
        
        ty = -dy_dt / l
        tx = dx_dt / l
        
        T[yi, xi, 0, 0] = tx * tx
        T[yi, xi, 0, 1] = tx * ty
        T[yi, xi, 1, 0] = T[yi, xi, 0, 1]
        T[yi, xi, 1, 1] = ty * ty
        
        t = t + dt
        
    if noise != 0:
        T = addGaussianT(T, noise, 0)
    return T


       
# saves an image stack, assuming the color value is the fastest (last) dimension    
def saveColorStack(filename, I):
    # get the number of images to save
    nz = I.shape[0]
    digits = len(str(nz))
    
    # uint conversion
    I8 = (I * 255).astype(np.uint8)
    
    for zi in range(nz):
        filestring = filename + "%0" + str(digits) + "d.bmp"
        ski.io.imsave(filestring %zi, I8[zi, :, :, :])
    

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
    
# if __name__ == "__main__":    
#     N = 1000
#     boxes = 5
#     width = 1
#     max_noise = 1.0
#     n_noise = 5
#     noise = np.linspace(0.0, 1.0, n_noise)
    
#     if not os.path.exists("data"):
#         os.mkdir("data")
    
#     for ni in range(len(noise)):
    
#         I = genBoxGrid2(N, boxes, width, noise[ni])
#         ski.io.imsave("data/boxgrid2_" + str(noise[ni]*10) + ".png", I.astype(np.uint8))
        
#         I = genCircleGrid2(N, boxes, width, noise[ni])
#         ski.io.imsave("data/circlegrid2_"+ str(noise[ni]*10) + ".png", I.astype(np.uint8))
        
#         T = genBoxGrid2T(N, boxes, width, noise[ni])
#         np.save("data/boxgrid2t_" + str(noise[ni]*10) + ".npy", T.astype(np.float32))
        
#         T = genCircleGrid2T(N, boxes, width, noise[ni])
#         np.save("data/circlegrid2t_" + str(noise[ni]*10) + ".npy", T.astype(np.float32))

N = 2000
spiral_size = 200
sigma = 3
noise = 2
grid_size = 10
line_width = 1

T = genSpiral2T(N, spiral_size, 0)
T = addShiftT(T, 3, 0.9)
#T = addGaussianT(T, 0.2)
np.save("spiral.npy", T)
tv.visualize(T)

#V = tv.vote2(T)
#plt.figure()
#tv.visualize(V, glyphs=False)

I = genBoxGrid2(N, grid_size, line_width)
ski.io.imsave("boxgrid.bmp", I.astype(np.uint8))

I = genCircleGrid2(N, grid_size, line_width)
ski.io.imsave("circlegrid.bmp", I.astype(np.uint8))

I = ski.io.imread("physarum.bmp")[:, :, 0].astype(np.float32) / 255.0

eta = np.random.normal(0.0, noise, I.shape)
In = I + np.abs(eta)

I = (In - np.min(In)) / (np.max(In) - np.min(In))
ski.io.imsave("physarumn.bmp", (I * 255).astype(np.uint8))
#%%    
# N = 500
# d = 60
# I = genSpiral3(N, d)
# sigma = 2
# I = sp.ndimage.gaussian_filter(I, (sigma, sigma, sigma))
# I = I * 15
# I[I > 1] = 1

# # convert the single-channel image into a color image
# c = np.linspace(0, 1, N).astype(np.float32)
# cr = np.linspace(1, 0, N).astype(np.float32)
# R, G, B = np.meshgrid(cr, c, c)

# C = np.zeros((N, N, N, 3))
# C[:, :, :, 0] = I * R
# C[:, :, :, 1] = I * G
# C[:, :, :, 2] = I * B

# plt.imshow(C[60])
# saveColorStack("test", C)

# #T = genSpiral2T(N, 20)
# #B = T #sp.ndimage.gaussian_filter(T, (sigma, sigma, 0, 0))
# #tv.visualize(B)
# #V = tv.vote2(T)
# #tv.visualize(V)