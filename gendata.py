import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import image2tensor as st
import tensorvote as tv
import tensorvote3d as tv3
import skimage as ski
import os

# adds Gaussian noise to a symmetric second order tensor field        
def addGaussian2T(T, sigma1, sigma0 = None):
    
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


def addShiftT(T, sigma, dropout=0.5):

        
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
        T = addGaussian2T(T, noise, 0)
        
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
        T = addGaussian2T(T, noise, 0)
        
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
        T = addGaussian2T(T, noise, 0)
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
    
def genSample3D(shape, noise=0):
    vol = np.zeros((shape));
    
    # for solid sphere
    center_solid = (shape[0] // 4, shape[1] // 4, shape[2] // 4)
    radius_solid = shape[0] // 10

    # for hollow sphere
    center_hollow = (shape[0] - shape[0] // 4, shape[1] - shape[1] // 4, shape[2] - shape[2] // 4)
    radius_hollow = shape[0] // 10
    thick_hollow = 2
    
    # for donut (torus shape)
    center_donut = (shape[0] // 4, shape[1] // 2, shape[2] - shape[2] // 4)
    radius_donut = shape[0] // 8
    thick_donut = 3
    
    # for curved tube (partial donut)
    center_curve = (shape[0] - shape[0] // 3, shape[1] // 8, -shape[2] // 6)
    radius_curve = shape[0] // 1.5
    thick_curve = 2
    
    # for straight tube
    radius_tube = 2
    
    t_values = np.linspace(0, 1, 100)  # Parameter range for smooth curve
    curve_x = (t_values) * (shape[0] // 2) + shape[0] // 2          # Starts at max x, curves to x=0
    curve_y = t_values * (shape[1] // 2) + shape[1] // 2            # Moves from y=0 to max y
    curve_z = t_values * (shape[2] // 2)                            # Moves from z=0 to max z
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                
                # solid sphere
                if np.sqrt((x - center_solid[0])**2 + (y - center_solid[1])**2 + (z - center_solid[2])**2) <= radius_solid:
                    vol[x, y, z] = 1
                
                # hollow sphere
                dist_hollow = np.sqrt((x - center_hollow[0])**2 + (y - center_hollow[1])**2 + (z - center_hollow[2])**2)
                if radius_hollow - thick_hollow <= dist_hollow <= radius_hollow:
                    vol[x, y, z] = 1
                    
                # donut
                distance_xy = np.sqrt((y - center_donut[1])**2 + (x - center_donut[0])**2)
                if (radius_donut - thick_donut) <= distance_xy <= (radius_donut + thick_donut):
                    # check the vertical distance from the center of the tube
                    if np.abs(z - center_donut[2]) <= thick_donut:
                        # check if the distance from the tube's center is correct
                        distance_donut = np.sqrt((distance_xy - radius_donut)**2 + (z - center_donut[2])**2)
                        if distance_donut <= thick_donut:
                            vol[x, y, z] = 1
                            
                # curved tube
                distance_yz = np.sqrt((y - center_curve[1])**2 + (z - center_curve[2])**2)
                if (radius_curve - thick_curve) <= distance_yz <= (radius_curve + thick_curve):
                    if np.abs(x - center_curve[0]) <= thick_curve:
                        distance_curve = np.sqrt((distance_yz - radius_curve)**2 + (x - center_curve[0])**2)
                        if distance_curve <= thick_curve:
                            vol[x, y, z] = 1
                
                # straight tube
                dist_tube = np.sqrt((x - curve_x)**2 + (y - curve_y)**2 + (z - curve_z)**2)
                if np.min(dist_tube) <= radius_tube:
                    vol[x, y, z] = 1
    #vol *= 255
    if noise != 0:
        vol = np.random.normal(0, noise, vol.shape) + vol
    vol[vol<0] = 0
    #vol = np.floor(((vol - np.min(vol)) / (np.max(vol) - np.min(vol))) * 255).astype(np.uint8)
    return vol

# generates an empty field with a plate "impulse" tensor at the center
def gen_plate_impulse(N, impulsefile, votefile, numericalfile, normalize=True):
    # generate an empty tensor field
    T = np.zeros(N + (3, 3))

    # generate a plate tensor
    P = np.zeros((3,3))
    P[0, 0] = 1
    P[1, 1] = 1

    # set the center pixel to a plate tensor
    C = tuple(int(n/2) for n in N)
    T[C] = P

    # save the plate tensor "impulse"
    np.save(impulsefile, T.astype(np.float32))

    # perform tensor voting
    T_vote = tv3.platevote3(T, 3, 0, normalize)
    # save the result of tensor voting
    np.save(votefile, T_vote.astype(np.float32))
    
    T_numerical = tv3.platevote3_numerical(T, 3, 0, 1, 3, normalize)
    np.save(numericalfile, T_numerical.astype(np.float32))
    
    return T, T_vote, T_numerical


# generate an impulse plate field
print("Generating plate impulse response:")
T, Ta, Tn = gen_plate_impulse((1, 1, 1), "plate_field.npy", "plate_python.npy", "plate_python_numerical.npy", normalize=False)

#%%
#plt.subplot(1, 2, 1)
#plt.imshow(Tv[:, :, 5, 0, 0])
#plt.subplot(1, 2, 2)
#plt.imshow(Tn[:, :, 5, 0, 0])

#size = 100
#vol = genSample3D((size, size, size), 0.6)
#st = st.structure3d(vol.astype(np.float32), 3)
#vol = st.addGaussian3T(vol, 2)
#np.save('synthetic.npy', st)

#np.save("impulse.npy", tv3.impulse3(size, 1, 0, 0))


