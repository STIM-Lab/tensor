import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Generates a spiral image at the given resolution with the specified spacing
# N is the resolution of the image
# s is the number of rotations
# noise is the standard deviation of the gaussian noise added to the output
def genSpiral2(N, s, noise = 0):
    
    I = np.zeros((N, N)).astype(np.float32)
    
    
    dt = 0.001
    
    t = 0
    while True:
        x = t * np.cos(t)
        y = t * np.sin(t)
        
        
        # calculate the direction of the spiral (as the derivative)
        #dx_dt = np.cos(t) - t * np.sin(t)
        #dy_dt = t * np.cos(t) + np.sin(t)
        
        xp = (x / (11 * s) + 0.5) * N
        yp = (y / (11 * s) + 0.5) * N
        
        
        xi = int(xp)
        yi = int(yp)
        if(xi < 0 or yi < 0 or xi >= N or yi >= N):
            break
        
        
        
        #l = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
        
        #ty = -dy_dt / l
        #tx = dx_dt / l
        
        #T[yi, xi, 0, 0] = tx * tx
        #T[yi, xi, 0, 1] = tx * ty
        #T[yi, xi, 1, 0] = T[yi, xi, 0, 1]
        #T[yi, xi, 1, 1] = ty * ty
        
        I[yi, xi] = 1.0;
        
        t = t + dt
        
    if noise != 0:
        eta = np.random.normal(0.0, noise, (N, N))
        I = I + eta
        
    return np.abs(I)

I = genSpiral2(1000, 40, 0.3)
plt.imshow(I)
ski.io.imsave("spiral.png", (I * 255).astype(np.uint8))
