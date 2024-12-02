import numpy as np
import matplotlib.pyplot as plt
import skimage as ski

# Generates a spiral image at the given resolution with the specified spacing
# N is the resolution of the image
# lines is the number of grid lines
# noise is the standard deviation of the gaussian noise added to the output
def genGrid2(N, lines, noise = 0):
    
    I = np.zeros((N, N)).astype(np.float32)
    
    
    # calculate the spacing between grid lines
    d = int(N / (lines + 1))
    
    for li in range(lines):
        i = li * (d + 1)
        I[i, :] = 1
        I[:, i] = 1
        
    if noise != 0:
        eta = np.random.normal(0.0, noise, (N, N))
        I = I + eta
        
    return np.abs(I)

I = genGrid2(1000, 40, 0.3)
plt.imshow(I)
ski.io.imsave("grid.png", (I * 255).astype(np.uint8))
