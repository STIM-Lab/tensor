import numpy as np
import matplotlib.pyplot as plt
import structure2d as st
import tensorvote as tv


# generates an NxN axis-aligned grid with b boxes
# thickness is the width of each grid line
def axis_grid(N, b, linewidth):
    
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
        
    return I
    

N = 1000
boxes = 50
width = 2
noise = 0.5

testgrid = axis_grid(N, boxes, width)


noisy_testgrid = np.random.normal(0, noise, testgrid.shape) + testgrid
T = st.structure2d(noisy_testgrid, 2)

plt.figure()
plt.imshow(noisy_testgrid)
plt.title("Original Image")


plt.figure()
tv.visualize(T)
plt.title("Structure Tensor Field")

np.save("axis_grid.npy", T)

