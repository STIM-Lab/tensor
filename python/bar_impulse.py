import numpy as np
import image2tensor

N = 11

I = np.zeros((N, N, N))

I[:, int(N/2), int(N/2)] = 1

T = image2tensor.structure3d(I, 0)

np.save("bar_impulse.npy", T)