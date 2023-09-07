import numpy as np
import tqdm
import matplotlib.pyplot as plt

'''

arg1: input file
arg2: output file
arg3: sigma for image to tensor conversion
arg: tensor voting sigma

'''

# calculate the decay function for a single tensor destination tensor
# angle is the angle between the source tensor orientation and the receiver
# length is the distance between the source tensor and receiver
# sigma is the falloff
def decay(angle, length, sigma):

    alpha = np.arccos(np.abs(np.cos(np.pi/2 - angle)))
    
    # calculate c (see math)
    c = (-16 * np.log10(0.1) * (sigma - 1)) / (np.pi ** 2)
    
    # calculate the saliency decay   
    if alpha == 0:
        S = length
    else:
        S = (alpha * length) / np.sin(alpha)            # arc length
    
    if length == 0:
        return 1
    else:
        kappa = (2 * np.sin(alpha)) / length
    
    S_kappa = S ** 2 + c * kappa ** 2
    E = -1.0 * S_kappa / (sigma ** 2)
    d = np.exp(E)
    if alpha > np.pi / 4 or alpha < -np.pi / 4:
        d = 0

    return d

# Calculate the vote contribution of a stick tensor pointed in direction theta
# theta is the orientation of the source tensor in polar coordinates
# u, v is the 2D coordinates of the destination tensor relative to the source (0, 0)
# sigma is the falloff
def saliency_theta(theta, u, v, sigma=10):
    
    theta_cos, theta_sin = np.cos(theta), np.sin(theta)

    Rtheta_r = np.array(((theta_cos, theta_sin), (-theta_sin, theta_cos)))
    Rtheta_l = np.array(((theta_cos, -theta_sin), (theta_sin, theta_cos)))    
   
    p = np.dot(Rtheta_r, [u, v])
    
    l = np.sqrt(p[0] * p[0] + p[1] * p[1])                      # calculate the length term  
    
    # calculate the tensor at (u,v) when theta = 0
    phi = np.arctan2(p[1], p[0])
    
    # calculate the decay
    d = decay(phi, l, sigma)
    
    phi2 = 2 * phi
    # calculate a rotation matrix to apply to the line tensor
    phi2_cos, phi2_sin = np.cos(phi2), np.sin(phi2)
    Rphi2 = np.array(((phi2_cos, -phi2_sin), (phi2_sin, phi2_cos)))
    
    V_source = np.dot(Rphi2, [1, 0])
    
    V = np.dot(Rtheta_l, V_source)

    return np.outer(V, V), d
    
# This function calculates the contribution of the vote from an input tensor T
# T is a 2x2 symmetric tensor
# u, v is a spatial coordinate relative to the input tensor (input tensor is at u = 0, v = 0)
# sigma is the the distance falloff of the tensor voting signal
def saliency(T, u, v, sigma):
    
    # if the input tensor is zero, just return 0 (the contribution is obviously 0)
    if not T.any():
        return np.zeros([2, 2]), 0
    
    # calculate the eigendecomposition
    LAMBDA, K = np.linalg.eigh(T)
    kl = K[:, 1]                                    # large eigenvector
    
    theta = np.arctan2(kl[1], kl[0])
    
    VT, d = saliency_theta(theta, u, v, sigma)      # get the reciever tensor and saliency
    
    # scale by eccentricity and largest eigenvector
    lambdal = LAMBDA[1]                             # large eigenvalue
    lambdas = LAMBDA[0]                             # small eigenvalue
    ecc = np.sqrt(1.0 - (lambdas**2 / lambdal**2))  # calculate the eccentricity
    
    return VT, d * ecc * lambdal
    
    

# calculates the contribution of T across multiple coordinates U, V specified relative to the position of T
# T is a 2x2 symmetric tensor
# U, V provide spatial coordinates relative to the input tensor (input tensor is at u = 0, v = 0)
# sigma is the distance falloff of the tenso voting field
def votefield(T, U, V, sigma):
    
    VF = np.zeros([U.shape[0], U.shape[1], 2, 2])
    
    for vi in range(U.shape[0]):
        for ui in range(U.shape[1]):
            u = U[ui, vi]
            v = V[ui, vi]
            VT, d = saliency(T, u, v, sigma)
            VTval, VTvec = np.linalg.eigh(VT)
            VF[ui, vi, :, :] = VT * d    
    
    return VF

# generates a test voting field around a stick tensor pointed in the direction (x, y)
# (x, y) is the orientation of the stick tensor
# N is the resolution of the test field
# sigma is the decay falloff
def testfield(x, y, N=100, sigma=10):

    # size of the tensor field in units
    S = (N - 1)/2

    t = vector2tensor(x, y)

    u = np.linspace(-S, S, N)
    v = np.linspace(-S, S, N)
    U, V = np.meshgrid(v, u)

    VF = votefield(t, U, V, sigma)

    return VF

def vector2tensor(x, y):

    vector = np.array([x, y])
    vector = vector / np.linalg.norm(vector)


    tensor = np.zeros([2, 2])
    tensor[0,0] = vector[0] * vector[0]
    tensor[0,1] = vector[0] * vector[1]
    tensor[1,0] = vector[1] * vector[0]
    tensor[1,1] = vector[1] * vector[1]

    return tensor

# apply tensor voting to the tensor field T and return the result
# T is an NxMx2x2 tensor field
def vote(T, sigma, w=None):
    
    if w is None:
        w = int(6 * sigma / 2)
    
    X = T.shape[0]
    Y = T.shape[1]
    
    #sum the last two channels
    T_sum1 = np.sum(T, 3)
    T_sum2 = np.sum(T_sum1, 2)
    binary_mask = T_sum2 != 0
    nonzero_pixels = np.count_nonzero(binary_mask)
    
    # create an output tensor field that is the same size as the input
    VT = np.zeros(T.shape)
    
    pbar = tqdm.tqdm(total=nonzero_pixels)
    # for each (x, y) pixel in the input image
    for x in range(X):
        for y in range(Y):
            # if this pixel contains a non-zero tensor
            if(binary_mask[x, y]):
                # for each (u, v) pixel surrounding this tensor within the window
                for u in range(-w, w):
                    for v in range(-w, w):
                        # if the window pixel is inside the original image
                        if y + u >= 0 and y + u < T.shape[1] and x + v >= 0 and x + v < T.shape[0]:
                            # calculate the vote tensor and saliency
                            vt, d = saliency(T[x, y], u, v, sigma)

                            # add the vote to the output field
                            VT[x + v, y + u] = VT[x + v, y + u] + vt * d
                pbar.update(1)
            
    return VT

# visualize a tensor field T (NxMx2x2)
def visualize(T):
    Eval, Evec = np.linalg.eigh(T)
    plt.quiver(Evec[:, :, 0, 1], Evec[:, :, 1, 1], pivot="middle", headwidth=0, headlength=0, headaxislength=0)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.imshow(Eval[:, :, 1], origin="lower")