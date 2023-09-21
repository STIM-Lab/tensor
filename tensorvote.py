import numpy as np
import tqdm
import matplotlib.pyplot as plt

'''

arg1: input file
arg2: output file
arg3: sigma for image to tensor conversion
arg: tensor voting sigma

'''

def decay_wu(cos_theta, length, sigma, focus=1):

    c = np.exp(-(length**2) / (sigma**2))
    
    scale = 1.0 / (np.pi * (sigma**2) / 2)
    
    radial = (1 - cos_theta ** 2)
    
    D = scale * c * radial
    D[length == 0] = scale
    
    return D

# generate a saliency field for a stick tensor with direction e sampled for all
# points given by the distance L and orientation vector V
def saliency_wu(e, L, V0, V1, sigma, focus):
    
    # calculate the dot product between the eigenvector and the orientation
    eTv = e[0] * V0 + e[1] * V1
    
    # calculate the radius of the osculating circle
    R = np.divide(L, (2 * eTv), out=np.zeros_like(L), where=eTv!=0)
    
    d = decay_wu(eTv, L, sigma, focus)
    
    # calculate the target tensor orientation
    Ep0 = np.divide(d * (R * e[0] - L * V0), R, out=d * e[0], where=R!=0)
    Ep1 = np.divide(d * (R * e[1] - L * V1), R, out=d * e[1], where=R!=0) 
    
    
    
    S = np.zeros((V0.shape[0], V1.shape[1], 2, 2))
    S[:, :, 0, 0] = Ep0 ** 2
    S[:, :, 1, 1] = Ep1 ** 2
    S[:, :, 1, 0] = Ep0 * Ep1
    S[:, :, 0, 1] = S[:, :, 1, 0]
    
    return S
    

# calculate the vote result of the tensor field T
def vote_k_wu(T, k=0, sigma=3, focus=1):
    
    # perform the eigendecomposition of the field
    evals, evecs = np.linalg.eigh(T)
    
    # store the eigenvector corresponding to eigenvalue k
    E = evecs[:, :, :, k]
    
    # calculate the eccentricity
    #ecc = np.sqrt(1.0 - (evals[:, :, 0]**2 / evals[:, :, 1]**2))  # calculate the eccentricity
    #ecc[np.isnan(ecc)] = 0
    
    # calculate the optimal window size
    w = int(6 * sigma + 1)
    x = np.linspace(-w/2, w/2, w)
    X0, X1 = np.meshgrid(x, x)
    L = np.sqrt(X0**2 + X1**2)
    
    # calculate the normalized vector from (0, 0) to each point
    V0 = np.divide(X0, L, where=L!=0)
    V1 = np.divide(X1, L, where=L!=0)
    
    # create a padded vote field to store the vote results
    pad = int(3*sigma)
    VF = np.pad(np.zeros(T.shape), ((pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        #vfx0 = x0 + pad
        for x1 in range(T.shape[1]):
            #vfx1 = x1 + pad
            scale = evals[x0, x1, 1] # * ecc[x0, x1]
            S = scale * saliency_wu(E[x0, x1], L, V0, V1, sigma, focus)
            VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] = VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] + S
    return VF[pad:-pad, pad:-pad, :, :]
    
# generates a test voting field around a stick tensor pointed in the direction (x, y)
# (x, y) is the orientation of the stick tensor
# N is the resolution of the test field
# sigma is the decay falloff
def testfield_wu(x, y, N=100, sigma=10, focus=1):

    t = vector2tensor(x, y)

    T = np.zeros((N, N, 2, 2))
    T[int(N/2), int(N/2), :, :] = t

    VF = vote_k_wu(T, 0, sigma, focus)

    return VF

# calculate the decay function for a single tensor destination tensor
# angle is the angle between the source tensor orientation and the receiver
# length is the distance between the source tensor and receiver
# sigma is the falloff
def decay(angle, length, sigma, cutoff = np.pi/4):

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
    if alpha > cutoff or alpha < -cutoff:
        d = 0

    return d

# Calculate the vote contribution of a stick tensor pointed in direction theta
# theta is the orientation of the source tensor in polar coordinates
# u, v is the 2D coordinates of the destination tensor relative to the source (0, 0)
# sigma is the falloff
def saliency_theta(theta, u, v, sigma=10, focus=1):
    
    theta_cos, theta_sin = np.cos(theta), np.sin(theta)

    Rtheta_r = np.array(((theta_cos, theta_sin), (-theta_sin, theta_cos)))
    Rtheta_l = np.array(((theta_cos, -theta_sin), (theta_sin, theta_cos)))    
   
    p = np.dot(Rtheta_r, [u, v])
    
    l = np.sqrt(p[0] * p[0] + p[1] * p[1])                      # calculate the length term  
    
    # calculate the tensor at (u,v) when theta = 0
    phi = np.arctan2(p[1], p[0])
    
    # calculate the decay
    #d = decay(phi, l, sigma, cutoff)
    d = decay_wu(phi, l, sigma, focus)
    
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
def saliency(T, u, v, sigma, cutoff=2):
    
    # if the input tensor is zero, just return 0 (the contribution is obviously 0)
    if not T.any():
        return np.zeros([2, 2]), 0
    
    # calculate the eigendecomposition
    LAMBDA, K = np.linalg.eigh(T)
    kl = K[:, 1]                                    # large eigenvector
    
    theta = np.arctan2(kl[1], kl[0])
    
    VT, d = saliency_theta(theta, u, v, sigma, cutoff)      # get the reciever tensor and saliency
    
    # scale by eccentricity and largest eigenvector
    lambdal = LAMBDA[1]                             # large eigenvalue
    lambdas = LAMBDA[0]                             # small eigenvalue
    ecc = np.sqrt(1.0 - (lambdas**2 / lambdal**2))  # calculate the eccentricity
    
    return VT, d * ecc * lambdal
    
    

# calculates the contribution of T across multiple coordinates U, V specified relative to the position of T
# T is a 2x2 symmetric tensor
# U, V provide spatial coordinates relative to the input tensor (input tensor is at u = 0, v = 0)
# sigma is the distance falloff of the tenso voting field
def votefield(T, U, V, sigma, focus=1):
    
    VF = np.zeros([U.shape[0], U.shape[1], 2, 2])
    
    for vi in range(U.shape[0]):
        for ui in range(U.shape[1]):
            u = U[ui, vi]
            v = V[ui, vi]
            VT, d = saliency(T, u, v, sigma, focus)
            VTval, VTvec = np.linalg.eigh(VT)
            VF[ui, vi, :, :] = VT * d    
    
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
def vote(T, sigma, focus=1, w=None):
    
    if w is None:
        w = int(6 * sigma / 2)
    
    X = T.shape[0]
    Y = T.shape[1]
    
    #sum the last two channels
    T_sum1 = np.sum(T**2, 3)
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
                            vt, d = saliency(T[x, y], u, v, sigma, focus)

                            # add the vote to the output field
                            VT[x + v, y + u] = VT[x + v, y + u] + vt * d
                pbar.update(1)
            
    return VT

# visualize a tensor field T (NxMx2x2)
def visualize(T, mode=None):
    Eval, Evec = np.linalg.eigh(T)
    plt.quiver(Evec[:, :, 0, 1], Evec[:, :, 1, 1], pivot="middle", headwidth=0, headlength=0, headaxislength=0, width=0.001)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    if mode is None or mode == "eval":
        plt.imshow(Eval[:, :, 1], origin="lower")
    if mode == "eccentricity":
        e0_2 = Eval[:, :, 0] ** 2
        e1_2 = Eval[:, :, 1] ** 2
        ratio = np.divide(e0_2, e1_2, out=np.ones_like(e1_2), where=e1_2!=0)
        ecc = np.sqrt(1 - ratio)
        plt.imshow(ecc, origin="lower")
    plt.colorbar()