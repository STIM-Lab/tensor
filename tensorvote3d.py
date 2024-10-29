import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

'''

arg1: input file
arg2: output file
arg3: sigma for image to tensor conversion
arg: tensor voting sigma

'''

def eigmag(T):
    
    eigenValues, eigenVectors = np.linalg.eigh(T)
    magValues = np.abs(eigenValues)
    
    idx = np.argsort(magValues, -1)
    
    sortedValues = np.take_along_axis(eigenValues, idx, -1)
    #sortedValues = eigenValues
    sortedVectors = np.zeros_like(eigenVectors)
    sortedVectors[..., 0, :] = np.take_along_axis(eigenVectors[..., 0, :], idx, -1)
    sortedVectors[..., 1, :] = np.take_along_axis(eigenVectors[..., 1, :], idx, -1)
    sortedVectors[..., 2, :] = np.take_along_axis(eigenVectors[..., 2, :], idx, -1)
    #eigenVectors = np.take_along_axis(eigenVectors, idx, -1)
    
    return sortedValues, sortedVectors

# calculates the normalization factor for a stick tensor field
def eta(sigma1, sigma2, p):
    num = np.pi * math.factorial(2*p)
    den = 2**(2*p) * (math.factorial(p)**2)
    s = sigma1**2 + sigma2**2
    integral = (num / den) * s
    return 1.0 / integral


# calculates the 3D stick field
def stickfield3(qx, qy, qz, RX, RY, RZ, sigma1, sigma2=0, power=1):
    
    q = np.zeros((3, 1))
    q[0] = qx
    q[1] = qy
    q[2] = qz
    
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2 + RZ**2)
    
    # calculate the normalized direction vector
    D = np.zeros((RX.shape[0], RX.shape[1], RX.shape[2], 3, 1))
       
    # if L == 0, assume that D is zero (and 1-qTd = 1)
    #D[:, :, 0, 0] = np.divide(RX, L, out=np.zeros_like(RX), where=L!=0)
    #D[:, :, 1, 0] = np.divide(RY, L, out=np.zeros_like(RY), where=L!=0)
    
    # if L == 0, assume that D is zero (and 1-qTd = 1)
    D[:, :, :, 0, 0] = np.divide(RX, L, out=np.ones_like(RX)*qx, where=L!=0)
    D[:, :, :, 1, 0] = np.divide(RY, L, out=np.ones_like(RY)*qy, where=L!=0)
    D[:, :, :, 2, 0] = np.divide(RZ, L, out=np.ones_like(RZ)*qz, where=L!=0)
    
    # calculate the rotated stick direction
    Dt = np.transpose(D, axes=(0, 1, 3, 2))
    I = np.eye(2)
    R = I - 2 * np.matmul(D, Dt)
    Rq = np.matmul(R, q)
    Rqt = np.transpose(Rq, (0, 1, 3, 2))
    
    # calculate the decay based on the desired properties
    if sigma1 == 0:
        d1 = 0
    else:
        d1 = np.exp(- L**2 / sigma1**2)
        
    if sigma2 == 0:
        d2 = 0
    else:
        d2 = np.exp(- L**2 / sigma2**2)
    
    qTd = np.squeeze(np.matmul(np.transpose(q), D))
    cos_2_theta = qTd**2
    sin_2_theta = 1 - cos_2_theta    
    
    DECAY = (d1 * sin_2_theta + d2 * cos_2_theta)[..., np.newaxis, np.newaxis]
    #decay = g1 * sin_2_theta + g2 * cos_2_theta
    
    V = eta(sigma1, sigma2, power) * DECAY * np.matmul(Rq, Rqt)
    return V

# calculate the vote result of the tensor field T
# k is the eigenvector used as the voting direction
# sigma is the standard deviation of the vote field
def stickvote3(T, sigma=3, sigma2=0):

    evals, evecs = eigmag(T)
    evals_mag = np.abs(evals)
    
    # store the eigenvector corresponding to the largest eigenvalue
    E = evecs[:, :, :, :, 2]
    
    sigmax = max(sigma, sigma2)
    
    # calculate the optimal window size
    w = int(6 * sigmax + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1, X2 = np.meshgrid(x, x)
    
    # create a padded vote field to store the vote results
    pad = int(3 * sigmax)
    VF = np.pad(np.zeros(T.shape), ((pad, pad, pad), (pad, pad, pad), (0, 0, 0), (0, 0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        for x1 in range(T.shape[1]):
            for x2 in range(T.shape[2]):
                scale = (evals_mag[x0, x1, x2, 2] - evals_mag[x0, x1, x2, 1]) * np.sign(evals[x0, x1, x2, 2])
                S = scale * stickfield3(E[x0, x1, x2, 0], E[x0, x1, x2, 1], E[x0, x1, x2, 2], X0, X1, X2, sigma, sigma2)
                VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1], x2:x2 + S.shape[2]] = VF[x0:x0 + S.shape[0], 
                                                                                    x1:x1 + S.shape[1], x2:x2 + S.shape[2]] + S
    return VF[pad:-pad, pad:-pad, pad:-pad, :, :]