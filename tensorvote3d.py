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
    num1 = 2
    den1 = (2*p) + 1
    term1 = sigma1**2 * (num1/den1)
    
    num2 = -2**(2*p + 1) * math.factorial(p) ** 2
    den2 = math.factorial(2*p + 1)
    term2 = sigma2**2 * (num2/den2)
    
    integral =  np.pi * (term1 + term2)
    return 1.0 / integral


def decay_integrate(sigma1, sigma2=0, power=1, N=100, L_max=10):
    total = 0
    
    dtheta = 2 * np.pi / N
    dphi = np.pi / N
    dl = L_max / N
    
    for n in range(N):
        theta = n * dtheta
        for m in range(N):
            phi = m * dphi
            
            x = np.cos(theta) * np.sin(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(phi)
            
            for k in range(N):
                L = k * dl
                dV = L**2 * dl * dtheta * dphi * np.sin(phi)

                d1 = np.exp(-L**2 / sigma1**2) if sigma1 != 0 else 0
                d2 = np.exp(-L**2 / sigma2**2) if sigma2 != 0 else 0
                
                qTd = x**2 + y**2 + z**2                # not sure about this line???
                cos_2p_theta = (qTd**2)**power
                sin_2p_theta = (1 - qTd**2)**power
                
                total += dV * (d1 * sin_2p_theta + d2 * cos_2p_theta)
    
    return total
    
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
    Dt = np.transpose(D, axes=(0, 1, 2, 4, 3))
    I = np.eye(3)
    R = I - 2 * np.matmul(D, Dt)
    Rq = np.matmul(R, q)
    Rqt = np.transpose(Rq, (0, 1, 2, 4, 3))
    
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
    cos_2p_theta = (qTd**2)**power
    sin_2p_theta = (1 - qTd**2)**power    
    
    DECAY = (d1 * sin_2p_theta + d2 * cos_2p_theta)[..., np.newaxis, np.newaxis]
    
    #decay = g1 * sin_2_theta + g2 * cos_2_theta
    
    V = eta(sigma1, sigma2, power) * DECAY * np.matmul(Rq, Rqt)
    #V = DECAY * np.matmul(Rq, Rqt)
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
    X0, X1, X2 = np.meshgrid(x, x, x)
    
    # create a padded vote field to store the vote results
    pad = int(3 * sigmax)
    Z = np.zeros(T.shape)
    VF = np.pad(Z, ((pad, pad), (pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        for x1 in range(T.shape[1]):
            for x2 in range(T.shape[2]):
                scale = (evals_mag[x0, x1, x2, 2] - evals_mag[x0, x1, x2, 1]) * np.sign(evals[x0, x1, x2, 2])
                S = scale * stickfield3(E[x0, x1, x2, 0], E[x0, x1, x2, 1], E[x0, x1, x2, 2], X0, X1, X2, sigma, sigma2)
                VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1], x2:x2 + S.shape[2]] = VF[x0:x0 + S.shape[0], 
                                                                                    x1:x1 + S.shape[1], x2:x2 + S.shape[2]] + S
    return VF[pad:-pad, pad:-pad, pad:-pad, :, :]

# generate an impulse tensor field to test tensor voting
def impulse3(N, x, y, z, l2=1, l1=0, l0=0, sigma1=5, sigma2=0, power=1):
    
    r = np.linspace(-N/2, N/2, N)
    X, Y, Z = np.meshgrid(r, r, r)
    
    l = np.sqrt(x**2 + y**2 + z**2)
    
    V = stickfield3(x/l, y/l, z/l, X, Y, Z, sigma1, sigma2, power)
    
    # m = np.zeros((3, 3))
    # m[0, 0] = x * x
    # m[0, 1] = x * y
    # m[0, 2] = x * z
    # m[1, 0] = x * y
    # m[1, 1] = y * y
    # m[1, 2] = y * z
    # m[2, 0] = x * z
    # m[2, 1] = z * y
    # m[2, 2] = z * z
    
    # l, v = np.linalg.eigh(m)
    
    # l[0] = l0
    # l[1] = l1
    # l[2] = l2
    
    # m = v @ np.diag(l) @ v.transpose()
    
    # T[int(N/2), int(N/2), int(N/2)] = m
    
    return V

#V = impulse3(101, 1, 0, 0, sigma1=20, sigma2=0, power=1)
#vals, vecs = np.linalg.eigh(V)
N = 101
r = np.linspace(-N/2, N/2, N)
X, Y, Z = np.meshgrid(r, r, r)

l = 1
decay = decay_integrate(10, 5, 2)
eta = eta(10, 5, 2)
#plt.imshow(vals[:, :, 50, 2])
#np.save("stickfield.npy", V.astype(np.float32))