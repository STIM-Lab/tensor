import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import argparse

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
    sortedVectors = np.zeros_like(eigenVectors)
    sortedVectors[..., 0, :] = np.take_along_axis(eigenVectors[..., 0, :], idx, -1)
    sortedVectors[..., 1, :] = np.take_along_axis(eigenVectors[..., 1, :], idx, -1)
    
    return sortedValues, sortedVectors

# calculates the normalization factor for a stick tensor field
def eta(sigma1, sigma2, p):
    num = np.pi * math.factorial(2*p)
    den = 2**(2*p) * (math.factorial(p)**2)
    s = sigma1**2 + sigma2**2
    integral = (num / den) * s
    return 1.0 / integral
    
# calculates the voting field for a stick tensor using refined tensor voting (Goharbavang, et al., 2025)
# qx, qy is the eigenvector direction
# RX, RY are the x and y coordinates for the field
# sigma1, sigma2 are the standard deviations for the refined tensor voting algorithm
# power is the refinement term
def stickfield2(qx, qy, RX, RY, sigma1, sigma2=0, power=1):
    
    q = np.zeros((2, 1))
    q[0] = qx
    q[1] = qy
    
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2)
    
    # calculate the normalized direction vector
    D = np.zeros((RX.shape[0], RX.shape[1], 2, 1))
       
    D[:, :, 0, 0] = np.divide(RX, L, out=np.ones_like(RX)*qx, where=L!=0)
    D[:, :, 1, 0] = np.divide(RY, L, out=np.ones_like(RY)*qy, where=L!=0)
    
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
    
    V = eta(sigma1, sigma2, power) * DECAY * np.matmul(Rq, Rqt)
    return V

# calculate the vote result of the tensor field T using refined tensor voting (Goharbavang, et al., 2025)
# sigma is the standard deviation of the vote field
def stickvote2(T, sigma=3, sigma2=0):
    
    evals, evecs = eigmag(T)
    evals_mag = np.abs(evals)
    
    # store the eigenvector corresponding to the largest eigenvalue
    E = evecs[:, :, :, 1]
    
    sigmax = max(sigma, sigma2)
    
    # calculate the optimal window size
    w = int(6 * sigmax + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1 = np.meshgrid(x, x)
    
    # create a padded vote field to store the vote results
    pad = int(3 * sigmax)
    VF = np.pad(np.zeros(T.shape), ((pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        for x1 in range(T.shape[1]):
            scale = (evals_mag[x0, x1, 1] - evals_mag[x0, x1, 0]) * np.sign(evals[x0, x1, 1])
            S = scale * stickfield2(E[x0, x1, 0], E[x0, x1, 1], X0, X1, sigma, sigma2)
            VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] = VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] + S
    return VF[pad:-pad, pad:-pad, :, :]

# calculates the voting field for a plate tensor using refined tensor voting (Goharbavang, et al., 2025)
# RX, RY are the x and y coordinates for the field
# sigma1, sigma2 are the standard deviations for the refined tensor voting algorithm
def platefield2(RX, RY, sigma1, sigma2=0):
    
    ALPHA = np.arctan2(RY, RX)
    TWO_ALPHA = 2 * ALPHA
    
    L = np.sqrt(RX**2 + RY**2)
    
    c = 1 / (np.pi * (sigma1**2 + sigma2**2))
    
    e1 = 0
    if(sigma1 > 0):
        e1 = np.exp(- L**2 / sigma1**2)
    e2 = 0
    if(sigma2 > 0):
        e2 = np.exp(- L**2 / sigma2**2)
    
    M = np.zeros((RX.shape[0], RX.shape[1], 2, 2))
    
    COS_2ALPHA = np.cos(TWO_ALPHA)
    SIN_2ALPHA = np.sin(TWO_ALPHA)
    
    M[:, :, 0, 0] = (0.25 * (COS_2ALPHA + 2))
    M[:, :, 0, 1] = (0.25 * SIN_2ALPHA)
    M[:, :, 1, 0] = (0.25 * SIN_2ALPHA)
    M[:, :, 1, 1] = (0.25 * (2 - COS_2ALPHA))
    
    T = np.zeros((RX.shape[0], RX.shape[1], 2, 2))    
    
    T[:, :, 0, 0] = c * (e1 * (1 - M[:, :, 0, 0]) + e2 * M[:, :, 0, 0])
    T[:, :, 0, 1] = c * (e1 * (0 - M[:, :, 0, 1]) + e2 * M[:, :, 0, 1])
    T[:, :, 1, 0] = c * (e1 * (0 - M[:, :, 1, 0]) + e2 * M[:, :, 1, 0])
    T[:, :, 1, 1] = c * (e1 * (1 - M[:, :, 1, 1]) + e2 * M[:, :, 1, 1])
    
    return T

def platefield2_numerical(RX, RY, sigma1, sigma2=0, N=10):
    
    T = np.zeros((RX.shape[0], RX.shape[1], 2, 2))
    
    dtheta = np.pi / N
    for n in range(N):
        x = np.cos(n * dtheta)
        y = np.sin(n * dtheta)
        T = T + (1.0 / N) * stickfield2(x, y, RX, RY, sigma1, sigma2)
        
    return T
        
# calculate the vote result of the tensor field T
# k is the eigenvector used as the voting direction
# sigma is the standard deviation of the vote field
def platevote2_numerical(T, sigma1=3, sigma2=0, N=10):
    
    # perform the eigendecomposition of the field
    evals, evecs = np.linalg.eigh(T)
    evals_mag = np.abs(evals)
    
    # calculate the optimal window size
    sigma = max(sigma1, sigma2)
    w = int(6 * sigma + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1 = np.meshgrid(x, x)
    
    # create a padded vote field to store the vote results
    pad = int(3*sigma)
    VF = np.pad(np.zeros(T.shape), ((pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        for x1 in range(T.shape[1]):
            scale = (evals_mag[x0, x1, 1] - evals_mag[x0, x1, 0]) * np.sign(evals[x0, x1, 1])
            S = scale * platefield2_numerical(X0, X1, sigma, sigma2)
            VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] = VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] + S
    return VF[pad:-pad, pad:-pad, :, :]

# calculate the vote result of the tensor field T
# k is the eigenvector used as the voting direction
# sigma is the standard deviation of the vote field
def platevote2(T, sigma=3, sigma2=0):
    
    # perform the eigendecomposition of the field
    evals, _ = np.linalg.eigh(T)
    
    # calculate the optimal window size
    w = int(6 * sigma + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1 = np.meshgrid(x, x)
    
    # create a padded vote field to store the vote results
    pad = int(3*sigma)
    VF = np.pad(np.zeros(T.shape), ((pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        #vfx0 = x0 + pad
        for x1 in range(T.shape[1]):
            scale = evals[x0, x1, 0]
            S = scale * platefield2(X0, X1, sigma, sigma2)
            VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] = VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] + S
    return VF[pad:-pad, pad:-pad, :, :]

'''
Main function for analytical 2D tensor voting. This function applies both stick
and plate voting, and allows for both alpha and beta vote orientations. You can
use beta voting by assigning sigma=0 and the desired standard deviation for sigma_beta.
'''

def vote2(T, sigma=3, sigma_beta=0):
    
    S = stickvote2(T, sigma, sigma_beta)
    P = platevote2(T, sigma, sigma_beta)
    
    V = S + P
    return V

def vector2tensor(x, y):

    tensor = np.zeros([2, 2])
    tensor[0,0] = x * x
    tensor[0,1] = x * y
    tensor[1,0] = y * x
    tensor[1,1] = y * y

    return tensor

#generate an NxN field with a stick tensor pointing in the (x,y) direction
def generate_stick_field(x=0, y=1, N=101):
    T = np.zeros((N, N, 2, 2))
    center = int(N/2)
    T[center, center] = vector2tensor(x, y)

    return T

def generate2(x=0, y=1, N=51, sigma1=20, sigma2=10):
    
    l = np.sqrt(x**2 + y**2)
    if l==0:
        print("ERROR, eigenvector has zero length")
        x=0
        y=1
    else:
        x = x / l
        y = y / l
        
    X = np.linspace(-N/2, N/2, N)
    RX, RY = np.meshgrid(X, X)
    
    T = stickfield2(x, y, RX, RY, sigma1, sigma2)
    
    return T

# generate an impulse tensor field to test tensor voting
def impulse(N, x, y, l1=1, l0=0):
    T = np.zeros((N, N, 2, 2))
    
    m = np.zeros((2, 2))
    m[0, 0] = x * x
    m[1, 0] = x * y
    m[0, 1] = y * x
    m[1, 1] = y * y
    
    l, v = np.linalg.eigh(m)
    
    l[0] = l0
    l[1] = l1
    
    m = v @ np.diag(l) @ v.transpose()
    
    T[int(N/2), int(N/2)] = m
    
    return T
    
def vec2theta(V):
    return np.arctan2(V[..., 1], V[..., 0])

def eccentricity(T):
    L, V = eigmag(T)
    
    L0_2 = L[..., 0]**2
    L1_2 = L[..., 1]**2
    
    ratio = np.where(L1_2 > 0, L0_2 / L1_2, 1)
    
    E = np.sqrt(1 - ratio)
    return E

def eccentricity_decay(T, rate):
    
    ecc = eccentricity(T)
    return T * np.power(ecc[..., np.newaxis, np.newaxis], rate)

# visualize a tensor field T (NxMx2x2)
def visualize(T, glyphs=False, glyphscalar="l1"):
    
    L, V = eigmag(T)
    THETA0 = vec2theta(V[..., :, 0])                 # convert the eigenvector to polar coordinates
    
    neg = THETA0 < 0                                 # normalize to [0, 1] for color mapping
    THETA0[neg] = np.pi - np.abs(THETA0[neg])
    THETA0 = THETA0 / np.pi
    
    cmap = matplotlib.colormaps["hsv"]              # get the angle color
    C0 = cmap(THETA0)
    
    THETA1 = vec2theta(V[..., :, 1])                 # convert the eigenvector to polar coordinates
    
    neg = THETA1 < 0                                 # normalize to [0, 1] for color mapping
    THETA1[neg] = np.pi - np.abs(THETA1[neg])
    THETA1 = THETA1 / np.pi
    
    cmap = matplotlib.colormaps["hsv"]              # get the angle color
    C1 = cmap(THETA1)
    
    ecc = eccentricity(T)[..., np.newaxis]
    
    C0 = ecc * C0 + (1 - ecc)                         # scale the saturation by the eccentricity (lower eccentricity is whiter)
    l1_max = np.max(np.abs(L[..., 1]))
    C0 = C0 * (np.abs(L[..., 1]) / l1_max)[..., np.newaxis]
    
    C1 = ecc * C1 + (1 - ecc)                         # scale the saturation by the eccentricity (lower eccentricity is whiter)
    l1_max = np.max(np.abs(L[..., 1]))
    C1 = C1 * (np.abs(L[..., 1]) / l1_max)[..., np.newaxis]
    
    plt.subplot(2, 3, 1)
    plt.imshow(C0[:, :, 0:3], origin="lower")
    plt.title("Vector 0 Angle")
    plt.colorbar()
    
    plt.subplot(2, 3, 2)
    plt.imshow(C1[:, :, 0:3], origin="lower")
    plt.title("Vector 1 Angle")
    plt.colorbar()
    
    plt.subplot(2, 3, 3)
    plt.imshow(ecc, cmap="magma", origin="lower")
    plt.title("Eccentricity")
    plt.colorbar()
    
    plt.subplot(2, 3, 4)
    l0_max = np.max(np.abs(L[..., 0]))
    plt.imshow(L[:, :, 0], vmin=-l0_max, vmax=l0_max, cmap="RdYlBu_r", origin="lower")
    plt.colorbar()
    l0_sum = np.sum(L[..., 0])
    plt.title("Eigenvalue 0, integral = " + str(l0_sum))
    
    plt.subplot(2, 3, 5)
    l1_max = np.max(np.abs(L[..., 1]))
    plt.imshow(L[:, :, 1], vmin=-l1_max, vmax=l1_max, cmap="RdYlBu_r", origin="lower")
    plt.colorbar()
    l1_sum = np.sum(L[..., 1])
    plt.title("Eigenvalue 1, integral = " + str(l1_sum))
    
    plt.subplot(2, 3, 6)
    l1_max = np.max(np.abs(L[..., 1]))
    
    if glyphs:
        plt.quiver(V[:,:,0, 1], V[:, :, 1, 1], headwidth=0, headaxislength=0, headlength=0, pivot="mid", color=(0.0, 0.0, 0.0, 0.5))

        if glyphscalar == "l1":
            plt.imshow(L[:, :, 1], vmin=-l1_max, vmax=l1_max, cmap="RdYlBu_r", origin="lower")
            plt.colorbar()
        l1_sum = np.sum(L[..., 1])
        plt.title("Eigenvalue 1, integral = " + str(l1_sum))
        
    plt.show()
    
# performs iterative voting on a tensor field T
# sigma is the standard deviation for the first iteration
# iterations is the number of iterations (voting passes)
# dsigma shows how much sigma is increased every iteration
def iterative_vote2(T, sigma, iterations, dsigma=1):
    
    V = []
    V.append(T)
    for i in range(iterations):
        
        T = vote2(V[i], sigma + dsigma * i)

        V.append(T)
        
    return V

# performs iterative voting on a tensor field T
# sigma is the standard deviation for the first iteration
# iterations is the number of iterations (voting passes)
# dsigma shows how much sigma is increased every iteration
def iterative_stick2(T, sigma, iterations, dsigma=1):
    
    V = []
    V.append(T)
    for i in range(iterations):
        
        T = stickvote2(V[i], sigma + dsigma * i)

        V.append(T)
        
    return V

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensor Voting')
    parser.add_argument('-i', '--input', type=str, help='Input file (npy)')
    parser.add_argument('-o', '--output', type=str, help='Output file (npy)')
    parser.add_argument('-s', '--sigma', type=float, help='Sigma for tensor vote. Default=5', default=5)
    parser.add_argument('-s2', '--sigma2', type=float, help='Sigma to add orthogonal votes. Default=0', default=0)
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize the tensor field')
    parser.add_argument('-t', '--tensor', type=str, help='Type of tensor field (both, stick, plate, none)')
    args = parser.parse_args()
    
    if args.input is None:
        print("No input file specified")
        exit()

    T = np.load(args.input)

    tvRan = False
    if args.tensor == 'both':
        V = vote2(T, args.sigma, args.sigma2)
        tvRange = True
    elif args.tensor == 'stick':
        V = stickvote2(T, args.sigma, args.sigma2)
        tvRange = True
    elif args.tensor == 'plate':
        V = platevote2(T, args.sigma, args.sigma2)
        tvRange = True
        
    if args.output is not None:
        np.save(args.output, V)    
    
    if args.visualize:
        if tvRan:
            visualize(V)
        else:
            visualize(T)

