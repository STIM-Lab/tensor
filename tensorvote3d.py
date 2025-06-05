import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp


def eigmag(T):
    
    eigenValues, eigenVectors = np.linalg.eigh(T)
    magValues = np.abs(eigenValues)
    
    idx = np.argsort(magValues, -1)
    
    sortedValues = np.take_along_axis(eigenValues, idx, -1)
    
    sortedVectors = np.zeros_like(eigenVectors)
    sortedVectors[..., 0, :] = np.take_along_axis(eigenVectors[..., 0, :], idx, -1)
    sortedVectors[..., 1, :] = np.take_along_axis(eigenVectors[..., 1, :], idx, -1)
    sortedVectors[..., 2, :] = np.take_along_axis(eigenVectors[..., 2, :], idx, -1)
    
    return sortedValues, sortedVectors

# calculates the normalization factor for a stick tensor field
def eta(sigma1, sigma2, p):
    
    if sigma1 == 0 and sigma2 == 0:
        return 1.0
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
    
# calculates the voting field for a stick tensor using refined tensor voting
# qx, qy, qz is the orientation of the stick field (largest eigenvector)
def stickfield3(qx, qy, qz, RX, RY, RZ, sigma1, sigma2=0, power=1, normalize=True):
    q = np.zeros((3, 1))
    q[0] = qx
    q[1] = qy
    q[2] = qz
    
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2 + RZ**2)
    
    # calculate the normalized direction vector
    D = np.zeros((RX.shape[0], RX.shape[1], RX.shape[2], 3, 1))
    
    # if L == 0, assume that D is zero (and 1-qTd = 1) and it contributes to itself
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
        d1 = np.zeros_like(L)
        d1[L == 0] = 1
        #d1 = 0
    else:
        d1 = np.exp(- L**2 / sigma1**2)
        
    if sigma2 == 0:
        d2 = np.zeros_like(L)
        d2[L == 0] = 1
        #d2 = 0
    else:
        d2 = np.exp(- L**2 / sigma2**2)
    
    qTd = np.squeeze(np.matmul(np.transpose(q), D))
    cos_2p_theta = (qTd**2)**power
    sin_2p_theta = (1 - qTd**2)**power    
    
    DECAY = (d1 * sin_2p_theta + d2 * cos_2p_theta)[..., np.newaxis, np.newaxis]
    
    #decay = g1 * sin_2_theta + g2 * cos_2_theta
    if normalize:
        V = eta(sigma1, sigma2, power) * DECAY * np.matmul(Rq, Rqt)
        return V.astype(np.float32)
    else:
        V = DECAY * np.matmul(Rq, Rqt)
        return V.astype(np.float32)

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
    Z = np.zeros(T.shape, dtype=np.float32)
    VF = np.pad(Z, ((pad, pad), (pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        print('\nx0: ', end='')
        for x1 in range(T.shape[1]):
            for x2 in range(T.shape[2]):
                scale = (evals_mag[x0, x1, x2, 2] - evals_mag[x0, x1, x2, 1]) * np.sign(evals[x0, x1, x2, 2])
                S = scale * stickfield3(E[x0, x1, x2, 0], E[x0, x1, x2, 1], E[x0, x1, x2, 2], X0, X1, X2, sigma, sigma2)
                VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1], x2:x2 + S.shape[2]] += S
        print(x0)
    return VF[pad:-pad, pad:-pad, pad:-pad, :, :]

def platefield3(RX, RY, RZ, sigma1=3, sigma2=0, p=1, normalize=False):
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2 + RZ**2)
    
    # calculate the direction matrix d
    d = np.zeros((RX.shape[0], RX.shape[1], RX.shape[2], 3, 1))
    d[:, :, :, 0, 0] = np.divide(RX, L, where=L!=0)
    d[:, :, :, 1, 0] = np.divide(RY, L, where=L!=0)
    d[:, :, :, 2, 0] = np.divide(RZ, L, where=L!=0)
    
    # calculate the length and the angle of the reflected direction matrix d
    alpha = np.sqrt(np.square(d[:, :, :, 0, 0]) + np.square(d[:, :, :, 1, 0]))
    a2 = alpha * alpha
    phi = 0
    phi = np.arctan2(d[:, :, :, 1, 0], d[:, :, :, 0, 0])
    
    # calculate the rotation matrix
    R_z = np.zeros(phi.shape + (3, 3))
    R_z[:, :, :, 0, 0] = np.cos(phi)
    R_z[:, :, :, 0, 1] = -np.sin(phi)
    R_z[:, :, :, 1, 0] = np.sin(phi)
    R_z[:, :, :, 1, 1] = np.cos(phi)
    R_z[:, :, :, 2, 2] = 1
    
    R_z_rev = np.copy(R_z)          # for -phi
    R_z_rev[:, :, :, 0, 1] = R_z[:, :, :, 1, 0]
    R_z_rev[:, :, :, 1, 0] = R_z[:, :, :, 0, 1]
    
    # calculate the pre-defined integrals
    J_0 = (np.pi/2) * sp.hyp2f1(-p, 1.5, 2, a2)
    J_1 = np.pi * sp.hyp2f1(-p, 0.5, 1, a2)
    K_0 = sp.beta(0.5, p + 1.5)
    K_1 = sp.beta(0.5, p + 0.5)
    
    # calculate each term
    A = np.zeros(a2.shape + (3, 3))
    A[:, :, :, 0, 0] = ((1-2*a2)**2) * J_0
    A[:, :, :, 0, 2] = -2*alpha*d[:, :, :, 2, 0]*(1-2*a2)*J_0
    A[:, :, :, 1, 1] = J_1 - J_0
    A[:, :, :, 2, 0] = A[:, :, :, 0, 2]
    A[:, :, :, 2, 2] = 4*a2*np.square(d[:, :, :, 2, 0]) * J_0
    
    a2_p = np.power(a2, p)
    B = np.zeros(A.shape)
    B[:, :, :, 0, 0] = a2_p*((1-2*a2)**2)*K_0
    B[:, :, :, 0, 2] = -2*alpha*a2_p*d[:, :, :, 2, 0]*(1-2*a2)*K_0
    B[:, :, :, 1, 1] = a2_p*(K_1 - K_0)
    B[:, :, :, 2, 0] = B[:, :, :, 0, 2]
    B[:, :, :, 2, 2] = 4*a2_p*a2*np.square(d[:, :, :, 2, 0])*K_0
    
    # rotate back to the original axis
    term_a = np.matmul(np.matmul(R_z, A), R_z_rev)
    term_b = np.matmul(np.matmul(R_z, B), R_z_rev)
    
    # define the exponential values
    e1 = 0
    e2 = 0
    if(sigma1 > 0):
        e1 = np.exp(- L**2 / sigma1**2)
        e1 = e1[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, 3, 3))
    if (sigma2 > 0):
        e2 = np.exp(- L**2 / sigma2**2)
        e2 = e2[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, 3, 3))
    
    PlateIntegral = e1 * term_a + e2 * term_b  
    
    if normalize == True:
        return eta(sigma1, sigma2, 1) * PlateIntegral
    else:
        return PlateIntegral
    
def platefield3_previous(RX, RY, RZ, sigma1, sigma2=0, normalize=True):
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2 + RZ**2)
    
    # define required matrices
    I_tilde = np.eye(3)
    I_tilde[2, 2] = 0
    
    # calculate the direction matrix
    d = np.zeros((RX.shape[0], RX.shape[1], RX.shape[2], 3, 1))
    d[:, :, :, 0, 0] = np.divide(RX, L, where=L!=0)
    d[:, :, :, 1, 0] = np.divide(RY, L, where=L!=0)
    d[:, :, :, 2, 0] = np.divide(RZ, L, where=L!=0)
    dt = np.transpose(d, axes=(0, 1, 2, 4, 3))
    D = np.matmul(d, dt)
    D_tilde = np.matmul(I_tilde, D)
    D_tilde_T = np.transpose(D_tilde, axes=(0, 1, 2, 4, 3))
    ALPHA = (d[:, :, :, 0, 0]**2) + (d[:, :, :, 1, 0]**2)
    ALPHA = ALPHA[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, 3, 3))
    
    # define the terms
    shared_term = D_tilde + D_tilde_T - (2*ALPHA*D)
    A = np.zeros((RX.shape[0], RX.shape[1], RX.shape[2], 3, 3))
    A = (np.pi / 2) * (I_tilde - 2*shared_term)
    B = np.zeros(A.shape)
    B = (np.pi / 8) * (ALPHA * I_tilde + 2 * np.matmul(D_tilde, I_tilde) - (6 * ALPHA * shared_term))
    
    # define exponential values
    e1 = 0
    e2 = 0
    if(sigma1 > 0):
        e1 = np.exp(- L**2 / sigma1**2)
        e1 = e1[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, 3, 3))
    if (sigma2 > 0):
        e2 = np.exp(- L**2 / sigma2**2)
        e2 = e2[:, :, :, np.newaxis, np.newaxis] * np.ones((1, 1, 1, 3, 3))
    
    PlateIntegral = (e1 * (A - B)) + (e2 * B)

    if normalize == True:
        return eta(sigma1, sigma2, 1) * PlateIntegral
    else:
        return PlateIntegral

def platefield3_numerical(RX, RY, RZ, sigma1, sigma2=0, p=1, N=10, normalize=True):
    T = np.zeros((RX.shape[0], RX.shape[1], RX.shape[2], 3, 3))
    
    dbeta = np.pi / N
    # since the integration is symmetric, we integrate across HALF the unit circle and multiply by 2
    for n in range(N):
        x = np.cos(n * dbeta)
        y = np.sin(n * dbeta)
        z = 0
        Tn = stickfield3(x, y, z, RX, RY, RZ, sigma1, sigma2, p, normalize)
        T = T + (np.pi / N) * Tn
        
    return T

def platevote3_numerical(T, sigma=3, sigma2=0, p=1, N=10, normalize=True):
    # perform eigendecomposition (eigenvalues requireed)
    evals, _ = eigmag(T)
    evals_mag = np.abs(evals)
    
    # optimal window size
    sigmax = max(sigma, sigma2)
    w = int(6 * sigmax + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1, X2 = np.meshgrid(x, x, x)
    
    # padded vote field to store the resutls
    pad = int(3 * sigmax)
    Z = np.zeros(T.shape, dtype=np.float32)
    VF = np.pad(Z, ((pad, pad), (pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        print('x0: ', end='')
        for x1 in range(T.shape[1]):
            for x2 in range(T.shape[2]):
                scale = (evals_mag[x0, x1, x2, 1] - evals_mag[x0, x1, x2, 0]) * np.sign(evals[x0, x1, x2, 1])
                P = scale * platefield3_numerical(X0, X1, X2, sigma, sigma2, p, N, normalize)
                VF[x0:x0 + P.shape[0], x1:x1 + P.shape[1], x2:x2 + P.shape[2]] += P
        print(x0)
    if pad == 0:
        return VF
    return VF[pad:-pad, pad:-pad, pad:-pad, :, :]

def platevote3(T, sigma=3, sigma2=0, p=1, normalize=True):
    # perform eigendecomposition (eigenvalues requireed)
    evals, _ = eigmag(T)
    evals_mag = np.abs(evals)
    
    sigmax = max(sigma, sigma2)
    
    # optimal window size
    w = int(6 * sigmax + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1, X2 = np.meshgrid(x, x, x)
    
    # padded vote field to store the resutls
    pad = int(3 * sigmax)
    Z = np.zeros(T.shape, dtype=np.float32)
    VF = np.pad(Z, ((pad, pad), (pad, pad), (pad, pad), (0, 0), (0, 0)))
    scale, P = 0, 0
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        print('x0: ', end='')
        for x1 in range(T.shape[1]):
            for x2 in range(T.shape[2]):
                scale = (evals_mag[x0, x1, x2, 1] - evals_mag[x0, x1, x2, 0]) * np.sign(evals[x0, x1, x2, 1])
                P = scale * platefield3(X0, X1, X2, sigma, sigma2, p, normalize)
                VF[x0:x0 + P.shape[0], x1:x1 + P.shape[1], x2:x2 + P.shape[2]] += P
        print(x0)
        
    if pad == 0:
        return VF
    
    VOTE = VF[pad:-pad, pad:-pad, pad:-pad, :, :]
    return VOTE

# generate an impulse tensor field to test tensor voting
# N is the size of the field (it will be a cube)
# x, y, z is the orientation of the largest eigenvector
# l2, l1, l0 are the eigenvalues, where l2 >= l1 >= l0
# sigma1 and sigma2
def impulse3(N, x, y, z, l2=1, l1=0, l0=0, sigma1=5, sigma2=0, power=1):
    
    r = np.linspace(-N/2, N/2, N)
    X, Y, Z = np.meshgrid(r, r, r)
    
    l = np.sqrt(x**2 + y**2 + z**2)
    
    #V = stickfield3(x/l, y/l, z/l, X, Y, Z, sigma1, sigma2, power)
    P = platefield3(X, Y, Z, sigma1, sigma2)
    
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
    
    # T = np.zeros((N, N, N, 3, 3)).astype(np.float32)
    
    # T[int(N/2), int(N/2), int(N/2)] = m
    
    # return T
    return P

def sanityfield3(N):
    x = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(x, x, x)
    
    T = np.zeros((N, N, N, 3, 3))
    T[:, :, :, 0, 0] = X * X
    T[:, :, :, 0, 1] = X * Y
    T[:, :, :, 0, 2] = X * Z
    
    T[:, :, :, 1, 0] = Y * X
    T[:, :, :, 1, 1] = Y * Y
    T[:, :, :, 1, 2] = Y * Z
    
    T[:, :, :, 2, 0] = Z * X
    T[:, :, :, 2, 1] = Z * Y
    T[:, :, :, 2, 2] = Z * Z
    
    return T

def visualize3(P):
    vals, vecs = np.linalg.eigh(P)
    l0 = vals[:, :, 0]
    l1 = vals[:, :, 1]
    l2 = vals[:, :, 2]
    
    plt.subplot(2, 3, 1)
    plt.imshow(l0)
    plt.subplot(2, 3, 2)
    plt.imshow(l1)
    plt.subplot(2, 3, 3)
    plt.imshow(l2)
    
    plt.subplot(2, 3, 4)
    l2_l1 = l2 - l1
    l0l1l2 = l0 + l1 + l2
    Cl = np.divide(l2_l1, l0l1l2, out=np.zeros_like(l0), where=l0l1l2!=0)
    plt.imshow(Cl, vmin=0, vmax=1)
    plt.title("Linear Anisotropy")
    
    plt.subplot(2, 3, 5)
    twol1_l0 = 2 * (l1 - l0)
    l0l1l2 = l0 + l1 + l2
    Cp = np.divide(twol1_l0, l0l1l2, out=np.zeros_like(l0), where=l0l1l2!=0)    
    plt.imshow(Cp, vmin=0, vmax=1)
    plt.title("Plate Anisotropy")
    
    plt.subplot(2, 3, 6)
    threel0 = 3 * l0
    l0l1l2 = l0 + l1 + l2
    Cs = np.divide(threel0, l0l1l2, out=np.zeros_like(l0), where=l0l1l2!=0)    
    plt.imshow(Cs, vmin=0, vmax=1)
    plt.title("Spherical Anisotropy")