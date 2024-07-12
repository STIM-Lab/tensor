import numpy as np
import matplotlib.pyplot as plt

'''

arg1: input file
arg2: output file
arg3: sigma for image to tensor conversion
arg: tensor voting sigma

'''

def stickfield2(qx, qy, RX, RY, sigma1, sigma2):
    
    q = np.zeros((2, 1))
    q[0] = qx
    q[1] = qy
    
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2)
    
    # calculate the normalized direction vector
    D = np.zeros((RX.shape[0], RX.shape[1], 2, 1))
    D[:, :, 0, 0] = np.divide(RX, L, out=np.zeros_like(RX), where=L!=0)
    D[:, :, 1, 0] = np.divide(RY, L, out=np.zeros_like(RY), where=L!=0)
    
    # calculate the rotated stick direction
    #R = np.zeros((DX.shape[0], DX.shape[1], 2, 2))
    Dt = np.transpose(D, axes=(0, 1, 3, 2))
    I = np.eye(2)
    R = I - 2 * np.matmul(D, Dt)
    Rq = np.matmul(R, q)
    Rqt = np.transpose(Rq, (0, 1, 3, 2))
    
    # calculate the decay based on the desired properties
    if sigma1 == 0:
        g1 = 0
    else:
        g1 = np.exp(- L**2 / sigma1**2)[..., np.newaxis, np.newaxis]
        
    if sigma2 == 0:
        g2 = 0
    else:
        g2 = np.exp(- L**2 / sigma2**2)[..., np.newaxis, np.newaxis]
    
    qTd = np.matmul(np.transpose(q), D)
    cos_2_theta = qTd**2
    sin_2_theta = 1 - cos_2_theta
    
    decay = g1 * sin_2_theta + g2 * cos_2_theta
    
    V = decay * np.matmul(Rq, Rqt)
    return V

# calculate the vote result of the tensor field T
# k is the eigenvector used as the voting direction
# sigma is the standard deviation of the vote field
def stickvote2(T, sigma=3, sigma2=0):
    
    # perform the eigendecomposition of the field
    evals, evecs = np.linalg.eigh(T)
    
    # store the eigenvector corresponding to the largest eigenvalue
    E = evecs[:, :, :, 1]
    
    # calculate the optimal window size
    w = int(6 * sigma + 1)
    x = np.linspace(-(w-1)/2, (w-1)/2, w)
    X0, X1 = np.meshgrid(x, x)
    
    # create a padded vote field to store the vote results
    pad = int(3*sigma)
    VF = np.pad(np.zeros(T.shape), ((pad, pad), (pad, pad), (0, 0), (0, 0)))
    
    # for each pixel in the tensor field
    for x0 in range(T.shape[0]):
        for x1 in range(T.shape[1]):
            scale = evals[x0, x1, 1] - evals[x0, x1, 0]
            S = scale * stickfield2(E[x0, x1, 0], E[x0, x1, 1], X0, X1, sigma, sigma2)
            VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] = VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] + S
    return VF[pad:-pad, pad:-pad, :, :]

def platefield2(RX, RY, sigma1, sigma2):
    # calculate the length (distance) value
    L = np.sqrt(RX**2 + RY**2)
    
    # calculate the decay based on the desired properties
    if sigma1 == 0:
        g1 = 0
    else:
        g1 = np.exp(- L**2 / sigma1**2)[..., np.newaxis, np.newaxis]
        
    if sigma2 == 0:
        g2 = 0
    else:
        g2 = np.exp(- L**2 / sigma2**2)[..., np.newaxis, np.newaxis]
        
    # calculate each component of the integral
    C1 = g1 * (np.pi/2) * np.eye(2)
    
    # calculate the normalized direction vector
    D = np.zeros((RX.shape[0], RX.shape[1], 2, 1))
    D[:, :, 0, 0] = np.divide(RX, L, out=np.zeros_like(RX), where=L!=0)
    D[:, :, 1, 0] = np.divide(RY, L, out=np.zeros_like(RY), where=L!=0)
    
    PHI = np.arctan2(D[:, :, 1, 0], D[:, :, 0, 0])
    
    # calculate the matrix component
    I2 = np.zeros((RX.shape[0], RX.shape[1], 2, 2))
    I2[:, :, 0, 0] = np.cos(2 * PHI) + 2
    I2[:, :, 1, 1] = 2 - np.cos(2 * PHI)
    I2[:, :, 0, 1] = np.sin(2 * PHI)
    I2[:, :, 1, 0] = I2[:, :, 0, 1]
    
    C2 = g2 * (np.pi/8) * I2
    C3 = g1 * (np.pi/8) * I2
    
    P = C1 + C2 - C3
    return P


# calculate the vote result of the tensor field T
# k is the eigenvector used as the voting direction
# sigma is the standard deviation of the vote field
def platevote2(T, sigma=3, sigma2=0):
    
    # perform the eigendecomposition of the field
    evals, evecs = np.linalg.eigh(T)
    
    # store the eigenvector corresponding to the smallest eigenvector
    #E = evecs[:, :, :, 0]
    
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

def vote2(T, sigma=3, sigma2=0):
    
    S = stickvote2(T, sigma, sigma2)
    P = platevote2(T, sigma, sigma2)
    
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
    
    
    
    

# visualize a tensor field T (NxMx2x2)
def visualize(T, title = "Tensor Field", fontsize=10, mode=None, glyphs=True):
    
    font = {'size' : fontsize}

    plt.rc('font', **font)
    #plt.figure()
    Eval, Evec = np.linalg.eigh(T)
    if(glyphs == True):
        plt.quiver(Evec[:, :, 0, 1], Evec[:, :, 1, 1], pivot="middle", headwidth=0, headlength=0, headaxislength=0, width=0.002)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    if mode is None or mode == "eval":
        plt.imshow(Eval[:, :, 1], origin="lower", cmap="magma")
    if mode == "eccentricity":
        e0_2 = Eval[:, :, 0] ** 2
        e1_2 = Eval[:, :, 1] ** 2
        ratio = np.divide(e0_2, e1_2, out=np.ones_like(e1_2), where=e1_2!=0)
        ecc = np.sqrt(1 - ratio)
        plt.imshow(ecc, origin="lower", cmap="RdYlBu_r")
    plt.colorbar()
    plt.title(title)
    #plt.show()
    
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