import numpy as np
import tqdm
import matplotlib.pyplot as plt

'''

arg1: input file
arg2: output file
arg3: sigma for image to tensor conversion
arg: tensor voting sigma

'''

def decay_wu(cos_theta, length, sigma):

    c = np.exp(-(length**2) / (sigma**2))
    
    radial = (1 - cos_theta ** 2)
    
    D = c * radial
   
    return D

# generate a saliency field for a stick tensor with direction e sampled for all
# points given by the distance L and orientation vector V
def saliency_wu(e, L, V0, V1, sigma):
    
    # calculate the dot product between the eigenvector and the orientation
    eTv = e[0] * V0 + e[1] * V1
    
    # calculate the radius of the osculating circle
    R = np.divide(L, (2 * eTv), out=np.zeros_like(L), where=eTv!=0)
    
    d = decay_wu(eTv, L, sigma)
    
    # calculate the target tensor orientation
    Ep0 = np.divide((R * e[0] - L * V0), R, out=np.ones_like(L) * e[0], where=R!=0)
    Ep1 = np.divide((R * e[1] - L * V1), R, out=np.ones_like(L) * e[1], where=R!=0) 
    
    
    # turn the orientation into a stick tensor
    S = np.zeros((V0.shape[0], V1.shape[1], 2, 2))
    S[:, :, 0, 0] = Ep0 ** 2
    S[:, :, 1, 1] = Ep1 ** 2
    S[:, :, 1, 0] = Ep0 * Ep1
    S[:, :, 0, 1] = S[:, :, 1, 0]
    
    # scale the stick tensor by the decay function
    d = np.expand_dims(d, 2)
    d = np.expand_dims(d, 3)
    
    return S * d
    

# calculate the vote result of the tensor field T
# k is the eigenvector used as the voting direction
# sigma is the standard deviation of the vote field
def vote_k_wu(T, k=0, sigma=3):
    
    # perform the eigendecomposition of the field
    evals, evecs = np.linalg.eigh(T)
    
    # store the eigenvector corresponding to eigenvalue k
    E = evecs[:, :, :, k]
    
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
            scale = evals[x0, x1, k]
            S = scale * saliency_wu(E[x0, x1], L, V0, V1, sigma)
            VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] = VF[x0:x0 + S.shape[0], x1:x1 + S.shape[1]] + S
    return VF[pad:-pad, pad:-pad, :, :]

def vector2tensor(x, y):

    vector = np.array([x, y])
    vector = vector / np.linalg.norm(vector)


    tensor = np.zeros([2, 2])
    tensor[0,0] = vector[0] * vector[0]
    tensor[0,1] = vector[0] * vector[1]
    tensor[1,0] = vector[1] * vector[0]
    tensor[1,1] = vector[1] * vector[1]

    return tensor

#generate an NxN field with a stick tensor pointing in the (x,y) direction
def generate_stick_field(x=0, y=1, N=100):
    T = np.zeros((N, N, 2, 2))
    center = int(N/2)
    T[center, center] = vector2tensor(x, y)

    return T
    
# generates a test voting field around a stick tensor pointed in the direction (x, y)
# (x, y) is the orientation of the stick tensor
# N is the resolution of the test field
# sigma is the decay falloff
def testfield(x, y, N=100, sigma=10):

    # generate a field with a single stick tensor in the middle
    T = generate_stick_field(x, y, N)

    # apply tensor voting
    VF = vote_k_wu(T, 0, sigma)

    return VF




# visualize a tensor field T (NxMx2x2)
def visualize(T, plot_title = "Tensor Field", mode=None):
    plt.figure()
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
    plt.title(plot_title)
    plt.show()
    
# performs iterative voting on a tensor field T
# sigma is the standard deviation for the first iteration
# iterations is the number of iterations (voting passes)
# dsigma shows how much sigma is increased every iteration
def iterative_vote(T, sigma, iterations, dsigma=1):
    
    V = []
    V.append(T)
    for i in range(iterations):
        
        T = vote_k_wu(V[i], 1, sigma + dsigma * i)

        V.append(T)
        
    return V