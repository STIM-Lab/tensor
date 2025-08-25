import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

def metric(alpha, A, B):
    Aa = alpha * A
    D = Aa - B
    return np.sum(np.linalg.norm(D, axis=(2,3)))

def opt_metric(A, B, steps):
    
    min_m = metric(0.0, A, B)
    min_a = 0.0
    
    step = 1.0 / steps
    for i in range(steps):
        alpha = (i + 1) * step
        m = metric(alpha, A, B)
        if m < min_m:
            min_m = m
            min_a = alpha
            
    return min_m, min_a

ground_truth = "C:/Users/david/Documents/build/tensor-bld/demo.npy"
test_case = "C:/Users/david/Documents/build/tensor-bld/demo_noise.npy"
dest_size = 92

A = np.load(ground_truth)
B = np.load(test_case)
B = np.reshape(B, (B.shape[0], B.shape[1], 2, 2))

A_size = A.shape[0]
A_start = int((A_size - dest_size) / 2)
A_end = int(A_size - A_start)
A = A[A_start:A_end, A_start:A_end, :, :]
 
B_size = B.shape[0]
B_start = int((B_size - dest_size) / 2)
B_end = int(B_size - B_start)
B = B[B_start:B_end, B_start:B_end, :, :]


x = np.array([0.0, 1.0])

m, a = opt_metric(A, B, 1000)