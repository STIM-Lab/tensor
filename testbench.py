import structure2d as st
import tensorvote as tv
import skimage
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


T = np.load("data/axis_grid.npy")

plt.figure()
tv.visualize(T)

sigma = 5
TF = tv.iterative_vote(T, sigma, 3, 1)

plt.figure()
tv.visualize(TF[-1])


