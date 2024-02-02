import structen as st
import tensorvote as tv
import skimage
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from generate_data import axis_grid_2d
from structen import structure2d_nz
import subprocess
import os
import time
import shutil

# function generates a series of grids used to test the structen and tensor voting algorithms
def generate_grids(lower: int, upper: int, step: int, boxes=10, linewidth=1, noise=0):
    print('Generating grids...')
   
    data = []
    for i in range(lower, upper, step):
        data.append(axis_grid_2d(i, boxes, linewidth, noise))
        skimage.io.imsave(os.path.join(data_directory, 'grid_' + str(i) + '.bmp'), np.uint8(data[-1] * 255))
    return data


# create a directory to store the raw image data
data_directory = "test_data"
if os.path.exists(data_directory):
    shutil.rmtree(data_directory)
os.makedirs(data_directory)    


# generate grids used as input
input_data = generate_grids(100, 200, 100)
input_filenames = os.listdir(data_directory)

# run the python structure tensor code
python_structuretensors = []
print('Generating python structure tensors from grids...')
for i in range(len(input_data)):
    start = time.time()
    python_structuretensors.append(structure2d_nz(input_data[i]))
    end = time.time()
    print('Generating structure tensor in Python for grid size', input_data[i].shape[0], 'took', end - start, 'seconds.')

print('Generating C structure tensors from grids...')

# run the C structure tensor code
for filename in input_filenames:
    start = time.time()
    subprocess.run(['./structen.exe', 
                    '--input', os.path.join(data_directory, filename), 
                    '--output', os.path.join(data_directory, "c_" + filename.split('.')[0] + '.npy')])
    end = time.time()
    print('Generating structure tensor in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')

# compare the C and Python structure tensors
# Omar: think of some comparison metric - maybe mean squared error?
    
sigma = 10
iterations = 1
cuda = -1
print('Running Python tensorvote...')
for i in range(len(input_data)):
    start = time.time()
    python_votefields = tv.iterative_vote(python_structuretensors[i], sigma, iterations)
    np.save(os.path.join(data_directory, 'python_vote_' + filename.split('.')[0] + '.npy'), python_votefields[-1].astype(np.float32))
    end = time.time()
    print('Running tensorvote in Python for grid size', input_data[i].shape[0], 'took', end - start, 'seconds.')

print('Running C tensorvote...')
for filename in input_filenames:
    start = time.time()
    subprocess.run(['./tensorvote.exe', 
                    '--input', os.path.join(data_directory, "c_" + filename.split('.')[0] + ".npy"), 
                    '--output', os.path.join(data_directory, "c_vote_" + filename.split('.')[0] + '.npy'), 
                    '--sigma', str(sigma), 
                    '--cuda', str(cuda)])
    end = time.time()
    print('Running tensorvote in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')