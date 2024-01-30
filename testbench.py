import structen as st
import tensorvote as tv
import skimage
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from generate_data import axis_grid_2d
from structure2d import structure2d_nz
import subprocess
import os

# This function generates test data and runs both the Python a C/CUDA code

# run the generate_data.py script to create raw images

if not os.path.exists('Data_Output'):
    os.makedirs('Data_Output')

print('Generating data...')
def generate_data(lower: int, upper: int, step: int, boxes=10, linewidth=1, noise=0):
   
    data = []
    for i in range(lower, upper, step):
        data.append(axis_grid_2d(i, boxes, linewidth, noise))
        skimage.io.imsave(os.path.join('Data_Output', 'data_' + str(i) + '.bmp'), np.uint8(data[-1] * 255))
    return data

# --------- STRUCTEN -----------------
# run the Python structen script to generate initial tensor fields

# run the C structen script to generate initial tensor fields

# compare the Python and C tensor fields
# input_data = generate_data(500, 5000, 500)
input_data = generate_data(100, 200, 100)
python_structuretensors = []
print('Generating python structure tensors...')
for i in range(len(input_data)):
    python_structuretensors.append(structure2d_nz(input_data[i]))

print('Generating C structure tensors...')
if not os.path.exists('C_structure'):
    os.makedirs('C_structure')

for filename in os.listdir('Data_Output'):
    
    subprocess.run(['./structen.exe', '--input', os.path.join('Data_Output', filename), '--output', os.path.join('C_structure', filename.split('.')[0] + '.npy')])

# -------- TENSORVOTE ----------------
# run the Python tensorvote script to generate a final tensor field

# run the C tensorvote script to generate a final tensor field

# LATER: run the CUDA tensorvote script to generate a final tensor field

# compare all of the tensor fields
    
if not os.path.exists('Python_Output'):
    os.makedirs('Python_Output')

if not os.path.exists('C_Output'):
    os.makedirs('C_Output')

sigma = 10
iterations = 1
print('Running python tensorvote...')
for i in range(len(input_data)):
    votefields = tv.iterative_vote(python_structuretensors[i], sigma, iterations)
    # tv.visualize(votefields[-1])
    np.save(os.path.join('Python_Output', 'vote_' + str(i) + '.npy'), votefields[-1].astype(np.float32))

print('Running C tensorvote...')
for filename in os.listdir('C_structure'):
    subprocess.run(['./tensorvote.exe', 
                    '--input', os.path.join('C_structure', filename), 
                    '--output', os.path.join('C_Output', filename), 
                    '--sigma', str(sigma), 
                    'cuda', '-1'])