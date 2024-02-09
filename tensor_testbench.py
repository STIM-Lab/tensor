import structen as st
import tensorvote as tv
import skimage
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from generate_data import axis_grid_2d
import subprocess
import os
import time
import shutil

# function generates a series of grids used to test the structen and tensor voting algorithms
def generate_grids(lower: int, upper: int, step: int, boxes=10, linewidth=1, noise=0):
    print('Generating grids...')
   
    data = []
    for i in range(lower, upper, step):
        data.append(axis_grid_2d(i, boxes, linewidth, noise) * 255)
        skimage.io.imsave(os.path.join(data_directory, 'grid_' + str(i) + '.bmp'), np.uint8(data[-1]))
    return data

def voting(input_data, input_filenames, python_structuretensors, sigma, iterations, cuda):
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
        
def structure_tensor_difference(python_structuretensors, c_structuretensors):
    summed_squared_error = np.zeros(input_data[0].shape)
    for struc in range(len(input_data)):
        for i in range(len(input_data[0])):
            for j in range(len(input_data[0])):
                squared_differences = (python_structuretensors[struc][i][j] - c_structuretensors[struc][i][j])**2
                summed_squared_error[i][j] += np.sum(squared_differences)
                
    return summed_squared_error


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
    python_structuretensors.append(st.structure2d(input_data[i], sigma=0))
    end = time.time()
    print('Generating structure tensor in Python for grid size', input_data[i].shape[0], 'took', end - start, 'seconds.')

print('Generating C structure tensors from grids...')

# run the C structure tensor code
for filename in input_filenames:
    start = time.time()
    subprocess.run(['./structen.exe', 
                    '--input', os.path.join(data_directory, filename), 
                    '--order', "2",
                    '--output', os.path.join(data_directory, "c_" + filename.split('.')[0] + '.npy')])
    end = time.time()
    print('Generating structure tensor in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')

# compare the C and Python structure tensors
# Omar: think of some comparison metric - maybe mean squared error?

# load the C structure tensors
c_structuretensors = []
for filename in input_filenames:
    c_structuretensors.append(np.load(os.path.join(data_directory, "c_" + filename.split('.')[0] + '.npy')))

# Calculate mean squared error for each pixel point
# subtract each pixel point and square it, then take the sum of the matrix for each pixel point    
# compare the C and Python structure tensors amd vosualize the difference
# for i in range(len(input_data)):
#     print('Comparing structure tensors for grid size', input_data[i].shape[0], '...')

#     error = np.mean((python_structuretensors[i] - c_structuretensors[i])**2)
#     print('Mean squared error:', error)
# create a matrix in the shape of the input data
# summed_squared_error = np.zeros(input_data[0].shape)
# for struc in range(len(input_data)):
#     for i in range(len(input_data[0])):
#         for j in range(len(input_data[0])):
#             squared_differences = (python_structuretensors[struc][i][j] - c_structuretensors[struc][i][j])**2
#             summed_squared_error[i][j] += np.sum(squared_differences)
    
summed_squared_difference = structure_tensor_difference(python_structuretensors, c_structuretensors)

# visualize the difference
plt.figure()
plt.imshow(summed_squared_difference, cmap='hot', interpolation='nearest')
plt.title("Mean Squared Error per Pixel")
plt.colorbar()
plt.show()

tv.visualize(python_structuretensors[0], "Python Structure Tensor Output")


tv.visualize(c_structuretensors[0], "C/C++ Structure Tensor Output")
    
sigma = 10
iterations = 1
cuda = -1
# voting(input_data, input_filenames, python_structuretensors, sigma, iterations, cuda)
# print('Running Python tensorvote...')
# for i in range(len(input_data)):
#     start = time.time()
#     python_votefields = tv.iterative_vote(python_structuretensors[i], sigma, iterations)
#     np.save(os.path.join(data_directory, 'python_vote_' + filename.split('.')[0] + '.npy'), python_votefields[-1].astype(np.float32))
#     end = time.time()
#     print('Running tensorvote in Python for grid size', input_data[i].shape[0], 'took', end - start, 'seconds.')

# print('Running C tensorvote...')
# for filename in input_filenames:
#     start = time.time()
#     subprocess.run(['./tensorvote.exe', 
#                     '--input', os.path.join(data_directory, "c_" + filename.split('.')[0] + ".npy"), 
#                     '--output', os.path.join(data_directory, "c_vote_" + filename.split('.')[0] + '.npy'), 
#                     '--sigma', str(sigma), 
#                     '--cuda', str(cuda)])
#     end = time.time()
#     print('Running tensorvote in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')