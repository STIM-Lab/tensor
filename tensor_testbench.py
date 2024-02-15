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
    return data

def generate_field_image(x, y, N=100, sigma=10):
    T = np.zeros((N, N))
    T[int(N/2), int(N/2)] = 1

    return T

def python_and_c_tensorvote(input_data, input_filenames, python_structuretensors, sigma, iterations, cuda, directory):
    python_tensorvote_fields = []
    print('Running Python tensorvote...')
    for i in range(len(input_data)):
        start = time.time()
        python_tensorvote_fields.append(tv.iterative_vote(python_structuretensors[i], sigma, iterations)[-1])
        end = time.time()
        print('Running tensorvote in Python for grid size', input_data[i].shape[0], 'took', end - start, 'seconds.')

    print('Running C tensorvote...')
    for filename in input_filenames:
        start = time.time()
        subprocess.run(['./tensorvote.exe', 
                        '--input', os.path.join(directory, "c_" + filename.split('.')[0] + ".npy"), 
                        '--output', os.path.join(directory, "c_vote_" + filename.split('.')[0] + '.npy'), 
                        '--sigma', str(sigma), 
                        '--cuda', str(cuda)])
        end = time.time()
        print('Running tensorvote in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')
        
    c_tensorvote_fields = []
    for filename in input_filenames:
        c_tensorvote_fields.append(np.load(os.path.join(directory, "c_vote_" + filename.split('.')[0] + '.npy')))
        
    return python_tensorvote_fields, c_tensorvote_fields
        
def tensor_field_difference(field_one, field_two):
    
    if field_one.shape != field_two.shape:
        raise ValueError("The two fields must have the same shape") 
    
    shape = field_one.shape
    
    summed_squared_error = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            squared_differences = (field_one[i][j] - field_two[i][j])**2
            summed_squared_error[i][j] += np.sum(squared_differences)
                
    return summed_squared_error

def main():
    # create a directory to store the raw image data    
    data_directory = "test_data"
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
    os.makedirs(data_directory)    

    # generate grids used as input
    # input_data = generate_grids(100, 200, 100)
    
    input_data = [generate_field_image(1, 1)]
    
    for image in input_data:
        skimage.io.imsave(os.path.join(data_directory, 'grid_' + str(image.shape[0]) + '.bmp'), np.uint8(image))
    
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

    # load the C structure tensors
    c_structuretensors = []
    for filename in input_filenames:
        c_structuretensors.append(np.load(os.path.join(data_directory, "c_" + filename.split('.')[0] + '.npy')))
        
    sigma = 10
    iterations = 1
    cuda = -1
    python_votefields, c_votefields = python_and_c_tensorvote(input_data, input_filenames, python_structuretensors, sigma, iterations, cuda, data_directory)
    
    for field in python_votefields:
        np.save(os.path.join(data_directory, 'python_vote_' + input_filenames[i].split('.')[0] + '.npy'), (field).astype(np.float32))
        
    
    # compare the tensor fields
    print('Comparing tensor fields...')
    difference = tensor_field_difference(python_votefields[0], c_votefields[0])
    plt.imshow(difference)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()