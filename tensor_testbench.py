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
        
def tensor_field_difference(field_one, field_two, visualize=False, name = ""):
    
    if field_one.shape != field_two.shape:
        raise ValueError("The two fields must have the same shape") 
    
    shape = field_one.shape
    
    summed_squared_error = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            squared_differences = (field_one[i][j] - field_two[i][j])**2
            summed_squared_error[i][j] += np.sum(squared_differences)
    if(visualize):
        plt.figure()
        plt.imshow(summed_squared_error)
        plt.title("Error for test case " + name)
        plt.show()
         
    return np.max(summed_squared_error)

def run_grid_test(data_directory, visualize=False):
    
    # generate the grids used as input to both algorithms
    input_data = []
    input_data.append(axis_grid_2d(100, 4, 2, 0.00) * 255)
    input_data.append(axis_grid_2d(200, 4, 4, 0.01) * 255)
    input_data.append(axis_grid_2d(300, 4, 6, 0.10) * 255)
    input_data.append(axis_grid_2d(400, 4, 8, 0.00) * 255)
    
    
    # save an image for each grid that will be used to test the C code
    input_filenames = []
    for i in range(len(input_data)):
        input_filenames.append('grid_' + str(input_data[i].shape[0]) + '.bmp')
        skimage.io.imsave(os.path.join(data_directory, input_filenames[i]), np.uint8(input_data[i]))
        
    # run each image through the Python structure tensor code
    python_structuretensors = []
    print('Generating python structure tensors from grids...')
    for i in range(len(input_data)):
        start = time.time()
        python_structuretensors.append(st.structure2d(input_data[i], sigma=0))
        end = time.time()
        print('Generating structure tensor in Python for grid size', input_data[i].shape[0], 'took', end - start, 'seconds.')
        
    print('Generating C structure tensors from grids...')

    # run the saved images through the C structure tensor code
    for filename in input_filenames:
        start = time.time()
        subprocess.run(['./structen.exe', 
                        '--input', os.path.join(data_directory, filename), 
                        '--order', "2",
                        '--output', os.path.join(data_directory, "c_" + filename.split('.')[0] + '.npy')])
        end = time.time()
        print('Generating structure tensor in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')
    
    # compare the results of the C sturcture tensor and Python code
    print("\n\n--------Comparing C/Python Structure Tensor Results--------")
    c_structuretensors = []
    for ti in range(len(input_filenames)):
        c_structuretensors.append(np.load(os.path.join(data_directory, "c_" + input_filenames[ti].split('.')[0] + '.npy')))
        error = tensor_field_difference(c_structuretensors[ti], python_structuretensors[ti], visualize, input_filenames[ti].split('.')[0])
        print("Grid test ( " + input_filenames[ti].split('.')[0] + " ):  error = " + str(error))
        
    return python_structuretensors, c_structuretensors

def main():
    # create a directory to store the raw image data    
    print("Creating test image directory...")
    data_directory = "test_data"
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
    os.makedirs(data_directory)    
    print("done.")

    # run the grid test
    python_structuretensors, c_structuretensors = run_grid_test(data_directory, True)
    
    # input_data = [tv.generate_stick_field(1, 1, 100)]
   
        
    # sigma = 10
    # iterations = 1
    # cuda = -1
    # python_votefields, c_votefields = python_and_c_tensorvote(input_data, input_filenames, python_structuretensors, sigma, iterations, cuda, data_directory)
    
    # for field in python_votefields:
    #     np.save(os.path.join(data_directory, 'python_vote_' + input_filenames[i].split('.')[0] + '.npy'), (field).astype(np.float32))
        
    
    # # compare the tensor fields
    # print('Comparing tensor fields...')
    # difference = tensor_field_difference(python_votefields[0], c_votefields[0])
    # plt.imshow(difference)
    # plt.colorbar()
    # plt.show()

if __name__ == "__main__":
    main()