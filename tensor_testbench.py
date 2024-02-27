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

def python_and_c_tensorvote(input_filenames, python_structuretensors, sigma, iterations, cuda, input_directory, output_directory):
    
    python_tensorvote_fields = []
    print('Running Python tensorvote...')
    for i in range(len(python_structuretensors)):
        start = time.time()
        
        python_tensorvote_fields.append(tv.iterative_vote(python_structuretensors[i], sigma, iterations)[-1])
        end = time.time()
        print('Running tensorvote in Python for grid size', python_structuretensors[i].shape[0], 'took', end - start, 'seconds.')

        np.save(os.path.join(output_directory, 'p_' + input_filenames[i].split('.')[0] + '.npy'), (python_tensorvote_fields[i]).astype(np.float32))

    print('Running C tensorvote...')
    for filename in input_filenames:
        start = time.time()
        subprocess.run(['./tensorvote.exe', 
                        '--input', os.path.join(input_directory, filename.split('.')[0] + ".npy"), 
                        '--output', os.path.join(output_directory, 'c_' + filename.split('.')[0] + '.npy'), 
                        '--sigma', str(sigma), 
                        '--cuda', str(cuda)])
        end = time.time()
        print('Running tensorvote in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')
        
    c_tensorvote_fields = []
    for filename in input_filenames:
        c_tensorvote_fields.append(np.load(os.path.join(output_directory, 'c_' + filename.split('.')[0] + '.npy')))
        
    return python_tensorvote_fields, c_tensorvote_fields
        
def tensor_field_difference(field_one, field_two, band=1, visualize=False, name = ""):
    
    if field_one.shape != field_two.shape:
        raise ValueError("The two fields must have the same shape") 
    
    shape = field_one.shape
    
    summed_squared_error = np.zeros((shape[0], shape[1]))
    for i in range(band, shape[0]-band):
        for j in range(band, shape[1]-band):
            squared_differences = (field_one[i][j] - field_two[i][j])**2
            summed_squared_error[i][j] += np.sum(squared_differences)
    if(visualize):
        plt.figure()
        plt.imshow(summed_squared_error)
        plt.title("Error for test case " + name)
        plt.colorbar()
        plt.show()
         
    return np.max(summed_squared_error)

def run_grid_test(data_directory, input_field_directory, visualize=False):
    
    image_directory = os.path.join(data_directory, "images")
    
    os.makedirs(image_directory)
    
    # generate the grids used as input to both algorithms
    input_data = []
    input_data.append(axis_grid_2d(3, 2, 1, 0.10))
    input_data.append(axis_grid_2d(100, 4, 2, 0.00))
    input_data.append(axis_grid_2d(200, 4, 4, 0.01))
    input_data.append(axis_grid_2d(300, 4, 6, 0.10))
    input_data.append(axis_grid_2d(400, 4, 8, 0.20))
    
    
    # save an image for each grid that will be used to test the C code
    input_filenames = []
    for i in range(len(input_data)):
        input_filenames.append('grid_' + str(input_data[i].shape[0]) + '.bmp')
        skimage.io.imsave(os.path.join(image_directory, input_filenames[i]), np.uint8(input_data[i]))
        
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
                        '--input', os.path.join(image_directory, filename), 
                        '--order', "2",
                        '--output', os.path.join(input_field_directory, filename.split('.')[0] + '.npy')])
        end = time.time()
        print('Generating structure tensor in C for grid size', filename.split('.')[0].split('_')[1], 'took', end - start, 'seconds.')
    
    # compare the results of the C sturcture tensor and Python code
    print("\n\n--------Comparing C/Python Structure Tensor Results--------")
    c_structuretensors = []
    for ti in range(len(input_filenames)):
        c_structuretensors.append(np.load(os.path.join(input_field_directory, input_filenames[ti].split('.')[0] + '.npy')))
        error = tensor_field_difference(c_structuretensors[ti], python_structuretensors[ti], 1, visualize, input_filenames[ti].split('.')[0])
        print("Grid test ( " + input_filenames[ti].split('.')[0] + " ):  error = " + str(error))
        
    return python_structuretensors, c_structuretensors, input_filenames

def run_field_test(data_directory, visualize=False):
    
    fields = []
    fields.append(tv.generate_stick_field(0, 1, 100))
    fields.append(tv.generate_stick_field(1, 0, 100))
    fields.append(tv.generate_stick_field(1, 1, 100))
    
    input_filenames = []
    
    for i in range(len(fields)):
        input_filenames.append('stick_field_' + str(i) + '.npy')
        np.save(os.path.join(data_directory, input_filenames[i]), fields[i].astype(np.float32))
        
    return fields, input_filenames
    

def main():
    # create a directory to store the raw image data    
    print("Creating test image directory...")
    data_directory = "test_data"
    input_field_directory = os.path.join(data_directory, "input_fields")
    output_field_directory = os.path.join(data_directory, "output_fields")
    if os.path.exists(data_directory):
        shutil.rmtree(data_directory)
    os.makedirs(data_directory)
    os.makedirs(input_field_directory)
    os.makedirs(output_field_directory)
    print("Created test image directory.")

    # run the grid test
    python_structuretensors, _, input_filenames = run_grid_test(data_directory, input_field_directory)
    # python_structuretensors, input_filenames = run_field_test(data_directory, True)
    
    
    sigma = 10
    iterations = 1
    cuda = -1
    python_votefields, c_votefields = python_and_c_tensorvote(input_filenames, 
                                                              python_structuretensors, sigma, iterations, cuda, input_field_directory, output_field_directory)
    
    # # compare the tensor fields
    print('Comparing tensor fields...')
    for i in range(len(python_votefields)):
        difference = tensor_field_difference(python_votefields[i], c_votefields[i], 1, visualize=True, name = input_filenames[i].split('.')[0])
        print("Test case " + input_filenames[i].split('.')[0] + " error = " + str(difference))


if __name__ == "__main__":
    main()