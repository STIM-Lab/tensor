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

def run_vote_field_test(x, y, N, sigma, cuda, visualize=False):
    T = tv.generate_stick_field(x, y, N).astype(np.float32)
    c_filename = "stick_" + str(x) + "_" + str(y) + "_" + str(N) + ".npy"
    np.save("test_data/input_fields/" + c_filename, T)
    subprocess.run(['./tensorvote.exe', 
                    '--input', "test_data/input_fields/" + c_filename, 
                    '--output', "test_data/output_fields/" + "c_" + c_filename, 
                    '--sigma', str(sigma), 
                    '--cuda', str(cuda)])
    TV_C = np.load("test_data/output_fields/" + "c_" + c_filename)
    TV_Python = tv.iterative_vote(T, sigma, 1)[-1]
    np.save("test_data/output_fields/p_" + c_filename, TV_Python)
    tensor_field_difference(TV_C, TV_Python, 1, visualize, c_filename)

def run_vote_test(input_filenames, python_structuretensors, sigma, iterations, cuda, input_directory, output_directory):
    
    python_tensorvote_fields = []
    print('Running Python tensorvote...')
    for i in range(len(python_structuretensors)):
        start = time.time()
        
        python_tensorvote_fields.append(tv.iterative_vote(python_structuretensors[i], sigma, iterations)[-1])
        end = time.time()
        print('Running tensorvote in Python for grid size', python_structuretensors[i].shape[0], 'took', end - start, 'seconds.')

        tv_filename = 'p_' + input_filenames[i].split('.')[0] + '.npy'
        np.save(os.path.join(output_directory, tv_filename), (python_tensorvote_fields[i]).astype(np.float32))
        print('Output saved as ' + tv_filename)

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
        
    #c_tensorvote_fields = []
    print("\n\n--------Comparing C/Python Tensor Voting Results--------")
    for filename in input_filenames:
        c_vote = np.load(os.path.join(output_directory, 'c_' + filename.split('.')[0] + '.npy'))
        p_vote = np.load(os.path.join(output_directory, 'p_' + filename.split('.')[0] + '.npy'))
        error = tensor_field_difference(c_vote, p_vote, 1, False, filename.split('.')[0])
        print("Grid test ( " + filename.split('.')[0] + " ):  error = " + str(error))
        
    # return python_tensorvote_fields, c_tensorvote_fields
        
def tensor_field_difference(true_field, test_field, band=1, visualize=False, name = ""):
    epsilon = 0.0001
    if true_field.shape != test_field.shape:
        raise ValueError("The two fields must have the same shape") 
    
    shape = true_field.shape
    
    absolute_error = np.abs(true_field - test_field)
    relative_error = absolute_error / (np.abs(true_field) + epsilon)
    max_relative_error = np.max(np.max(relative_error, 3), 2)
    
    summed_squared_error = np.zeros((shape[0], shape[1]))
    for i in range(band, shape[0]-band):
        for j in range(band, shape[1]-band):
            squared_differences = (true_field[i][j] - test_field[i][j])**2
            summed_squared_error[i][j] += np.sum(squared_differences)
    if(visualize):
        plt.figure()
        plt.subplot(2, 3, 1)
        tv.visualize(true_field)
        #plt.title("First Field")
        
        plt.subplot(2, 3, 2)
        tv.visualize(test_field)
        #plt.title("Second Field")
        
        plt.subplot(2, 3, 3)
        plt.imshow(summed_squared_error)
        plt.title("Sum Squared Error ")
        plt.colorbar()
        
        plt.subplot(2, 3, 4)
        plt.imshow(relative_error[:, :, 0, 0])
        plt.title("Relative Error in V[0, 0] ")
        plt.colorbar()
        
        plt.subplot(2, 3, 5)
        plt.imshow(relative_error[:, :, 0, 1])
        plt.title("Relative Error in V[0, 1]")
        plt.colorbar()
        
        plt.subplot(2, 3, 6)
        plt.imshow(relative_error[:, :, 1, 1])
        plt.title("Relative Error in V[1, 1]")
        plt.colorbar()

        plt.show()
         
    return np.max(summed_squared_error)

def run_grid_test(data_directory, input_field_directory, visualize=False):
    
    image_directory = os.path.join(data_directory, "images")
    
    os.makedirs(image_directory)
    
    # generate the grids used as input to both algorithms
    input_data = []
    #input_data.append(axis_grid_2d(3, 2, 1, 0.0))
    input_data.append(axis_grid_2d(100, 4, 2, 0.0))
    input_data.append(axis_grid_2d(200, 4, 4, 0.0))
    input_data.append(axis_grid_2d(300, 4, 6, 0.0))
    input_data.append(axis_grid_2d(400, 4, 8, 0.0))
    
    
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
    fields.append(tv.generate_stick_field(0, 1, 100).astype(np.float32))
    fields.append(tv.generate_stick_field(1, 0, 100).astype(np.float32))
    fields.append(tv.generate_stick_field(1, 1, 100).astype(np.float32))
    
    input_filenames = []
    
    for i in range(len(fields)):
        input_filenames.append('stick_field_' + str(i) + '.npy')
        np.save(os.path.join(data_directory, input_filenames[i]), fields[i].astype(np.float32))
        
    return fields, input_filenames

def display_fields(x, y, N=21, sigma_max=10, sigma_min=0):
    x = np.linspace(-N/2, N/2, N)
    X, Y = np.meshgrid(x, x)



    plt.subplot(2, 3, 1)
    STICK = tv.stickfield2(0, 1, X, Y, sigma_max, sigma_max)
    np.save("stick_uniform.npy", STICK.astype(np.float32))
    tv.visualize(STICK, "Uniform Stick Tensor", fontsize=22)
    plt.subplot(2, 3, 2)
    STICK = tv.stickfield2(0, 1, X, Y, sigma_max, sigma_min)
    np.save("stick_contour.npy", STICK.astype(np.float32))
    tv.visualize(STICK, "Contour Stick Tensor", fontsize=22)
    plt.subplot(2, 3, 3)
    STICK = tv.stickfield2(0, 1, X, Y, sigma_min, sigma_max)
    np.save("stick_layer.npy", STICK.astype(np.float32))
    tv.visualize(STICK, "Layer Stick Tensor", fontsize=22)

    plt.subplot(2, 3, 4)
    PLATE = tv.platefield2(X, Y, sigma_max, sigma_max)
    np.save("plate_uniform.npy", PLATE.astype(np.float32))
    tv.visualize(PLATE, "Uniform Plate Tensor", fontsize=22)
    plt.subplot(2, 3, 5)
    PLATE = tv.platefield2(X, Y, sigma_max, sigma_min)
    np.save("plate_contour.npy", PLATE.astype(np.float32))
    tv.visualize(PLATE, "Contour Plate Tensor", fontsize=22)
    plt.subplot(2, 3, 6)
    PLATE = tv.platefield2(X, Y, sigma_min, sigma_max)
    np.save("plate_layer.npy", PLATE.astype(np.float32))
    tv.visualize(PLATE, "Layer Plate Tensor", fontsize=22)
    

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
    print("\n\n\n--------------Processing Structure Tensor Fields--------------")
    python_structuretensors, _, input_filenames = run_grid_test(data_directory, input_field_directory)
    
    
    sigma = 2
    iterations = 1
    cuda = -1
    #run_vote_field_test(1, 0, 5, 2, -1, False)
    
    print("\n\n\n--------------Processing Tensor Voting Fields--------------")
    
    run_vote_test(input_filenames, 
                    python_structuretensors, 
                    sigma, 
                    iterations, 
                    cuda, 
                    input_field_directory, 
                    output_field_directory)
    
    # # compare the tensor fields
    #print('Comparing tensor fields...')
    #for i in range(len(python_votefields)):
    #    difference = tensor_field_difference(python_votefields[i], c_votefields[i], 1, visualize=True, name = input_filenames[i].split('.')[0])
    #    print("Test case " + input_filenames[i].split('.')[0] + " error = " + str(difference))


#if __name__ == "__main__":
#    main()

#T = tv.testfield(1, 1, 75, 20)
#tv.visualize(T)

sigma = 1
iterations = 3

G = axis_grid_2d(100, 3, 2, 1)
plt.imshow(G)

S = st.hessian(G, 1)
plt.figure()
tv.visualize(S)
TV = tv.iterative_stick2(S, sigma, iterations, 1)
plt.figure()
tv.visualize(TV[iterations])
np.save("votetest.npy", TV[1].astype(np.float32))
#display_fields(1, 1, N=21, sigma_max=10, sigma_min=7)


#plt.imshow(G)

# N = 21
# R = 21
# sigma = 10
# dtheta = np.pi / N

# T = np.zeros((R, R, 2, 2))
# for n in range(N):
#     x = np.cos(dtheta * n)
#     y = np.sin(dtheta * n)
    
#     T = T + tv.testfield(x, y, R, sigma)
#     print(n)

# np.save("circle_field.npy", T.astype(np.float32))
# tv.visualize(T, mode="eccentricity")
# TV = tv.iterative_vote(T, 1, 1)[-1]
# N = 5
# sigma = 2
# VF = tv.testfield(1, 0, N, sigma)
# T = tv.generate_stick_field(1, 0, N)
# VF = tv.iterative_vote(T, sigma, 1)[-1]

# tv.visualize(VF)