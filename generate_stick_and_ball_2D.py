import matplotlib.pyplot as plt
import numpy as np

def construct_matrix_from_eigenvalues(e1, e2):
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    A = e1 * np.outer(v1, v1) + e2 * np.outer(v2, v2)
    
    return A

def draw_ellipse(ax, center, width, height, angle, color='blue'):
    ellipse = plt.matplotlib.patches.Ellipse(center, width, height, angle=angle, edgecolor=color, facecolor='none')
    ax.add_patch(ellipse)

def generate_stick_and_ball_tensors(eigenvalues_array):
    fig, ax = plt.subplots()
    
    n = len(eigenvalues_array)
    ax.set_xlim(-10, 10 * n)
    ax.set_ylim(-10, 10)
    
    for i, (eigenvalue1, eigenvalue2) in enumerate(eigenvalues_array):
        center = (10 * i, 0)
        
        matrix = construct_matrix_from_eigenvalues(eigenvalue1, eigenvalue2)
        
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        width = 2 * np.sqrt(eigenvalues[1])  # Major axis
        height = 2 * np.sqrt(eigenvalues[0])  # Minor axis
        
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        
        draw_ellipse(ax, center, width, height, angle, color='blue')
    
    ax.set_aspect('equal')
    ax.grid(False)
    plt.show()


# Example array of eigenvalue tuples transitioning from stick to ball
eigenvalues_array = [
    (11, 1),
    (10, 2),
    (9, 3),
    (8, 4),
    (7, 5),
    (6, 6)
]

generate_stick_and_ball_tensors(eigenvalues_array)