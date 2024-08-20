import matplotlib.pyplot as plt
import numpy as np

def signpow(x, exponent):
    return np.sign(x) * np.abs(x) ** exponent

def sq_vertex(alpha, beta, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x = signpow(cos_theta, beta)
    y = signpow(sin_theta, alpha)
    return np.array([x, y])

def generate_superquadric_glyphs(eigvals, gamma, size):
    l0, l1 = eigvals

    theta = np.linspace(0, 2 * np.pi, 100)

    alpha = (1 - (l1 / (l0 + l1))) ** gamma
    beta = (1 - (l0 / (l0 + l1))) ** gamma

    vertices = np.array([sq_vertex(alpha, beta, t) for t in theta])
    vertices[:, 0] *= l0 * size
    vertices[:, 1] *= l1 * size

    return vertices

def calculate_eccentricity(eigvals):
    l0, l1 = eigvals
    return np.sqrt(1 - (l1**2 / l0**2))

def map_eccentricity_to_gamma(eccentricity):
    # Map the range of eccentricity [0, 1] to gamma [0.5, 2]
    return eccentricity

# draw a line of superquadrics with the given eigenvalues
def plot_superquadric_line(eigenvalues_list, gamma=None):
    size = 1.0
    fig, axes = plt.subplots(1, len(eigenvalues_list), figsize=(15, 5))

    for ax in axes:
        ax.set_axis_off()  # Remove borders

    for i, eigvals in enumerate(eigenvalues_list):
        eccentricity = calculate_eccentricity(eigvals)
        if gamma == None:
            gamma_i = map_eccentricity_to_gamma(eccentricity)
        else:
            gamma_i = gamma
        vertices = generate_superquadric_glyphs(eigvals, gamma_i, size)
        ax = axes[i]
        ax.plot(vertices[:, 0], vertices[:, 1], color='black', linestyle='-')
        ax.fill(vertices[:, 0], vertices[:, 1], color='purple', alpha=0.3)
        ax.set_aspect('equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# Generate a range of eigenvalues from stick to ball, including a perfect ball
eigenvalues_list = [np.array([3, 0.5]), np.array([3, 1]), np.array([3, 1.5]), np.array([3, 2]), np.array([3, 2.5]), np.array([3, 3])]

plot_superquadric_line(eigenvalues_list, None)
