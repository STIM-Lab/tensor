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

def plot_superquadric(vertices, title):
    fig, ax = plt.subplots()

    ax.plot(vertices[:, 0], vertices[:, 1], 'b-')
    ax.fill(vertices[:, 0], vertices[:, 1], 'b', alpha=0.3)
    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    plt.show()

# Example eigenvalues (lengths of the principal axes)
eigvals = np.array([3, 2])  # l0 = major axis, l1 = minor axis

# Generate and plot superquadric glyphs for different gamma values
gammas = [0.5, 1.0, 2.0]
size = 1.0

for gamma in gammas:
    vertices = generate_superquadric_glyphs(eigvals, gamma, size)
    plot_superquadric(vertices, f'Superquadric Glyph with gamma={gamma}')