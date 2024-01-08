import numpy as np
import matplotlib.pyplot as plt

charges1 = [(0.5, 0.6, 10), (0.5, 0.4, -10)]
charges2 = [(0.4, 0.6, 10), (0.4, 0.4, -10), (0.6, 0.6, -10), (0.6, 0.4, 10)]
grid_size = 1
x_vals, y_vals = np.meshgrid(np.linspace(0, grid_size, 50), np.linspace(0, grid_size, 50)) 

def calculate_electric_field(charges, grid_size):
    """
    Calculate the electric field on a two-dimensional grid numerically.

    Parameters:
    - charges: List of charges. Each charge is a tuple (x, y, strength).
    - grid_size: Size of the grid (e.g., 100 for a 100x100 grid).

    Returns:
    - Ex, Ey: Components of the electric field on the grid.
    """
    Ex = np.zeros(x_vals.shape)
    Ey = np.zeros(x_vals.shape)

    for charge in charges:
        x, y, strength = charge
        r_squared = (x_vals - x)**2 + (y_vals - y)**2
        r = np.sqrt(r_squared)
        r3 = r**3 + 1e-6  # Avoiding division by zero
        Ex += strength * (x_vals - x) / r3
        Ey += strength * (y_vals - y) / r3

    return Ex, Ey

def plot_electric_field_and_charges(Ex, Ey, charges):
    """
    Plot the electric field with arrows and charges as circles.

    Parameters:
    - Ex, Ey: Components of the electric field on the grid.
    - charges: List of charges. Each charge is a tuple (x, y, strength).
    """
    plt.figure(figsize=(8, 8))

    # Calculate the total electric field strength for normalization
    total_field_strength = np.sqrt(Ex**2 + Ey**2)

    # Normalize the field vectors to have the same length
    Ex_normalized = Ex / total_field_strength
    Ey_normalized = Ey / total_field_strength

    # Plot arrows with colormap based on the field strength
    plt.quiver(x_vals, y_vals, Ex_normalized, Ey_normalized, scale=50, scale_units='width',
               norm=plt.Normalize(), alpha=0.7)

    # Charges as circles
    for i, charge in enumerate(charges):
        x, y, strength = charge
        color = 'red' if strength > 0 else 'blue'
        plt.scatter(x, y, s=400, c=color, marker='o', edgecolors='black', label=f'Charge {i + 1}')

    #plt.title('Electric Field and Charges')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.colorbar()
    plt.xlim(0.2, 0.8)
    plt.ylim(0.2, 0.8)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.grid(False)
    plt.savefig("C:/Users/toni-/OneDrive/Alt/Desktop/Praktikum - Übertrag/PPD/Versuche/Nanoplasmonik/Bilder/c2.pdf", bbox_inches="tight")

# Example: Create a charge distribution with two charges
# Ex, Ey = calculate_electric_field(charges1, grid_size)
# plot_electric_field_and_charges(Ex, Ey, charges1)

Ex, Ey = calculate_electric_field(charges2, grid_size)
plot_electric_field_and_charges(Ex, Ey, charges2)