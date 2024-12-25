import numpy as np
import matplotlib.pyplot as plt

# Part a: Generate 100 numbers from a uniform distribution
x = np.random.uniform(-10, 10, 100)

# Part b: setting the seed
np.random.seed(84)

# Part c: Generate y data using the specified formula
y = 3 * x**2 - 6 * x + 5

# Part d: Plot x vs y
plt.scatter(x, y, alpha=0.5, color='blue')
plt.title('Scatter plot of x vs y (y = 3x^2 - 6x + 5)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Part e: Add noise to the y data
noise = np.random.normal(0, 1, 100)
y_noisy = 12 * x - 4 + noise

# Part f: Replot with noise
plt.scatter(x, y_noisy, alpha=0.5, color='red')
plt.title('Scatter plot of x vs y with noise (y = 12x - 4 + noise)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

