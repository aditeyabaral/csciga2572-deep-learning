import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Define the input range
x = np.linspace(-5, 5, 1000)

# 1. ReLU derivative
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 2. LeakyReLU derivative
negative_slope = 0.01
def leaky_relu_derivative(x):
    return np.where(x > 0, 1, negative_slope)

# 3. Softplus derivative (d/dx Softplus(x) = sigmoid(x))
beta = 1
def softplus_derivative(x):
    return 1 / (1 + np.exp(-beta * x))

# 4. GELU derivative (approximate version)
def gelu_derivative(x):
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    cdf = 0.5 * (1 + erf(x / np.sqrt(2)))
    pdf = sqrt_2_over_pi * np.exp(-0.5 * x**2)
    return cdf + x * pdf

# Standard color for curves
curve_color = 'blue'
# Softer color for axes lines
axis_color = 'lightgray'
axis_width = 0.8

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# List of functions, titles, and y-labels
functions = [
    (relu_derivative, "ReLU'(x)", "d(ReLU)/dx"),
    (leaky_relu_derivative, f"LeakyReLU'(x) (slope={negative_slope})", "d(LeakyReLU)/dx"),
    (softplus_derivative, f"Softplus'(x) (beta={beta})", "d(Softplus)/dx"),
    (gelu_derivative, "GELU'(x)", "d(GELU)/dx")
]

# Plot each subplot
for ax, (func, title, ylabel) in zip(axes, functions):
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(x.min(), x.max())
    ax.plot(x, func(x), color=curve_color)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    # Soft axes at x=0 and y=0
    ax.axhline(0, color=axis_color, linewidth=axis_width)
    ax.axvline(0, color=axis_color, linewidth=axis_width)

plt.tight_layout()
plt.savefig("activation_function_derivatives.png", dpi=300)
plt.show()
