# 2.4. Calculus

# limit procedure
# differential calculus = tell us how to increase or decrease a function’s value by manipulating its arguments
# comes in handy for the optimization problems that we face in deep learning, where we repeatedly update our parameters in order to decrease the loss function. 
# optimization addresses how to fit our models to training data

import numpy as np
from d2l import torch as d2l
from matplotlib_inline import backend_inline

# 2.4.1. Derivatives and Differentiation
# a derivative is the rate of change in a function with respect to changes in its arguments. 
# derivatives can tell us how rapidly a loss function would increase or decrease were we to 
# increase or decrease each parameter by an infinitesimally small amount.

# limit tells us what happens to the value of an expression as a specified variable approaches a particular value
# tells us what the ratio between a perturbation and the change in the function value f(x+h)-f(x) converges to as we shrink its size to zero.

# We can interpret the derivative f'(x) as the instantaneous rate of change of f(x) with respect to x

# define some function
def f(x):
    return 3 * x ** 2 - 4 * x

# setting x=1, f'(x) approaches 2 as h approaches 0
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')

# 2.4.2. Visualization Utilities

def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    if legend is None:
        legend = []
    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# plot the function and its tangent line at y = 2x - 3 at x = 1, where the coefficient 
# is the slope of the tangent line.
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
