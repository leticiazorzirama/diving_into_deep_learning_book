# 2.5. Automatic Differentiation

# ...often shortened to autograd
# As we pass data through each successive function, the 
# framework builds a computational graph that tracks how 
# each value depends on others. To calculate derivatives, 
# automatic differentiation works backwards through this 
# graph applying the chain rule. The computational algorithm 
# for applying the chain rule in this fashion is called 
# backpropagation.

import torch

# 2.5.1. A Simple Function

# Assign x an initial value
x = torch.arange(4.0)

x.requires_grad_(True)
print(x.grad)

# Create the function
y = 2 * torch.dot(x, x) 
print(y)

# Take the gradient of y with respect to x 
# by calling its backward method
y.backward()
print(x.grad)

# gradient should be 4
print(x.grad == 4 * x)

# Necessary to reset the gradient buffer
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# 2.5.2. Backward for Non-Scalar Variables

# Jacobian matrix = contains the partial derivatives of each
# component of a y vector with respect to each component of a x vector

# While Jacobians do show up in some advanced machine learning techniques,
# more commonly we want to sum up the gradients of each component of 
# y with respect to the full vector x, yielding a vector of the same 
# shape as x. For example, we often have a vector representing the 
# value of our loss function calculated separately for each example 
# among a batch of training examples. Here, we just want to sum up 
# the gradients computed individually for each example.

# Because deep learning frameworks vary in how they interpret gradients of non-scalar
# tensors, PyTorch takes some steps to avoid confusion. Invoking backward on a non-scalar 
# elicits an error unless we tell PyTorch how to reduce the object to 
# a scalar. More formally, we need to provide some vector such that backward will compute 
# the transpose vector of the partial derivatives rather than just one partial derivatives. 
# This next part may be confusing, but for reasons that will become clear 
# later, this argument (representing v) is named gradient. 

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
print(x.grad)

# 2.5.3. Detaching Computation

# move some calculations outside of the recorded computational graph.

# suppose we have z = x * y and y = x * x
# we want to focus on the direct influence of x on z rather than the influence conveyed via y

# create a new variable u that takes the same value as y but whose provenance 
# (how it was created) has been wiped out. 
# this procedure detaches y’s ancestors from the graph leading to z
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

# the computational graph leading to y persists and thus we can calculate the gradient of y with respect to x
z.sum().backward()
print(x.grad == u)

# 2.5.4. Gradients and Python Control Flow
# automatic differentiation is that even if building the computational 
# graph of a function required passing through a maze of Python control 
# flow (e.g., conditionals, loops, and arbitrary function calls), we can 
# still calculate the gradient of the resulting variable

# some function
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# TO UND
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# TO UND
print(a.grad == d / a)
