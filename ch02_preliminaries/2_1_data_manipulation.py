# 2.1. Data Manipulation

# 2.1.1. Getting Started

import torch

x = torch.arange(12, dtype=torch.float32)
x.numel()
print(x.shape)
print(x)

X = x.reshape(3,4)
print(X)

torch.zeros((2,3,4))
torch.ones((2,3,4))
torch.randn(3,4)
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])


# 2.1.2. Indexing and Slicing

X[-1], X[1:3] # only rows
X[1, 2] # rows and columns
X[1, 2] = 17 # assign new values for one index
print(X)

X[:2, :] # slice rows and select all columns
X[:2, :] = 12 # assign new values for many indexes
print(X)

# 2.1.3. Operations 

torch.exp(x) # elementwise unary scalar operators

x, y = torch.tensor([1.0, 2, 4, 8]), torch.tensor([2, 2, 2, 2]) # elementwise binary scalar operators
                                                                # implies vectors have the same shape
x+y, x-y, x*y, x/y, x**y

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

torch.cat((X,Y), dim=0) # axis 0 = rows

torch.cat((X,Y), dim=1) # axis 1 = columns

print(X == Y) # binary tensors

X.sum() # one element tensor

# 2.1.4. Broadcasting (when shapes differ)

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))

a+b

# 2.1.5. Saving Memory

before = id(Y)
print(before)

Y = Y+X

print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z)', id(Z))
Z[:] = X+Y
print('id(Z)', id(Z))

before = id(X)
X += Y
print(id(X) == before)

# 2.1.6. Conversion to Other Python Objects
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

# 2.1.7. Summary

# 2.1.8. Exercises

# 1. Change the conditional statement X == Y to X < Y or X > Y, 
# and then see what kind of tensor you can get.

print(X)
print(Y)
print(X == Y)

print(X < Y)
print(X > Y)

print(X.size() == Y.size())

# Get boolean tensors as the result of element-wise inequality operations.
# Equal dimensions of X and Y make the operation element-wise without the need of
# broadcasting.

# 2. Replace the two tensors that operate by element in the broadcasting mechanism 
# with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

x = torch.arange(9).reshape((3,3,1))
y = torch.ones(6).reshape((3,1,2))

print(x)
print(y)

print(x+y)

# There are different rules for broadcasting with other shapes like 3-dimensional.


